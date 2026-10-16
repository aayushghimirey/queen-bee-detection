from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from bee_prediction_data_prepration import CLIPS_PER_RECORDING, SAMPLE_RATE, list_our_recordings
from spectrogram import CLIP_SECONDS

OUTPUT_DIR = Path("hive_change_dataset")
FINGERPRINTS_FILE = OUTPUT_DIR / "fingerprints.npz"

RECENT_DAYS = (14, 0)  # a hive's recent "normal": its recordings from 14 days before up to now
LONG_AGO_DAYS = (75, 45)  # how it sounded "two months ago": its recordings from 75 to 45 days before
MIN_PAST_RECORDINGS = 10  # how many past recordings we need to know what a hive sounded like
MIN_HIVES_SAME_DAY = 3  # hives recorded on a day, to take away the change they all had

CHANGE_DB = 5  # sudden change: the hive sounds this many dB different from its last 14 days
DECLINE_DB = 10  # slow decline: the hive got this many dB quieter overall than two months ago
SAME_DECLINE_DAYS = 14  # slow-decline days less than 14 days apart count as one decline (one alert)
# Jumps bigger than this are too big for bees: check the microphone.
# The biggest real change seen in 2022 was 14 dB (HIVE-3690 after its split).
MICROPHONE_DB = 15


def fingerprint(path):
    """The average sound of one recording: how loud each of 128 pitch bands is, in dB.

    Measured on 10 two-second clips spread over the recording. Unlike make_spectrogram,
    this keeps the real loudness, because a hive getting quieter is one of the changes we want to see.
    """
    length = librosa.get_duration(path=path)
    power = []
    for start in np.linspace(0, length - CLIP_SECONDS, CLIPS_PER_RECORDING):
        clip, _ = librosa.load(path, sr=SAMPLE_RATE, offset=start, duration=CLIP_SECONDS)
        power.append(librosa.feature.melspectrogram(y=clip, sr=SAMPLE_RATE).mean(axis=1))
    return librosa.power_to_db(np.mean(power, axis=0))


def load_fingerprints(recordings):
    """Fingerprint every recording. This reads all the audio (a few minutes),
    so the result is saved and reused next time."""
    names = np.array([path.name for path in recordings["path"]])

    # Reuse the saved fingerprints, unless recordings were added or removed since.
    if FINGERPRINTS_FILE.exists():
        saved = np.load(FINGERPRINTS_FILE)
        if np.array_equal(saved["names"], names):
            return saved["fingerprints"]

    fingerprints = []
    for i, path in enumerate(recordings["path"]):
        if i % 500 == 0:
            print(f"fingerprints: {i}/{len(recordings)} recordings")
        fingerprints.append(fingerprint(path))

    OUTPUT_DIR.mkdir(exist_ok=True)
    np.savez(FINGERPRINTS_FILE, names=names, fingerprints=fingerprints)
    return np.array(fingerprints)


def overall_loudness(fingerprints):
    """How loud each recording is overall, as one number in dB: the power of all pitch bands together.

    This follows the bees' volume better than averaging the bands' dB, because the quiet
    bands stay near the background noise whatever the bees do.
    """
    return 10 * np.log10(np.mean(10 ** (fingerprints / 10), axis=1, keepdims=True))


def difference_from_past(recordings, fingerprints, days):
    """For each recording: how it differs from the same hive's past recordings, in dB per pitch band.

    days = (from, to) picks the past recordings, e.g. (14, 0) means from 14 days before up to now.
    The difference is against the middle (median) of them. It stays empty (NaN) when there
    aren't enough past recordings to know what the hive sounded like.
    """
    days_from, days_to = days
    differences = np.full(fingerprints.shape, np.nan)

    for i, recording in enumerate(recordings.itertuples()):
        past = (
            (recordings["hive"] == recording.hive)
            & (recordings["time"] >= recording.time - pd.Timedelta(days=days_from))
            & (recordings["time"] < recording.time - pd.Timedelta(days=days_to))
        ).to_numpy()

        if past.sum() >= MIN_PAST_RECORDINGS:
            differences[i] = fingerprints[i] - np.median(fingerprints[past], axis=0)

    return differences


def per_day(recordings, differences):
    """One difference per hive per day: the middle (median) of that day's recordings,
    so one odd recording can't set off an alert on its own."""
    table = pd.DataFrame(differences)
    table["hive"] = recordings["hive"]
    table["date"] = recordings["date"]
    return table.groupby(["hive", "date"]).median().dropna()


def minus_other_hives(days):
    """Take away the change all hives had on the same day (weather, a feeding round, the season),
    so what's left is each hive's own change.

    Only done on days when at least 3 hives were recorded; on other days the hive's change stays as it is.
    """
    all_hives = days.groupby(level="date").transform("median")
    hives_that_day = days.groupby(level="date")[0].transform("size")
    enough = (hives_that_day >= MIN_HIVES_SAME_DAY).to_numpy()

    own = days.copy()
    own[enough] = days[enough] - all_hives[enough]
    return own


def alert_for(day):
    """Which alert a hive gets on one day, if any."""
    if day.raw_change_db >= MICROPHONE_DB:
        return "check microphone"  # too big a jump to come from the bees
    if day.change_db >= CHANGE_DB:
        return "sudden change"
    if day.decline_db <= -DECLINE_DB:
        return "slow decline"
    return ""


def one_alert_per_decline(alerts):
    """A slow decline sets off an alert day after day. Turn each hive's run of slow-decline
    days into one alert, from its first day to its last. A run ends when no slow-decline day
    follows within 14 days. Sudden-change and microphone alerts stay one per day.

    The scores are the strongest seen during the alert.
    """
    alerts = alerts.reset_index().sort_values(["hive", "alert", "date"])
    dates = pd.to_datetime(alerts["date"])

    starts_new_alert = (
        (alerts["alert"] != "slow decline")
        | (alerts["hive"] != alerts["hive"].shift())
        | (alerts["alert"] != alerts["alert"].shift())
        | (dates.diff() > pd.Timedelta(days=SAME_DECLINE_DAYS))
    )

    return (
        alerts.groupby(starts_new_alert.cumsum())
        .agg(
            hive=("hive", "first"),
            alert=("alert", "first"),
            first_day=("date", "min"),
            last_day=("date", "max"),
            days=("date", "size"),
            raw_change_db=("raw_change_db", "max"),
            change_db=("change_db", "max"),
            decline_db=("decline_db", "min"),
        )
        .sort_values(["hive", "first_day"])
    )


def detect_changes():
    recordings = list_our_recordings().sort_values("time").reset_index(drop=True)
    recordings["date"] = recordings["time"].dt.date
    fingerprints = load_fingerprints(recordings)

    recent = per_day(recordings, difference_from_past(recordings, fingerprints, RECENT_DAYS))
    long_ago = per_day(recordings, difference_from_past(recordings, overall_loudness(fingerprints), LONG_AGO_DAYS))
    recent_own = minus_other_hives(recent)

    # Each score is an average over the 128 pitch bands, in dB.
    days = pd.DataFrame(
        {
            "recordings": recordings.groupby(["hive", "date"]).size(),
            "raw_change_db": recent.abs().mean(axis=1),  # before taking away the other hives' change
            "change_db": recent_own.abs().mean(axis=1),  # how different from its last 14 days
            "louder_db": recent_own.mean(axis=1),  # above 0: louder than its last 14 days, below 0: quieter
            "decline_db": minus_other_hives(long_ago)[0],  # below 0: quieter overall than two months ago
        }
    ).round(1)
    days["alert"] = [alert_for(day) for day in days.itertuples()]

    OUTPUT_DIR.mkdir(exist_ok=True)
    days.to_csv(OUTPUT_DIR / "day_scores.csv")

    alerts = one_alert_per_decline(days[days["alert"] != ""])
    alerts.to_csv(OUTPUT_DIR / "alerts.csv", index=False)

    print(f"\n{len(alerts)} alerts:", alerts["alert"].value_counts().to_dict())
    print(alerts.to_string(index=False))
    print(f"\nAlerts saved to {OUTPUT_DIR / 'alerts.csv'}, every day's scores to {OUTPUT_DIR / 'day_scores.csv'}")


if __name__ == "__main__":
    detect_changes()
