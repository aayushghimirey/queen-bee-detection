from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from spectrogram import CLIP_SECONDS, make_spectrogram

DATA_DIR = Path("bee_prediction_dataset")
AUDIO_DIR = DATA_DIR / "dataset"
INSPECTIONS_FILE = DATA_DIR / "annotations" / "inspections_2022.csv"
OUTPUT_DIR = DATA_DIR / "prepared"

# The position in this list is the label: 0 = queenright, 1 = queenless
CLASS_NAMES = ["queenright", "queenless"]

SAMPLE_RATE = 16000  # the hive recordings were made at 16 kHz
MAX_DAYS = 7  # only label a recording if the queen was checked within 7 days of it
CLIPS_PER_RECORDING = 10  # 2-second clips, spread evenly over each 15-minute recording
QUEENLESS_TRAIN_CLIPS = 30  # queenless training recordings are rare, so take more clips from each

rng = np.random.default_rng(42)  # same random changes every run

# Whole hives kept aside for testing, so the test shows how the model does on hives
# it has never heard. HIVE-3693 is one of only three hives with queenless recordings.
TEST_HIVES = ["HIVE-3627", "HIVE-3691", "HIVE-3693"]


def load_queen_checks():
    """Read the 2022 inspections and keep the rows where the queen was checked."""
    checks = pd.read_csv(INSPECTIONS_FILE)
    # The other rows (feeding, treatment, ...) only repeat the last queen status.
    checks = checks[checks["Category"] == "hive status"]

    return pd.DataFrame(
        {
            "hive": "HIVE-" + checks["Tag number"].astype(str),
            # Inspection times are in UTC. We match by days, so the hours don't matter.
            "check_time": pd.to_datetime(checks["Date"], format="ISO8601").dt.tz_localize(None),
            "status": checks["Action detail"],  # queenright, queenless or deadout
        }
    )


def list_recordings():
    """One row per recording, with the hive and time read from the file name.
    For example 05-06-2022_18h15_HIVE-3629.wav"""
    rows = []
    for file in sorted(AUDIO_DIR.glob("*.wav")):
        date, time, hive = file.stem.split("_")
        time = pd.to_datetime(f"{date} {time}", format="%d-%m-%Y %Hh%M")
        rows.append({"file": file.name, "hive": hive, "time": time})
    return pd.DataFrame(rows)


def label_recordings():
    """Give each recording the queen status from the nearest check on the same hive."""
    recordings = list_recordings().sort_values("time")
    checks = load_queen_checks().sort_values("check_time")

    labeled = pd.merge_asof(
        recordings,
        checks,
        left_on="time",
        right_on="check_time",
        by="hive",
        direction="nearest",
        tolerance=pd.Timedelta(days=MAX_DAYS),
    )

    # Drop recordings with no check close enough, and those nearest a "deadout" check.
    # This also drops HIVE-3629's recordings after it was marked dead on 21 Sep 2022.
    labeled = labeled[labeled["status"].isin(CLASS_NAMES)]
    labeled["split"] = np.where(labeled["hive"].isin(TEST_HIVES), "test", "train")
    return labeled


def cut_clips(file, count):
    """Cut `count` 2-second clips spread evenly across one recording."""
    audio, _ = librosa.load(AUDIO_DIR / file, sr=SAMPLE_RATE)
    clip_length = SAMPLE_RATE * CLIP_SECONDS
    starts = np.linspace(0, len(audio) - clip_length, count).astype(int)
    return [audio[start : start + clip_length] for start in starts]


def augment(spectrogram):
    """Make two changed copies of a spectrogram, for more queenless training examples."""
    # 1. Play it backwards. A hive's hum sounds much the same in either direction.
    backwards = spectrogram[:, ::-1]

    # 2. Blank out a random band of pitches and a random moment in time,
    #    so the model can't depend on one small detail.
    masked = spectrogram.copy()
    pitch = rng.integers(0, masked.shape[0] - 15)
    moment = rng.integers(0, masked.shape[1] - 10)
    masked[pitch : pitch + 15, :] = masked.min()
    masked[:, moment : moment + 10] = masked.min()

    return [backwards, masked]


def save_split(recordings, split):
    """Save the spectrograms and labels of the "train" or "test" recordings to one file."""
    X, y = [], []

    for i, (file, status) in enumerate(zip(recordings["file"], recordings["status"])):
        if i % 100 == 0:
            print(f"{split}: {i}/{len(recordings)} recordings")

        # Queenless training recordings are rare, so each one gives more clips,
        # and every clip gets two changed copies. The test set stays as recorded.
        extra = split == "train" and status == "queenless"
        clip_count = QUEENLESS_TRAIN_CLIPS if extra else CLIPS_PER_RECORDING

        for clip in cut_clips(file, clip_count):
            spectrogram = make_spectrogram(clip, SAMPLE_RATE)
            versions = ([spectrogram] + augment(spectrogram)) if extra else [spectrogram]
            X.extend(versions)
            y.extend([CLASS_NAMES.index(status)] * len(versions))

    # The model wants one extra axis at the end: (clips, height, width, 1)
    np.savez(OUTPUT_DIR / f"{split}.npz", X=np.array(X)[..., np.newaxis], y=np.array(y))


def prepare_dataset():
    labeled = label_recordings()

    # A list of which recording got which label, to check by hand.
    OUTPUT_DIR.mkdir(exist_ok=True)
    labeled.to_csv(OUTPUT_DIR / "recordings.csv", index=False)
    print(labeled.groupby(["split", "status"]).size().unstack())

    for split in ["train", "test"]:
        save_split(labeled[labeled["split"] == split], split)

    print(f"Dataset saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    prepare_dataset()
