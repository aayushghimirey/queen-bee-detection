from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from spectrogram import CLIP_SECONDS, make_spectrogram

DATA_DIR = Path("bee_prediction_dataset")
AUDIO_DIR = DATA_DIR / "dataset"
INSPECTIONS_FILE = DATA_DIR / "annotations" / "inspections_2022.csv"
# Public "To bee or not to bee" dataset (Nolasco & Benetos 2018, zenodo.org/records/1321278)
BEE_TOGETHER_DIR = DATA_DIR / "bee_together" / "all_files"
OUTPUT_DIR = DATA_DIR / "prepared"

# The position in this list is the label: 0 = queenright, 1 = queenless
CLASS_NAMES = ["queenright", "queenless"]

SAMPLE_RATE = 16000  # our recordings are 16 kHz; bee_together recordings are turned into 16 kHz too
MAX_DAYS = 7  # only label one of our recordings if the queen was checked within 7 days of it
CLIPS_PER_RECORDING = 10  # 2-second clips, spread evenly over each of our 15-minute recordings
QUEENLESS_TRAIN_CLIPS = 30  # our queenless training recordings are rare, so take more clips from each

rng = np.random.default_rng(42)  # same random choices every run

# Whole hives kept aside for testing, so the test shows how the model does on hives
# it has never heard. HIVE-3693 and Hive3 are in here because they have both
# queenright and queenless recordings.
TEST_HIVES = ["HIVE-3627", "HIVE-3691", "HIVE-3693", "Hive3"]


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


def list_our_recordings():
    """One row per recording, with the hive and time read from the file name.
    For example 05-06-2022_18h15_HIVE-3629.wav"""
    rows = []
    for path in sorted(AUDIO_DIR.glob("*.wav")):
        date, time, hive = path.stem.split("_")
        time = pd.to_datetime(f"{date} {time}", format="%d-%m-%Y %Hh%M")
        rows.append({"path": path, "hive": hive, "time": time})
    return pd.DataFrame(rows)


def label_our_recordings():
    """Give each of our recordings the queen status from the nearest check on the same hive."""
    recordings = list_our_recordings().sort_values("time")
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
    labeled["source"] = "ours"
    labeled["bee_parts"] = None  # no noise labels for our recordings, so the whole recording is used
    return labeled


def read_bee_parts():
    """Read beeAnnotations.mlf: for each bee_together recording, the parts (start, end in
    seconds) with only bee sound. Parts with outside noise (cars, birds, talking) are "nobee".

    The file has the recording name on one line, then "start  end  bee/nobee" lines, then ".".
    """
    bee_parts = {}
    for line in (BEE_TOGETHER_DIR / "beeAnnotations.mlf").read_text().splitlines():
        cells = line.split("\t")
        if line == ".":
            continue
        if len(cells) == 3:
            start, end, label = cells
            if label == "bee":
                bee_parts[name].append((float(start), float(end)))
        else:
            name = line
            bee_parts[name] = []
    return bee_parts


def queen_status(name):
    """Read the queen status from a bee_together recording name."""
    if "NO_QueenBee" in name or "Missing Queen" in name:
        return "queenless"
    if "QueenBee" in name or "Active" in name:  # "Active" means a normal working hive
        return "queenright"
    return None  # the swarming recordings say nothing about the queen


def list_bee_together_recordings():
    """One row per bee_together recording, with its hive and queen status.
    For example "Hive1_31_05_2018_NO_QueenBee_H1_audio___15_00_00" or "CJ001 - Missing Queen - Day -  (100)"."""
    audio_files = {path.stem: path for path in BEE_TOGETHER_DIR.iterdir() if path.suffix in (".wav", ".mp3")}

    rows = []
    for name, bee_parts in read_bee_parts().items():
        status = queen_status(name)
        # 22 recordings in the label file have no audio (they aren't on Zenodo either).
        if name not in audio_files or status is None:
            continue

        hive = name.replace("_", " ").split()[0]  # Hive1, Hive3, CF003, CJ001, ...
        rows.append(
            {"path": audio_files[name], "hive": hive, "status": status, "source": "bee_together", "bee_parts": bee_parts}
        )
    return pd.DataFrame(rows)


def list_all_recordings():
    """Our labeled recordings plus the bee_together ones, each marked "train" or "test"."""
    recordings = pd.concat([label_our_recordings(), list_bee_together_recordings()], ignore_index=True)
    recordings["split"] = np.where(recordings["hive"].isin(TEST_HIVES), "test", "train")
    return recordings


def clip_count(recording):
    """How many clips to take from one recording. None means every clip."""
    if recording.source == "bee_together":
        return None  # only a few short recordings, so use all of their bee sound
    if recording.split == "train" and recording.status == "queenless":
        return QUEENLESS_TRAIN_CLIPS
    return CLIPS_PER_RECORDING


def cut_clips(path, count, bee_parts):
    """Cut 2-second clips from one recording, spread evenly.

    bee_parts lists the (start, end) seconds with only bee sound; None means the whole
    recording. count is how many clips to take; None means every clip.
    """
    audio, _ = librosa.load(path, sr=SAMPLE_RATE)  # also turns stereo into mono
    clip_length = SAMPLE_RATE * CLIP_SECONDS
    if bee_parts is None:
        bee_parts = [(0, len(audio) / SAMPLE_RATE)]

    # The start of every 2-second clip that fits inside a bee part.
    # The check against len(audio) is because a few labels run past the end of their recording.
    starts = []
    for start, end in bee_parts:
        for k in range(int((end - start) // CLIP_SECONDS)):
            first = int((start + k * CLIP_SECONDS) * SAMPLE_RATE)
            if first + clip_length <= len(audio):
                starts.append(first)

    if count is not None and count < len(starts):
        picked = np.linspace(0, len(starts) - 1, count).astype(int)
        starts = [starts[i] for i in picked]

    return [audio[first : first + clip_length] for first in starts]


def balance(X, y, hives):
    """Give both classes the same number of clips, by randomly leaving out clips of the bigger class."""
    smaller = np.bincount(y).min()
    keep = [rng.choice(np.flatnonzero(y == label), smaller, replace=False) for label in range(len(CLASS_NAMES))]
    keep = np.sort(np.concatenate(keep))
    return X[keep], y[keep], hives[keep]


def augment(spectrogram):
    """Make two changed copies of a spectrogram, for more training examples."""
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
    """Save the spectrograms, labels and hive names of the "train" or "test" recordings to one file."""
    X, y, hives = [], [], []

    for i, recording in enumerate(recordings.itertuples()):
        if i % 100 == 0:
            print(f"{split}: {i}/{len(recordings)} recordings")

        for clip in cut_clips(recording.path, clip_count(recording), recording.bee_parts):
            X.append(make_spectrogram(clip, SAMPLE_RATE))
            y.append(CLASS_NAMES.index(recording.status))
            hives.append(recording.hive)

    X, y, hives = np.array(X), np.array(y), np.array(hives)

    if split == "train":
        X, y, hives = balance(X, y, hives)

        # Two changed copies of every training clip. Both classes get them, so the
        # changes can't become a clue for one class.
        copies = []
        for spectrogram in X:
            copies.extend(augment(spectrogram))
        X = np.concatenate([X, copies])
        y = np.concatenate([y, np.repeat(y, 2)])  # each clip's label, once per copy
        hives = np.concatenate([hives, np.repeat(hives, 2)])

    print(f"{split}: {np.bincount(y)[0]} queenright and {np.bincount(y)[1]} queenless clips")

    # The model wants one extra axis at the end: (clips, height, width, 1)
    np.savez(OUTPUT_DIR / f"{split}.npz", X=X[..., np.newaxis], y=y, hive=hives)


def prepare_dataset():
    recordings = list_all_recordings()

    # A list of which recording got which label, to check by hand.
    OUTPUT_DIR.mkdir(exist_ok=True)
    recordings.to_csv(OUTPUT_DIR / "recordings.csv", index=False)
    print(recordings.groupby(["split", "hive", "status"]).size().unstack(fill_value=0))

    for split in ["train", "test"]:
        save_split(recordings[recordings["split"] == split], split)

    print(f"Dataset saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    prepare_dataset()
