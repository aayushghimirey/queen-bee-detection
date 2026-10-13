import sys
from pathlib import Path

import keras
import librosa
import numpy as np
import soundfile as sf

from bee_cricket_noise_train import MODEL_PATH
from spectrogram import CLIP_LENGTH, SAMPLE_RATE, make_spectrogram

OUTPUT_DIR = Path("clean_audio_dataset")

BEE_THRESHOLD = 0.70  # Keep clips only when bee probability >= 70%
NOISE_REDUCTION = 1.5  # Controls the strength of frequency-based noise reduction


def remove_background(bee_audio, background_audio=None):
    """Attempt to reduce background noise in audio classified as bee."""

    if not len(bee_audio):
        return bee_audio

    # If there are no background reference clips, leave the audio unchanged.
    if background_audio is None or not len(background_audio):
        print(
            "No background reference available; saving bee clips without noise reduction."
        )
        return bee_audio

    # Analyze the frequencies in the bee audio and background audio.
    bee_sound = librosa.stft(bee_audio)
    background_sound = librosa.stft(background_audio)

    bee_level = np.median(np.abs(bee_sound), axis=1, keepdims=True)
    background_level = np.median(np.abs(background_sound), axis=1, keepdims=True)

    # Reduce frequencies that appear relatively strong in the background.
    volume = 1 - NOISE_REDUCTION * (background_level / (bee_level + 1e-10)) ** 2
    volume = np.clip(volume, 0.1, 1.0)

    cleaned = librosa.istft(
        bee_sound * volume,
        length=len(bee_audio),
    )

    # Safety check: avoid keeping the filtered result if it loses too much volume.
    if np.abs(cleaned).mean() < 0.5 * np.abs(bee_audio).mean():
        print("Noise reduction was too aggressive; keeping original bee audio.")
        return bee_audio

    return cleaned


def clean_recording(input_path, output_path):
    model = keras.models.load_model(MODEL_PATH)

    audio, _ = librosa.load(input_path, sr=SAMPLE_RATE)

    if len(audio) == 0:
        print("The input recording is empty.")
        return

    # Split the recording into clips.
    clips = [audio[i : i + CLIP_LENGTH] for i in range(0, len(audio), CLIP_LENGTH)]

    # Ask the model to classify each clip.
    spectrograms = np.array([make_spectrogram(clip) for clip in clips])

    predictions = model.predict(
        spectrograms[..., np.newaxis],
        verbose=0,
    )

    # Assumes class index 0 represents Bee.
    bee_chance = predictions[:, 0]

    # Keep only clips where the bee probability is at least 70%.
    bee_clips = [
        clip for clip, chance in zip(clips, bee_chance) if chance >= BEE_THRESHOLD
    ]

    discarded = len(clips) - len(bee_clips)

    print(
        f"{len(clips)} clips analyzed: "
        f"{len(bee_clips)} kept as bee, "
        f"{discarded} discarded."
    )

    # If no bee clips are found, do nothing and save no file.
    if not bee_clips:
        print("No clips met the 70% bee threshold. No file saved.")
        return

    # Join the accepted bee clips.
    bee_audio = np.concatenate(bee_clips)

    # Attempt noise reduction only on the accepted bee audio.
    # No background clips are used as references in this version.
    cleaned_audio = remove_background(bee_audio)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    sf.write(output_path, cleaned_audio, SAMPLE_RATE)

    print(
        f"Saved {len(cleaned_audio) / SAMPLE_RATE:.1f} seconds "
        f"of bee-classified audio to {output_path}"
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python bee_cricket_noise_clean_audio.py my_recording.wav")

    input_path = Path(sys.argv[1])

    if not input_path.is_file():
        sys.exit(f"Input file not found: {input_path}")

    output_path = OUTPUT_DIR / f"{input_path.stem}_clean.wav"

    clean_recording(input_path, output_path)
