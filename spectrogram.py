import librosa
import numpy as np

SAMPLE_RATE = 22050
CLIP_SECONDS = 2  # every clip is 2 seconds long
CLIP_LENGTH = SAMPLE_RATE * CLIP_SECONDS


def make_spectrogram(audio, sample_rate=SAMPLE_RATE):
    """Turn a clip of audio into a spectrogram (a 2D picture of the sound)."""
    # Pad with silence or cut, so every clip is exactly 2 seconds.
    audio = librosa.util.fix_length(audio, size=sample_rate * CLIP_SECONDS)
    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    return librosa.power_to_db(mel, ref=np.max)
