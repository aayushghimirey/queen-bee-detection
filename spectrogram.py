import librosa
import numpy as np

SAMPLE_RATE = 22050
CLIP_LENGTH = SAMPLE_RATE * 2  # every clip is 2 seconds long


def make_spectrogram(audio):
    """Turn a clip of audio into a spectrogram (a 2D picture of the sound)."""
    # Pad with silence or cut, so every clip is exactly 2 seconds.
    audio = librosa.util.fix_length(audio, size=CLIP_LENGTH)
    mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE)
    return librosa.power_to_db(mel, ref=np.max)
