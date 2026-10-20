import io
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
from scipy.io import wavfile
from fastapi import HTTPException, status

# Exact mapping between DB column names (valid SQL identifiers)
# and MSPB2 dataset feature channel names
FREQ_BAND_MAPPING: Dict[str, str] = {
    "hz_122_07": "hz_122.0703125",
    "hz_152_59": "hz_152.587890625",
    "hz_183_11": "hz_183.10546875",
    "hz_213_62": "hz_213.623046875",
    "hz_244_14": "hz_244.140625",
    "hz_274_66": "hz_274.658203125",
    "hz_305_18": "hz_305.17578125",
    "hz_335_69": "hz_335.693359375",
    "hz_366_21": "hz_366.2109375",
    "hz_396_73": "hz_396.728515625",
    "hz_427_25": "hz_427.24609375",
    "hz_457_76": "hz_457.763671875",
    "hz_488_28": "hz_488.28125",
    "hz_518_80": "hz_518.798828125",
    "hz_549_32": "hz_549.31640625",
    "hz_579_83": "hz_579.833984375",
}

DB_BAND_COLUMNS = list(FREQ_BAND_MAPPING.keys())
TARGET_FS = 15625
FFT_SIZE = 512


def validate_wav_header(file_path: Path) -> None:
    """Validate that the file begins with standard RIFF/WAVE magic bytes."""
    try:
        with file_path.open("rb") as f:
            header = f.read(12)
        if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            raise ValueError("Not a valid WAV file")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Uploaded audio is not a valid RIFF/WAVE file: {exc}",
        )


def extract_audio_features(file_path: Path) -> Dict[str, Any]:
    """
    Extracts the 16 frequency bands and acoustic summary metrics from a WAV recording
    matching the ESP32 edge sampling and MSPB2 training domain.
    """
    try:
        fs, data = wavfile.read(str(file_path))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to read WAV audio data: {exc}",
        )

    # Convert to mono if multichannel
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Normalize to [-1.0, 1.0] float
    if np.issubdtype(data.dtype, np.integer):
        max_val = float(np.iinfo(data.dtype).max)
        signal = data.astype(np.float32) / (max_val if max_val > 0 else 1.0)
    else:
        signal = data.astype(np.float32)

    duration_seconds = float(len(signal) / fs) if fs > 0 else 0.0

    # Frame processing with Hann window
    N = FFT_SIZE
    hop = N // 2
    if len(signal) < N:
        # Pad with zeros if shorter than one frame
        signal = np.pad(signal, (0, N - len(signal)))

    window = np.hanning(N)
    frames = [
        signal[i : i + N] * window for i in range(0, len(signal) - N + 1, hop)
    ]
    if not frames:
        frames = [signal[:N] * window]

    # Compute power spectra across all frames
    specs = [np.abs(np.fft.rfft(frame, n=N)) ** 2 for frame in frames]
    mean_spec = np.mean(specs, axis=0)

    # Bins 4 through 19 map to 122.07 Hz through 579.83 Hz for fs=15625, N=512
    # Convert power to relative dB
    db_power = 10.0 * np.log10(mean_spec + 1e-12)

    result: Dict[str, Any] = {
        "sample_rate": int(fs),
        "duration_seconds": round(duration_seconds, 3),
    }

    for idx, col in enumerate(DB_BAND_COLUMNS):
        bin_idx = 4 + idx
        val = float(db_power[bin_idx]) if bin_idx < len(db_power) else 0.0
        result[col] = float(np.clip(val, -100.0, 100.0))

    # Calculate acoustic summary indicators
    frame_energies = [float(np.mean(f**2)) for f in frames]
    energy_arr = np.array(frame_energies)
    noise_floor = np.percentile(energy_arr, 10) + 1e-8
    active_mask = energy_arr > (noise_floor * 2.0)

    audio_density = float(np.mean(active_mask) * 100.0)
    audio_density_ratio = float(np.mean(energy_arr) / (np.max(energy_arr) + 1e-8))
    density_variation = float(np.std(energy_arr) * 100.0)

    result["audio_density"] = round(audio_density, 3)
    result["audio_density_ratio"] = round(min(max(audio_density_ratio, 0.0), 1.0), 3)
    result["density_variation"] = round(density_variation, 3)

    return result
