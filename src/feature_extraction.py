"""
feature_extraction.py
Extracts the verified 75 physical features from a 24-hour sliding window of beehive sensor readings.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List

# The 21 raw physical sensor channels
SENSOR_CHANNELS: List[str] = [
    "temperature",
    "humidity",
    "hz_122.0703125",
    "hz_152.587890625",
    "hz_183.10546875",
    "hz_213.623046875",
    "hz_244.140625",
    "hz_274.658203125",
    "hz_305.17578125",
    "hz_335.693359375",
    "hz_366.2109375",
    "hz_396.728515625",
    "hz_427.24609375",
    "hz_457.763671875",
    "hz_488.28125",
    "hz_518.798828125",
    "hz_549.31640625",
    "hz_579.833984375",
    "audio_density",
    "audio_density_ratio",
    "density_variation",
]

# The 16 frequency bands and their center frequencies in Hz
FREQ_BANDS: List[str] = [c for c in SENSOR_CHANNELS if c.startswith("hz_")]
FREQ_VALUES: np.ndarray = np.array([float(c.replace("hz_", "")) for c in FREQ_BANDS])

# Low-frequency bands (122.07 - 244.14 Hz) associated with fanning / flight warmup
LOW_BANDS: List[str] = FREQ_BANDS[:5]
# High-frequency bands (457.76 - 579.83 Hz) associated with active buzz / piping
HIGH_BANDS: List[str] = FREQ_BANDS[11:]
# Middle fanning bands (213.62 - 274.66 Hz)
FANNING_BANDS: List[str] = FREQ_BANDS[3:6]
# Middle piping bands (396.73 - 457.76 Hz)
PIPING_BANDS: List[str] = FREQ_BANDS[9:12]


def compute_24h_features(
    window_df: pd.DataFrame, time_col: str = "published_at"
) -> Dict[str, float]:
    """
    Computes 75 engineered features from a 24-hour window DataFrame.

    Parameters:
        window_df: DataFrame containing sensor readings over a 24h period.
        time_col: Column name containing datetime timestamps.

    Returns:
        Dict[str, float] containing exactly 75 named float features.
    """
    if len(window_df) == 0:
        raise ValueError("Cannot extract features from an empty 24-hour window.")

    features: Dict[str, float] = {}

    # ---------------------------------------------------------
    # Group 1: Statistical Summary Features (21 channels x 3 = 63 features)
    # ---------------------------------------------------------
    for col in SENSOR_CHANNELS:
        if col in window_df.columns:
            s = pd.to_numeric(window_df[col], errors="coerce").dropna()
            if len(s) > 0:
                mean_val = float(s.mean())
                std_val = float(s.std(ddof=1)) if len(s) > 1 else 0.0
                range_val = float(s.max() - s.min())
            else:
                mean_val, std_val, range_val = 0.0, 0.0, 0.0
        else:
            mean_val, std_val, range_val = 0.0, 0.0, 0.0

        features[f"{col}_mean"] = mean_val
        features[f"{col}_std"] = std_val
        features[f"{col}_range"] = range_val

    # ---------------------------------------------------------
    # Group 2: Biological Thermoregulation & Stability (4 features)
    # ---------------------------------------------------------
    temp_s = pd.to_numeric(window_df["temperature"], errors="coerce").dropna()
    t_mean = features["temperature_mean"]
    t_std = features["temperature_std"]

    # 1. Temperature Coefficient of Variation (lower in strong hives)
    features["temp_cv"] = float(t_std / (abs(t_mean) + 1e-5))

    # 2. Brood Core Adherence (|mean_T - 34.5°C|)
    features["brood_temp_adherence"] = float(abs(t_mean - 34.5))

    # 3. Day vs. Night Temperature Drop
    # Night: 00:00 to 10:00 UTC (roughly 20:00 to 06:00 EDT local time)
    # Day: 10:00 to 00:00 UTC (roughly 06:00 to 20:00 EDT local time)
    if time_col in window_df.columns:
        dt = pd.to_datetime(window_df[time_col])
        day_mask = (dt.dt.hour >= 10) & (dt.dt.hour < 24)
        night_mask = ~day_mask

        day_temp = pd.to_numeric(
            window_df.loc[day_mask, "temperature"], errors="coerce"
        ).dropna()
        night_temp = pd.to_numeric(
            window_df.loc[night_mask, "temperature"], errors="coerce"
        ).dropna()

        if len(day_temp) > 0 and len(night_temp) > 0:
            features["temp_diurnal_drop"] = float(day_temp.mean() - night_temp.mean())
        else:
            features["temp_diurnal_drop"] = 0.0

        # Day vs. Night Audio Dynamism
        day_audio = pd.to_numeric(
            window_df.loc[day_mask, "audio_density"], errors="coerce"
        ).dropna()
        night_audio = pd.to_numeric(
            window_df.loc[night_mask, "audio_density"], errors="coerce"
        ).dropna()
        if len(day_audio) > 0 and len(night_audio) > 0:
            features["audio_diurnal_dynamism"] = float(
                day_audio.mean() - night_audio.mean()
            )
        else:
            features["audio_diurnal_dynamism"] = 0.0

        # 4. Temperature Linear Trend (Slope over 24 hours)
        if len(temp_s) > 3:
            time_sec = (dt.dropna() - dt.dropna().min()).dt.total_seconds().values
            t_hours = time_sec / 3600.0
            if len(t_hours) == len(temp_s) and np.std(t_hours) > 1e-3:
                slope, _ = np.polyfit(t_hours, temp_s.values, deg=1)
                features["temp_linear_trend"] = float(slope)
            else:
                features["temp_linear_trend"] = 0.0
        else:
            features["temp_linear_trend"] = 0.0
    else:
        features["temp_diurnal_drop"] = 0.0
        features["audio_diurnal_dynamism"] = 0.0
        features["temp_linear_trend"] = 0.0

    # ---------------------------------------------------------
    # Group 3: Acoustic Dynamics & Harmonic Features (8 features)
    # ---------------------------------------------------------
    band_means = np.array([features[f"{b}_mean"] for b in FREQ_BANDS])
    lin_powers = np.power(10.0, np.clip(band_means / 10.0, -10.0, 5.0))
    sum_lin = np.sum(lin_powers)

    # 1. Spectral Centroid (Hz)
    if sum_lin > 1e-9:
        features["spectral_centroid"] = float(
            np.sum(FREQ_VALUES * lin_powers) / sum_lin
        )
    else:
        features["spectral_centroid"] = float(np.mean(FREQ_VALUES))

    # 2. Piping-to-Fanning Energy Ratio
    piping_p = sum(features[f"{b}_mean"] for b in PIPING_BANDS)
    fanning_p = sum(features[f"{b}_mean"] for b in FANNING_BANDS)
    features["piping_to_fanning_ratio"] = float(piping_p / (abs(fanning_p) + 1e-4))

    # 3. Audio Density Interquartile Range (IQR)
    ad_s = pd.to_numeric(window_df["audio_density"], errors="coerce").dropna()
    if len(ad_s) > 3:
        q75, q25 = np.percentile(ad_s, [75, 25])
        features["audio_density_iqr"] = float(q75 - q25)
    else:
        features["audio_density_iqr"] = 0.0

    # 4. Low Band Power Sum (122 to 244 Hz)
    features["low_band_power_sum"] = float(
        sum(features[f"{b}_mean"] for b in LOW_BANDS)
    )

    # 5. High Band Power Sum (457 to 579 Hz)
    features["high_band_power_sum"] = float(
        sum(features[f"{b}_mean"] for b in HIGH_BANDS)
    )

    # 6. Dominant Frequency Band Index (0 to 15)
    features["dominant_freq_bin"] = float(np.argmax(band_means))

    # 7. Temperature-Audio Correlation over 24h
    if len(temp_s) > 5 and len(ad_s) > 5 and len(temp_s) == len(ad_s):
        corr = np.corrcoef(temp_s.values, ad_s.values)[0, 1]
        features["temp_audio_correlation"] = float(corr) if not np.isnan(corr) else 0.0
    else:
        features["temp_audio_correlation"] = 0.0

    # Sanitize any NaNs or Infs
    for k, v in features.items():
        if np.isnan(v) or np.isinf(v):
            features[k] = 0.0

    assert len(features) == 75, f"Expected exactly 75 features, got {len(features)}"
    return features


ENVIRONMENTAL_FEATURES: List[str] = [
    "temperature_mean",
    "temperature_std",
    "temperature_range",
    "humidity_mean",
    "humidity_std",
    "humidity_range",
    "temp_cv",
    "brood_temp_adherence",
    "temp_diurnal_drop",
    "temp_linear_trend",
]


def compute_24h_environmental_features(
    window_df: pd.DataFrame, time_col: str = "published_at"
) -> Dict[str, float]:
    """
    Computes exactly the 10 microclimate environmental features from a 24-hour sliding window.
    Strictly temperature and humidity based, requiring zero acoustic computation.
    """
    if len(window_df) == 0:
        raise ValueError("Cannot extract features from an empty 24-hour window.")

    features: Dict[str, float] = {}

    for col in ["temperature", "humidity"]:
        if col in window_df.columns:
            s = pd.to_numeric(window_df[col], errors="coerce").dropna()
            if len(s) > 0:
                features[f"{col}_mean"] = float(s.mean())
                features[f"{col}_std"] = float(s.std(ddof=1)) if len(s) > 1 else 0.0
                features[f"{col}_range"] = float(s.max() - s.min())
            else:
                (
                    features[f"{col}_mean"],
                    features[f"{col}_std"],
                    features[f"{col}_range"],
                ) = (0.0, 0.0, 0.0)
        else:
            (
                features[f"{col}_mean"],
                features[f"{col}_std"],
                features[f"{col}_range"],
            ) = (0.0, 0.0, 0.0)

    t_mean = features["temperature_mean"]
    t_std = features["temperature_std"]
    temp_s = (
        pd.to_numeric(window_df["temperature"], errors="coerce").dropna()
        if "temperature" in window_df.columns
        else pd.Series([], dtype=float)
    )

    # 1. Temperature Coefficient of Variation (thermal stability)
    features["temp_cv"] = float(t_std / (abs(t_mean) + 1e-5))

    # 2. Brood Core Adherence (|mean_T - 34.5°C|)
    features["brood_temp_adherence"] = float(abs(t_mean - 34.5))

    # 3. Day vs. Night Temperature Drop
    if time_col in window_df.columns:
        dt = pd.to_datetime(window_df[time_col])
        day_mask = (dt.dt.hour >= 10) & (dt.dt.hour < 24)
        night_mask = ~day_mask

        day_temp = (
            pd.to_numeric(
                window_df.loc[day_mask, "temperature"], errors="coerce"
            ).dropna()
            if "temperature" in window_df.columns
            else []
        )
        night_temp = (
            pd.to_numeric(
                window_df.loc[night_mask, "temperature"], errors="coerce"
            ).dropna()
            if "temperature" in window_df.columns
            else []
        )

        if len(day_temp) > 0 and len(night_temp) > 0:
            features["temp_diurnal_drop"] = float(day_temp.mean() - night_temp.mean())
        else:
            features["temp_diurnal_drop"] = 0.0

        # 4. Temperature Linear Trend (Slope over 24h)
        if len(temp_s) > 3:
            time_sec = (dt.dropna() - dt.dropna().min()).dt.total_seconds().values
            t_hours = time_sec / 3600.0
            if len(t_hours) == len(temp_s) and np.std(t_hours) > 1e-3:
                slope, _ = np.polyfit(t_hours, temp_s.values, deg=1)
                features["temp_linear_trend"] = float(slope)
            else:
                features["temp_linear_trend"] = 0.0
        else:
            features["temp_linear_trend"] = 0.0
    else:
        features["temp_diurnal_drop"] = 0.0
        features["temp_linear_trend"] = 0.0

    for k, v in features.items():
        if np.isnan(v) or np.isinf(v):
            features[k] = 0.0

    assert (
        len(features) == 10
    ), f"Expected exactly 10 environmental features, got {len(features)}"
    return features


def get_feature_names() -> List[str]:
    """Returns the ordered list of 75 feature names."""
    dummy_data = {c: [1.0, 2.0] for c in SENSOR_CHANNELS}
    dummy_data["published_at"] = pd.to_datetime(
        ["2021-06-01 00:00:00", "2021-06-01 12:00:00"]
    )
    dummy_df = pd.DataFrame(dummy_data)
    feats = compute_24h_features(dummy_df)
    return list(feats.keys())


if __name__ == "__main__":
    print("=" * 80)
    print("FEATURE EXTRACTION MODULE DEMO")
    print("=" * 80)
    names = get_feature_names()
    print(f"Total features defined: {len(names)}\n")

    # Generate a realistic synthetic 24-hour window (96 readings, 15 min intervals)
    np.random.seed(42)
    n_steps = 96
    timestamps = pd.date_range(
        "2021-06-16 00:00:00", periods=n_steps, freq="15min", tz="UTC"
    )

    sim_data = {
        "published_at": timestamps,
        "temperature": 34.2 + np.random.normal(0, 0.4, n_steps),
        "humidity": 55.0 + np.random.normal(0, 1.5, n_steps),
        "audio_density": 15.0 + np.random.normal(0, 2.0, n_steps),
        "audio_density_ratio": 0.85 + np.random.normal(0, 0.05, n_steps),
        "density_variation": 4.5 + np.random.normal(0, 0.5, n_steps),
    }
    for b in FREQ_BANDS:
        sim_data[b] = np.random.normal(-5.0, 2.0, n_steps)

    sample_window = pd.DataFrame(sim_data)
    print(
        f"Input: 24-hour window with {len(sample_window)} readings and {len(sim_data)} sensor channels."
    )

    feats = compute_24h_features(sample_window)
    print(f"\nExtracted: Exactly {len(feats)} physical features.")
    print("\nSample Extracted Features:")
    print(f"  temperature_mean:       {feats['temperature_mean']:.2f} °C")
    print(f"  temperature_std:        {feats['temperature_std']:.3f} °C")
    print(f"  temperature_range:      {feats['temperature_range']:.2f} °C")
    print(
        f"  brood_temp_adherence:   {feats['brood_temp_adherence']:.3f} °C (|mean - 34.5°C|)"
    )
    print(f"  temp_cv:                {feats['temp_cv']:.4f}")
    print(f"  spectral_centroid:      {feats['spectral_centroid']:.1f} Hz")
    print(f"  piping_to_fanning_ratio:{feats['piping_to_fanning_ratio']:.3f}")
    print(
        f"  dominant_freq_bin:      {int(feats['dominant_freq_bin'])} ({FREQ_BANDS[int(feats['dominant_freq_bin'])]})"
    )
    print(
        "\n[OK] Module is working properly and ready for training and real-time backend inference."
    )
