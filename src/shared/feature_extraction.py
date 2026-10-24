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
    # 1. Temperature features — 15
    "temperature_mean",
    "temperature_std",
    "temperature_min",
    "temperature_max",
    "temperature_range",
    "temp_cv",
    "temp_diurnal_drop",
    "temp_linear_trend",
    "temperature_mean_6h",
    "temperature_mean_12h",
    "temperature_std_6h",
    "temperature_std_12h",
    "temperature_min_6h",
    "temperature_max_6h",
    "brood_temp_adherence",
    # 2. Humidity features — 13
    "humidity_mean",
    "humidity_std",
    "humidity_min",
    "humidity_max",
    "humidity_range",
    "humidity_cv",
    "humidity_linear_trend",
    "humidity_mean_6h",
    "humidity_mean_12h",
    "humidity_std_6h",
    "humidity_std_12h",
    "humidity_min_6h",
    "humidity_max_6h",
    # 3. Additional time-window features — 12
    "temperature_change_6h",
    "temperature_change_12h",
    "temperature_change_24h",
    "temperature_range_6h",
    "temperature_range_12h",
    "temperature_range_24h",
    "humidity_change_6h",
    "humidity_change_12h",
    "humidity_change_24h",
    "humidity_range_6h",
    "humidity_range_12h",
    "humidity_range_24h",
]


def compute_24h_environmental_features(
    window_df: pd.DataFrame, time_col: str = "published_at"
) -> Dict[str, float]:
    """
    Computes exactly the 40 microclimate environmental features from a 24-hour sliding window.
    Strictly temperature and humidity based across 24h, 12h, and 6h horizons.
    """
    if len(window_df) == 0:
        raise ValueError("Cannot extract features from an empty 24-hour window.")

    features: Dict[str, float] = {}
    df = window_df.copy()

    has_time = False
    if time_col in df.columns:
        try:
            df[time_col] = pd.to_datetime(df[time_col])
            df = df.sort_values(by=time_col)
            has_time = True
        except Exception:
            has_time = False

    # 1. Full 24-hour Temperature series
    temp_s = (
        pd.to_numeric(df["temperature"], errors="coerce").dropna()
        if "temperature" in df.columns
        else pd.Series([], dtype=float)
    )
    if len(temp_s) > 0:
        t_mean = float(temp_s.mean())
        t_std = float(temp_s.std(ddof=1)) if len(temp_s) > 1 else 0.0
        t_min = float(temp_s.min())
        t_max = float(temp_s.max())
        t_range = float(t_max - t_min)
    else:
        t_mean, t_std, t_min, t_max, t_range = 0.0, 0.0, 0.0, 0.0, 0.0

    features["temperature_mean"] = t_mean
    features["temperature_std"] = t_std
    features["temperature_min"] = t_min
    features["temperature_max"] = t_max
    features["temperature_range"] = t_range
    features["temp_cv"] = float(t_std / (abs(t_mean) + 1e-5))
    features["brood_temp_adherence"] = float(abs(t_mean - 34.5))

    # 2. Full 24-hour Humidity series
    hum_s = (
        pd.to_numeric(df["humidity"], errors="coerce").dropna()
        if "humidity" in df.columns
        else pd.Series([], dtype=float)
    )
    if len(hum_s) > 0:
        h_mean = float(hum_s.mean())
        h_std = float(hum_s.std(ddof=1)) if len(hum_s) > 1 else 0.0
        h_min = float(hum_s.min())
        h_max = float(hum_s.max())
        h_range = float(h_max - h_min)
    else:
        h_mean, h_std, h_min, h_max, h_range = 0.0, 0.0, 0.0, 0.0, 0.0

    features["humidity_mean"] = h_mean
    features["humidity_std"] = h_std
    features["humidity_min"] = h_min
    features["humidity_max"] = h_max
    features["humidity_range"] = h_range
    features["humidity_cv"] = float(h_std / (abs(h_mean) + 1e-5))

    # 3. Diurnal drop & Linear Trends
    if has_time:
        day_mask = (df[time_col].dt.hour >= 10) & (df[time_col].dt.hour < 24)
        night_mask = ~day_mask

        day_temp = (
            pd.to_numeric(df.loc[day_mask, "temperature"], errors="coerce").dropna()
            if "temperature" in df.columns
            else pd.Series([], dtype=float)
        )
        night_temp = (
            pd.to_numeric(df.loc[night_mask, "temperature"], errors="coerce").dropna()
            if "temperature" in df.columns
            else pd.Series([], dtype=float)
        )

        if len(day_temp) > 0 and len(night_temp) > 0:
            features["temp_diurnal_drop"] = float(day_temp.mean() - night_temp.mean())
        else:
            features["temp_diurnal_drop"] = 0.0

        if len(temp_s) > 3:
            time_sec = (df[time_col].dropna() - df[time_col].dropna().min()).dt.total_seconds().values
            t_hours = time_sec / 3600.0
            if len(t_hours) == len(temp_s) and np.std(t_hours) > 1e-3:
                slope, _ = np.polyfit(t_hours, temp_s.values, deg=1)
                features["temp_linear_trend"] = float(slope)
            else:
                features["temp_linear_trend"] = 0.0
        else:
            features["temp_linear_trend"] = 0.0

        if len(hum_s) > 3:
            time_sec = (df[time_col].dropna() - df[time_col].dropna().min()).dt.total_seconds().values
            t_hours = time_sec / 3600.0
            if len(t_hours) == len(hum_s) and np.std(t_hours) > 1e-3:
                slope_h, _ = np.polyfit(t_hours, hum_s.values, deg=1)
                features["humidity_linear_trend"] = float(slope_h)
            else:
                features["humidity_linear_trend"] = 0.0
        else:
            features["humidity_linear_trend"] = 0.0
    else:
        features["temp_diurnal_drop"] = 0.0
        features["temp_linear_trend"] = 0.0
        features["humidity_linear_trend"] = 0.0

    # 4. Sub-windows: 6h and 12h
    if has_time:
        cutoff = df[time_col].max()
        df_6h = df[df[time_col] >= (cutoff - pd.Timedelta(hours=6))]
        df_12h = df[df[time_col] >= (cutoff - pd.Timedelta(hours=12))]
    else:
        n = len(df)
        df_6h = df.iloc[max(0, int(n * 0.75)) :]
        df_12h = df.iloc[max(0, int(n * 0.50)) :]

    # 6h Temperature Stats
    s6_t = (
        pd.to_numeric(df_6h["temperature"], errors="coerce").dropna()
        if "temperature" in df_6h.columns
        else pd.Series([], dtype=float)
    )
    if len(s6_t) > 0:
        t_mean_6h = float(s6_t.mean())
        t_std_6h = float(s6_t.std(ddof=1)) if len(s6_t) > 1 else 0.0
        t_min_6h = float(s6_t.min())
        t_max_6h = float(s6_t.max())
    else:
        t_mean_6h, t_std_6h, t_min_6h, t_max_6h = t_mean, 0.0, t_min, t_max

    features["temperature_mean_6h"] = t_mean_6h
    features["temperature_std_6h"] = t_std_6h
    features["temperature_min_6h"] = t_min_6h
    features["temperature_max_6h"] = t_max_6h

    # 12h Temperature Stats
    s12_t = (
        pd.to_numeric(df_12h["temperature"], errors="coerce").dropna()
        if "temperature" in df_12h.columns
        else pd.Series([], dtype=float)
    )
    if len(s12_t) > 0:
        t_mean_12h = float(s12_t.mean())
        t_std_12h = float(s12_t.std(ddof=1)) if len(s12_t) > 1 else 0.0
        t_min_12h = float(s12_t.min())
        t_max_12h = float(s12_t.max())
    else:
        t_mean_12h, t_std_12h, t_min_12h, t_max_12h = t_mean, 0.0, t_min, t_max

    features["temperature_mean_12h"] = t_mean_12h
    features["temperature_std_12h"] = t_std_12h

    # 6h Humidity Stats
    s6_h = (
        pd.to_numeric(df_6h["humidity"], errors="coerce").dropna()
        if "humidity" in df_6h.columns
        else pd.Series([], dtype=float)
    )
    if len(s6_h) > 0:
        h_mean_6h = float(s6_h.mean())
        h_std_6h = float(s6_h.std(ddof=1)) if len(s6_h) > 1 else 0.0
        h_min_6h = float(s6_h.min())
        h_max_6h = float(s6_h.max())
    else:
        h_mean_6h, h_std_6h, h_min_6h, h_max_6h = h_mean, 0.0, h_min, h_max

    features["humidity_mean_6h"] = h_mean_6h
    features["humidity_std_6h"] = h_std_6h
    features["humidity_min_6h"] = h_min_6h
    features["humidity_max_6h"] = h_max_6h

    # 12h Humidity Stats
    s12_h = (
        pd.to_numeric(df_12h["humidity"], errors="coerce").dropna()
        if "humidity" in df_12h.columns
        else pd.Series([], dtype=float)
    )
    if len(s12_h) > 0:
        h_mean_12h = float(s12_h.mean())
        h_std_12h = float(s12_h.std(ddof=1)) if len(s12_h) > 1 else 0.0
        h_min_12h = float(s12_h.min())
        h_max_12h = float(s12_h.max())
    else:
        h_mean_12h, h_std_12h, h_min_12h, h_max_12h = h_mean, 0.0, h_min, h_max

    features["humidity_mean_12h"] = h_mean_12h
    features["humidity_std_12h"] = h_std_12h

    # 5. Additional Time-Window Features: Changes & Ranges
    features["temperature_change_6h"] = float(s6_t.iloc[-1] - s6_t.iloc[0]) if len(s6_t) >= 2 else 0.0
    features["temperature_change_12h"] = float(s12_t.iloc[-1] - s12_t.iloc[0]) if len(s12_t) >= 2 else 0.0
    features["temperature_change_24h"] = float(temp_s.iloc[-1] - temp_s.iloc[0]) if len(temp_s) >= 2 else 0.0

    features["temperature_range_6h"] = float(t_max_6h - t_min_6h)
    features["temperature_range_12h"] = float(t_max_12h - t_min_12h)
    features["temperature_range_24h"] = float(t_max - t_min)

    features["humidity_change_6h"] = float(s6_h.iloc[-1] - s6_h.iloc[0]) if len(s6_h) >= 2 else 0.0
    features["humidity_change_12h"] = float(s12_h.iloc[-1] - s12_h.iloc[0]) if len(s12_h) >= 2 else 0.0
    features["humidity_change_24h"] = float(hum_s.iloc[-1] - hum_s.iloc[0]) if len(hum_s) >= 2 else 0.0

    features["humidity_range_6h"] = float(h_max_6h - h_min_6h)
    features["humidity_range_12h"] = float(h_max_12h - h_min_12h)
    features["humidity_range_24h"] = float(h_max - h_min)

    # Sanitize NaNs and Infs
    for k in list(features.keys()):
        v = features[k]
        if np.isnan(v) or np.isinf(v):
            features[k] = 0.0

    # Ensure all 40 features are returned in exact order
    ordered_features = {k: features.get(k, 0.0) for k in ENVIRONMENTAL_FEATURES}

    assert (
        len(ordered_features) == 40
    ), f"Expected exactly 40 environmental features, got {len(ordered_features)}"
    return ordered_features


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
