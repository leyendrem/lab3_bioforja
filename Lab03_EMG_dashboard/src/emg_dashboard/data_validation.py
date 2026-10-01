import numpy as np
import pandas as pd
from scipy.signal import welch


def sampling_report(df: pd.DataFrame, fs: float) -> dict:
    if fs <= 0:
        raise ValueError("fs debe ser > 0")
    t_all = np.sort(df["time_s"].dropna().to_numpy())
    t = np.unique(t_all)
    dt = np.diff(t)
    positive = dt[dt > 0]
    n_dup = int(len(t_all) - len(t))  # duplicados = total - únicos
    return {
        "fs_configured_or_estimated_hz": float(fs),
        "n_unique_times": len(t),
        "duplicate_time_points": n_dup,
        "median_dt_s": float(np.median(positive)) if len(positive) else float("nan"),
        "cv_dt": float(np.std(positive) / np.mean(positive)) if len(positive) and np.mean(positive) else float("nan"),
        "duration_s": float(t[-1] - t[0]) if len(t) > 1 else 0.0,
    }

# median_dt_s: mediana del tiempo que pasa entre una muestra y la siguiente.
# cv_dt: Mide la irregularidad oen el tiempo de adquisición (cero o cercano 
# a él, las muestras tienen intervalos regulares y constantes).



def channel_quality(x: np.ndarray) -> dict:
    a = np.asarray(x, dtype=float)
    finite = a[np.isfinite(a)]
    if len(finite) == 0:
        return {
        "valid_fraction": 0.0,
        "nan_fraction": 1.0,
        "flat": True,
        "saturation_fraction": 0.0,
        "min": float("nan"),
        "max": float("nan"),
        "mean": float("nan"),
        "std": float("nan"), }
    # Saturación/recorte: fracción de muestras que alcanzan el máximo absoluto
    # observado. Un valor > ~0.1 % sugiere señal recortada (clipping).
    peak = float(np.max(np.abs(finite)))
    saturation_fraction = float(np.mean(np.isclose(np.abs(finite), peak))) if peak > 0 else 0.0
    return {
        "valid_fraction": float(np.mean(np.isfinite(a))),
        "nan_fraction": float(np.mean(~np.isfinite(a))),
        "flat": bool(np.nanstd(a) == 0),
        "saturation_fraction": saturation_fraction,
        "min": float(np.nanmin(a)),
        "max": float(np.nanmax(a)),
        "mean": float(np.nanmean(a)),
        "std": float(np.nanstd(a)),
    }

# Si flat == 0, entonces es una línea plana y el electrodo está desconectado.
# Valor mínimo y máximo absoluto registrado en ese músculo (rango dinámico de la señal).
# mean --> en una señal de EMG bien centrada, suele estar muy cerca de cero.
# std --> músculos con alta actividad mostrarán una desv. estándar mayor que los músculos en reposo.



def detect_powerline_interference(
    x: np.ndarray,
    fs: float,
    target_hz: float = 60.0,
    tolerance_hz: float = 1.0,
    threshold_ratio: float = 3.5,
) -> dict:
    """Evidencia espectral para decidir si hace falta un notch.

    Compara la PSD (Welch, ventana de 2 s) en ``target_hz`` con la mediana de
    su vecindario (±5 Hz, excluyendo ±``tolerance_hz``). Un cociente >
    ``threshold_ratio`` indica un pico estrecho compatible con interferencia
    de red (60 Hz en Colombia). Cociente ≈ 1 → no hay pico → no usar notch.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 16:
        return {"has_noise": False, "ratio": 0.0}
    freqs, psd = welch(x, fs=fs, nperseg=min(len(x), int(fs * 2)))
    if target_hz >= fs / 2:
        return {"has_noise": False, "ratio": 0.0}
    peak_power = psd[np.argmin(np.abs(freqs - target_hz))]
    mask = (np.abs(freqs - target_hz) <= 5.0) & (np.abs(freqs - target_hz) > tolerance_hz)
    if not np.any(mask):
        return {"has_noise": False, "ratio": 0.0}
    background = float(np.median(psd[mask]))
    ratio = float(peak_power / background) if background > 0 else 0.0
    return {
        "has_noise": bool(ratio > threshold_ratio),
        "peak_hz_power": float(peak_power),
        "background_median": background,
        "ratio": ratio,
    }
