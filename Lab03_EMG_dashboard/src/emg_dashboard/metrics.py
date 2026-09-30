"""Métricas de amplitud, espectrales y de activación para sEMG.

Convenciones
------------
* ``x`` es un ``np.ndarray`` 1-D en la MISMA unidad de la adquisición
  (los TDF de BTS entregan voltios; el bloque no declara la unidad, por lo que
  se asume V y debe verificarse con el equipo).
* ``fs`` está en Hz; los tiempos devueltos están en segundos.
* Las funciones NO ocultan valores no finitos: los omiten avisando con
  ``warnings.warn`` (métricas de amplitud) o fallan con un mensaje claro
  (métricas espectrales, que requieren muestras contiguas).
* Todas fallan con ``ValueError`` y un mensaje comprensible cuando la señal es
  demasiado corta, para que el dashboard pueda mostrarlo al usuario.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.signal import welch

MIN_SPECTRAL_SAMPLES = 16


# --------------------------------------------------------------------------
# Utilidades de validación
# --------------------------------------------------------------------------
def _finite(x, min_samples: int = 10, name: str = "señal") -> np.ndarray:
    """Devuelve las muestras finitas de ``x`` avisando si se omitió alguna."""
    a = np.asarray(x, dtype=float).ravel()
    finite = a[np.isfinite(a)]
    n_bad = len(a) - len(finite)
    if n_bad:
        warnings.warn(
            f"{name}: se omitieron {n_bad} muestras no finitas "
            f"({100 * n_bad / len(a):.2f} %).",
            RuntimeWarning,
            stacklevel=3,
        )
    if len(finite) < min_samples:
        raise ValueError(
            f"Se requieren al menos {min_samples} muestras válidas en {name}; "
            f"se recibieron {len(finite)}."
        )
    return finite


def _check_fs(fs: float) -> None:
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs debe ser un número positivo (Hz).")


# --------------------------------------------------------------------------
# Métricas de amplitud
# --------------------------------------------------------------------------
def moving_rms(x: np.ndarray, fs: float, window_ms: float = 200.0) -> np.ndarray:
    """RMS móvil (misma unidad que ``x``) con ventana centrada de ``window_ms``.

    Las muestras no finitas no se sustituyen por cero: se excluyen del
    promedio de cada ventana y, si una ventana no tiene muestras válidas, el
    resultado es ``NaN`` en ese punto. En los bordes el promedio se normaliza
    con las muestras realmente disponibles.
    """
    _check_fs(fs)
    if window_ms <= 0:
        raise ValueError("window_ms debe ser positivo.")
    a = np.asarray(x, dtype=float).ravel()
    n = max(1, int(round(window_ms * fs / 1000.0)))
    if n > len(a):
        raise ValueError(
            f"La ventana RMS ({window_ms:.0f} ms = {n} muestras) es mayor que "
            f"la señal ({len(a)} muestras = {1000 * len(a) / fs:.0f} ms)."
        )
    valid = np.isfinite(a)
    sq = np.where(valid, a, 0.0) ** 2
    kernel = np.ones(n)
    num = np.convolve(sq, kernel, mode="same")
    den = np.convolve(valid.astype(float), kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.sqrt(np.where(den > 0, num / den, np.nan))
    return out


def rms(x: np.ndarray) -> float:
    """RMS del segmento completo (misma unidad que ``x``)."""
    a = _finite(x, 10, "RMS")
    return float(np.sqrt(np.mean(a**2)))


def iemg(x: np.ndarray, fs: float) -> float:
    """iEMG = Σ|x|·Δt  (unidad·s). Depende de la duración del segmento."""
    _check_fs(fs)
    a = _finite(x, 10, "iEMG")
    return float(np.sum(np.abs(a)) / fs)


def waveform_length(x: np.ndarray) -> float:
    """WL = Σ|x[n+1] − x[n]| (misma unidad que ``x``). Sensible al ruido."""
    a = _finite(x, 2, "WL")
    return float(np.sum(np.abs(np.diff(a))))


def percent_mvc(rms_value: float, mvc_reference: float | None) -> float:
    """%MVC = 100·RMS/RMS_MVC. Devuelve NaN si no hay referencia válida."""
    if mvc_reference is None or not np.isfinite(mvc_reference) or mvc_reference <= 0:
        return float("nan")
    return float(100.0 * rms_value / mvc_reference)


# --------------------------------------------------------------------------
# Métrica espectral
# --------------------------------------------------------------------------
def median_frequency(x: np.ndarray, fs: float, nperseg: int = 1024) -> float:
    """Frecuencia mediana (Hz) a partir de la PSD de Welch (solape 50 %).

    Requiere muestras contiguas: si hay valores no finitos o el segmento es
    más corto que ``nperseg`` falla con un mensaje explícito en lugar de
    devolver un número engañoso.
    """
    _check_fs(fs)
    a = np.asarray(x, dtype=float).ravel()
    if not np.all(np.isfinite(a)):
        raise ValueError(
            "El segmento contiene valores no finitos; la frecuencia mediana "
            "requiere muestras contiguas."
        )
    nperseg = int(nperseg)
    if nperseg < MIN_SPECTRAL_SAMPLES:
        raise ValueError(
            f"La ventana espectral es demasiado corta (mínimo {MIN_SPECTRAL_SAMPLES} muestras)."
        )
    if len(a) < nperseg:
        raise ValueError(
            f"El segmento ({len(a) / fs:.2f} s) es más corto que la ventana "
            f"espectral ({nperseg / fs:.2f} s). Amplíe el intervalo o reduzca la ventana."
        )
    freqs, pxx = welch(a, fs=fs, nperseg=nperseg, detrend="constant")
    total = float(np.sum(pxx))
    if not np.isfinite(total) or total <= 0:
        return float("nan")  # señal plana: el espectro no tiene potencia
    idx = int(np.searchsorted(np.cumsum(pxx), total / 2.0))
    return float(freqs[min(idx, len(freqs) - 1)])


def median_frequency_trend(
    x: np.ndarray,
    fs: float,
    window_s: float = 2.0,
    overlap: float = 0.5,
    t0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """MDF por ventanas deslizantes → (tiempos centrales [s], MDF [Hz]).

    ``overlap`` es la fracción de solapamiento entre ventanas consecutivas
    (0.5 = 50 %). ``t0`` desplaza el eje al tiempo absoluto del segmento.
    Las ventanas donde la MDF no puede calcularse quedan como ``NaN``.
    """
    _check_fs(fs)
    if not 0 <= overlap < 1:
        raise ValueError("overlap debe estar en [0, 1).")
    a = np.asarray(x, dtype=float).ravel()
    n = int(round(window_s * fs))
    if n < MIN_SPECTRAL_SAMPLES:
        raise ValueError("La ventana espectral es demasiado corta.")
    if len(a) < n:
        raise ValueError(
            f"El segmento ({len(a) / fs:.2f} s) es más corto que la ventana "
            f"espectral ({window_s:.2f} s): no hay tendencia que calcular."
        )
    hop = max(1, int(round(n * (1 - overlap))))
    times, values = [], []
    for start in range(0, len(a) - n + 1, hop):
        win = a[start : start + n]
        try:
            values.append(median_frequency(win, fs, nperseg=n))
        except ValueError:
            values.append(float("nan"))
        times.append(t0 + (start + n / 2) / fs)
    return np.asarray(times), np.asarray(values)


# --------------------------------------------------------------------------
# Activación
# --------------------------------------------------------------------------
def activation_threshold(
    envelope: np.ndarray,
    fs: float,
    baseline_seconds: float = 1.0,
    threshold_multiplier: float = 3.0,
) -> float:
    """Umbral = mediana + k·1.4826·MAD del primer ``baseline_seconds`` de la envolvente.

    Es una heurística de análisis (no un umbral clínico). Si la línea base
    es degenerada se usa el percentil 75 de la envolvente.
    """
    env = np.asarray(envelope, dtype=float)
    _check_fs(fs)
    if len(env) < 3:
        raise ValueError("Señal insuficiente para detectar activaciones.")
    n0 = min(len(env), max(1, int(round(baseline_seconds * fs))))
    base = env[:n0]
    med = np.nanmedian(base)
    mad = np.nanmedian(np.abs(base - med))
    thr = float(med + threshold_multiplier * 1.4826 * mad)
    if not np.isfinite(thr) or thr <= 0:
        thr = float(np.nanpercentile(env, 75))
    return thr


def activation_segments(
    envelope: np.ndarray,
    fs: float,
    threshold: float | None = None,
    baseline_seconds: float = 1.0,
    threshold_multiplier: float = 3.0,
    min_duration_s: float = 0.10,
    merge_gap_s: float = 0.10,
    t0: float = 0.0,
) -> list[dict]:
    """Detecta activaciones (comienzo, fin, duración) sobre la envolvente.

    ``t0`` es el tiempo absoluto de la primera muestra, de modo que los
    segmentos se puedan dibujar directamente sobre el eje de tiempo del
    registro. Cada segmento incluye el umbral usado para trazabilidad.
    """
    env = np.asarray(envelope, dtype=float)
    _check_fs(fs)
    if len(env) < 3:
        raise ValueError("Señal insuficiente para detectar activaciones.")
    if threshold is None:
        threshold = activation_threshold(env, fs, baseline_seconds, threshold_multiplier)
    active = np.nan_to_num(env, nan=0.0) > threshold  # NaN = "no activo" (explícito)
    starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    ends = np.flatnonzero(active & ~np.r_[active[1:], False])

    raw = []
    for s, e in zip(starts, ends):
        dur = (e - s + 1) / fs
        if dur >= min_duration_s:
            raw.append([t0 + s / fs, t0 + (e + 1) / fs])
    merged: list[list[float]] = []
    for s, e in raw:
        if merged and s - merged[-1][1] <= merge_gap_s:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [
        {"start_s": s, "end_s": e, "duration_s": e - s, "threshold": float(threshold)}
        for s, e in merged
    ]


# --------------------------------------------------------------------------
# Resumen por segmento y análisis de sensibilidad
# --------------------------------------------------------------------------
def segment_summary(
    x_filt: np.ndarray,
    envelope: np.ndarray,
    fs: float,
    mvc_ref: float | None = None,
    spectral_window_s: float = 2.0,
) -> dict:
    """Métricas de un segmento: RMS, %MVC, iEMG, WL, MDF y envolvente media.

    Si la MDF no puede calcularse (segmento corto, NaN) se devuelve ``NaN``
    y el motivo queda en la clave ``"MDF_note"`` en lugar de ocultarse.
    """
    r = rms(x_filt)
    out = {
        "RMS": r,
        "%MVC": percent_mvc(r, mvc_ref),
        "iEMG": iemg(x_filt, fs),
        "WL": waveform_length(x_filt),
        "mean_envelope": float(np.nanmean(envelope)),
        "MDF_Hz": float("nan"),
        "MDF_note": "",
    }
    try:
        out["MDF_Hz"] = median_frequency(x_filt, fs, nperseg=int(round(spectral_window_s * fs)))
    except ValueError as exc:
        out["MDF_note"] = str(exc)
    return out


def window_sensitivity(
    x: np.ndarray,
    fs: float,
    rms_windows_ms: tuple[float, ...] = (100, 200, 400),
    mdf_windows_s: tuple[float, ...] = (1.0, 2.0, 4.0),
    overlap: float = 0.5,
) -> tuple[dict, dict]:
    """Compara ventanas menor / elegida / mayor para RMS y MDF.

    Devuelve dos diccionarios ``{ventana: {mean, std, cv, n}}``. La desviación
    estándar mide la variabilidad de la estimación; el número de puntos
    (``n``) resume la resolución temporal disponible.
    """
    rms_res: dict[float, dict] = {}
    for w in rms_windows_ms:
        try:
            y = moving_rms(x, fs, float(w))
            m, s = float(np.nanmean(y)), float(np.nanstd(y))
            rms_res[float(w)] = {"mean": m, "std": s, "cv": s / m if m else float("nan"), "n": int(np.isfinite(y).sum())}
        except ValueError:
            rms_res[float(w)] = {"mean": float("nan"), "std": float("nan"), "cv": float("nan"), "n": 0}

    mdf_res: dict[float, dict] = {}
    for w in mdf_windows_s:
        try:
            _, y = median_frequency_trend(x, fs, float(w), overlap)
            m, s = float(np.nanmean(y)), float(np.nanstd(y))
            mdf_res[float(w)] = {"mean": m, "std": s, "cv": s / m if m else float("nan"), "n": int(np.isfinite(y).sum())}
        except ValueError:
            mdf_res[float(w)] = {"mean": float("nan"), "std": float("nan"), "cv": float("nan"), "n": 0}
    return rms_res, mdf_res
