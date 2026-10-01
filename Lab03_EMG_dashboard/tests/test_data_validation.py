import numpy as np
import pandas as pd
import pytest

from emg_dashboard.data_validation import channel_quality, sampling_report


def test_sampling_report():
    """Valida que el reporte de muestreo calcule correctamente los intervalos y la duración."""
    # Creamos un DataFrame con un muestreo regular a 1000 Hz (dt = 0.001s)
    time_s = np.array([0.0, 0.001, 0.002, 0.003, 0.004])
    df = pd.DataFrame({"time_s": time_s, "emg": [1, 2, 3, 4, 5]})
    
    report = sampling_report(df, fs=1000.0)
    
    assert report["fs_configured_or_estimated_hz"] == 1000.0
    assert report["n_unique_times"] == 5
    assert report["duplicate_time_points"] == 0
    assert pytest.approx(report["median_dt_s"]) == 0.001
    assert report["cv_dt"] == 0.0  # Al ser perfectamente constante, el coeficiente de variación es 0
    assert pytest.approx(report["duration_s"]) == 0.004


def test_channel_quality_normal():
    """Valida el análisis de calidad para una señal EMG normal y limpia."""
    x = np.array([0.1, -0.2, 0.0, 0.3, -0.1])
    quality = channel_quality(x)
    
    assert quality["valid_fraction"] == 1.0
    assert quality["nan_fraction"] == 0.0
    assert quality["flat"] is False
    assert "mean" in quality
    assert "std" in quality


def test_channel_quality_flat_and_nans():
    """Valida que detecte canales planos o con valores nulos (NaN)."""
    # Señal plana (desviación estándar cero -> electrodo desconectado)
    x_flat = np.array([5.0, 5.0, 5.0, 5.0])
    q_flat = channel_quality(x_flat)
    assert q_flat["flat"] is True

    # Señal con NaNs
    x_nan = np.array([1.0, np.nan, 3.0, np.nan])
    q_nan = channel_quality(x_nan)
    assert q_nan["valid_fraction"] == 0.5
    assert q_nan["nan_fraction"] == 0.5