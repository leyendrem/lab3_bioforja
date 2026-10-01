import numpy as np
import pytest

from emg_dashboard.preprocessing_metrics import (
    bandpass_emg,
    notch_filter,
    preprocess_emg,
    rectify,
    remove_dc,
)


def test_remove_dc():
    """Verifica que se elimine correctamente la componente DC centrando la señal en cero."""
    x = np.array([10.0, 12.0, 14.0])  # Media = 12.0
    cleaned = remove_dc(x)
    assert pytest.approx(np.mean(cleaned)) == 0.0
    assert np.allclose(cleaned, [-2.0, 0.0, 2.0])


def test_rectify():
    """Verifica que la rectificación convierta todos los valores en positivos (absolutos)."""
    x = np.array([-1.5, 0.0, 2.5])
    rectified = rectify(x)
    assert np.all(rectified >= 0.0)
    assert np.allclose(rectified, [1.5, 0.0, 2.5])


def test_bandpass_emg_invalid_freqs():
    """Verifica que se lance un ValueError si las frecuencias de corte no cumplen con el teorema de Nyquist."""
    x = np.zeros(100)
    fs = 1000.0
    # Frecuencia alta mayor que Nyquist (fs / 2 = 500)
    with pytest.raises(ValueError):
        bandpass_emg(x, fs=fs, low_hz=10.0, high_hz=600.0)


def test_notch_filter_invalid_band():
    """Verifica que el filtro notch lance un ValueError si la banda sale de los límites permitidos."""
    x = np.zeros(100)
    fs = 1000.0
    with pytest.raises(ValueError):
        notch_filter(x, fs=fs, center_hz=600.0, bandwidth_hz=2.0)


def test_preprocess_emg_pipeline():
    """Verifica que el pipeline completo devuelva todas las etapas esperadas con las dimensiones correctas."""
    fs = 1000.0
    # Creamos una señal sintética de 1 segundo (1000 muestras)
    t = np.linspace(0, 1, int(fs))
    x = np.sin(2 * np.pi * 50 * t) + 5.0  # Senoide a 50 Hz con offset DC de 5.0
    
    stages = preprocess_emg(x, fs=fs, use_notch=True)
    
    # Comprobar que el diccionario contenga todas las etapas clave
    expected_keys = {"raw", "dc_removed", "filtered", "rectified", "envelope"}
    assert expected_keys.issubset(stages.keys())
    
    # Comprobar que mantenga la misma longitud de muestras
    for key in expected_keys:
        assert len(stages[key]) == len(x)