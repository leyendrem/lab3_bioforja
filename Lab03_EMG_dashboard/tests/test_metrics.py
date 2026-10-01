import numpy as np
import pytest

from emg_dashboard.metrics import (
    activation_segments,
    iemg,
    median_frequency,
    median_frequency_trend,
    moving_rms,
    percent_mvc,
    rms,
    segment_summary,
    waveform_length,
    window_sensitivity,
)

FS = 1000.0


def _sine(freq=50.0, dur=4.0, amp=1.0):
    t = np.arange(0, dur, 1 / FS)
    return amp * np.sin(2 * np.pi * freq * t)


def test_rms_of_sine():
    assert rms(_sine(amp=2.0)) == pytest.approx(2.0 / np.sqrt(2), rel=1e-3)


def test_iemg_depends_on_duration():
    assert iemg(np.ones(2000), FS) == pytest.approx(2 * iemg(np.ones(1000), FS))


def test_waveform_length_constant_is_zero():
    assert waveform_length(np.full(100, 3.0)) == 0.0


def test_percent_mvc_without_reference_is_nan():
    assert np.isnan(percent_mvc(1.0, None))
    assert np.isnan(percent_mvc(1.0, 0.0))
    assert percent_mvc(1.0, 2.0) == 50.0


def test_moving_rms_window_larger_than_signal_raises():
    with pytest.raises(ValueError, match="mayor que la señal"):
        moving_rms(np.ones(50), FS, window_ms=200)


def test_moving_rms_does_not_hide_nan():
    x = np.ones(1000)
    x[400:700] = np.nan
    y = moving_rms(x, FS, 100)
    assert np.isnan(y[550])  # sin muestras válidas → NaN, no 0
    assert y[100] == pytest.approx(1.0)


def test_short_signal_raises_clear_message():
    with pytest.raises(ValueError, match="al menos"):
        rms(np.ones(3))


def test_nan_is_reported_not_hidden():
    x = np.ones(100)
    x[:5] = np.nan
    with pytest.warns(RuntimeWarning, match="omitieron 5"):
        rms(x)


def test_median_frequency_of_sine():
    assert median_frequency(_sine(80.0), FS, nperseg=1000) == pytest.approx(
        80.0, abs=2.0
    )


def test_median_frequency_segment_shorter_than_window():
    with pytest.raises(ValueError, match="más corto que la ventana"):
        median_frequency(np.random.randn(500), FS, nperseg=2000)


def test_median_frequency_rejects_nan():
    x = np.random.randn(3000)
    x[10] = np.nan
    with pytest.raises(ValueError, match="no finitos"):
        median_frequency(x, FS, nperseg=1000)


def test_median_frequency_flat_signal_is_nan():
    assert np.isnan(median_frequency(np.zeros(3000), FS, nperseg=1000))


def test_mdf_trend_shape_and_time_axis():
    t, y = median_frequency_trend(
        _sine(dur=10.0), FS, window_s=2.0, overlap=0.5, t0=5.0
    )
    assert len(t) == len(y) == 9  # (10-2)/1 + 1
    assert t[0] == pytest.approx(6.0)  # 5 + 2/2
    assert np.allclose(y, 50.0, atol=2.0)


def test_activation_segments_time_offset_and_threshold():
    env = np.zeros(5000)
    env[2000:3000] = 1.0
    segs = activation_segments(env, FS, threshold=0.5, t0=10.0)
    assert len(segs) == 1
    assert segs[0]["start_s"] == pytest.approx(12.0)
    assert segs[0]["duration_s"] == pytest.approx(1.0)
    assert segs[0]["threshold"] == 0.5


def test_activation_min_duration_filters_blips():
    env = np.zeros(2000)
    env[500:510] = 1.0  # 10 ms
    assert activation_segments(env, FS, threshold=0.5, min_duration_s=0.1) == []


def test_segment_summary_reports_mdf_failure_instead_of_hiding():
    x = np.random.randn(500)
    out = segment_summary(x, np.abs(x), FS, spectral_window_s=2.0)
    assert np.isnan(out["MDF_Hz"]) and "más corto" in out["MDF_note"]
    assert np.isfinite(out["RMS"])


def test_window_sensitivity_larger_window_is_more_stable():
    rng = np.random.default_rng(0)
    x = rng.normal(size=20000)
    rms_res, mdf_res = window_sensitivity(x, FS, (50, 200, 800), (1.0, 2.0, 4.0))
    assert rms_res[50.0]["cv"] > rms_res[800.0]["cv"]
    assert (
        mdf_res[1.0]["n"] > mdf_res[4.0]["n"]
    )  # más resolución temporal con ventana corta


def test_moving_rms_rejects_empty_or_nonfinite_window():
    with pytest.raises(ValueError, match="vacía"):
        moving_rms([], FS, 100)
    with pytest.raises(ValueError, match="positivo"):
        moving_rms(np.ones(100), FS, np.nan)


def test_activation_parameters_and_summary_lengths_are_validated():
    with pytest.raises(ValueError, match="baseline"):
        from emg_dashboard.metrics import activation_threshold

        activation_threshold(np.ones(100), FS, baseline_seconds=0)
    with pytest.raises(ValueError, match="misma longitud"):
        segment_summary(np.ones(100), np.ones(99), FS)


def test_window_sensitivity_longer_than_signal_returns_coherent_empty_results():
    rms_res, mdf_res = window_sensitivity(np.ones(100), FS, (200,), (2.0,))
    assert rms_res[200.0]["n"] == 0
    assert mdf_res[2.0]["n"] == 0


def test_spectral_and_activation_parameters_have_clear_guards():
    with pytest.raises(ValueError, match="nperseg"):
        median_frequency(np.ones(100), FS, nperseg=np.nan)
    with pytest.raises(ValueError, match="window_s"):
        median_frequency_trend(np.ones(1000), FS, window_s=0)
    with pytest.raises(ValueError, match="finito"):
        activation_segments(np.zeros(100), FS, threshold=0.5, t0=np.nan)
    with pytest.raises(ValueError, match="overlap"):
        window_sensitivity(np.ones(2000), FS, overlap=1.0)
