import numpy as np

from emg_dashboard.visualization import signal_figure, spectrum_figure, trend_figure


def test_signal_figure():
    # Datos pequeños, inventados únicamente para esta prueba.
    tiempo = [10.0, 10.1, 10.2, 10.3]
    cruda = [0.2, -0.4, 0.6, -0.2]
    filtrada = [0.1, -0.3, 0.5, -0.1]
    envolvente = [0.2, 0.3, 0.3, 0.2]
    activaciones = [{"start_s": 10.1, "end_s": 10.2}]

    figura = signal_figure(
        tiempo, cruda, filtrada, envolvente, activation_segments=activaciones
    )

    # Tres señales, en orden y con sus datos.
    assert len(figura.data) == 3
    assert [trazo.name for trazo in figura.data] == ["Cruda", "Filtrada", "Envolvente"]
    for trazo, valores in zip(figura.data, [cruda, filtrada, envolvente]):
        assert list(trazo.x) == tiempo
        assert list(trazo.y) == valores

    # Una franja de activación con sus límites.
    assert len(figura.layout.shapes) == 1
    assert figura.layout.shapes[0].x0 == 10.1
    assert figura.layout.shapes[0].x1 == 10.2

    # Títulos por defecto.
    assert figura.layout.title.text == "EMG: cruda, filtrada y envolvente"
    assert figura.layout.xaxis.title.text == "Tiempo (s)"
    assert figura.layout.yaxis.title.text == "Amplitud EMG"


def test_signal_figure_sin_activaciones():
    tiempo = [0.0, 0.1, 0.2]
    valores = [0.0, 1.0, 0.0]

    sin_argumento = signal_figure(tiempo, valores, valores, valores)
    lista_vacia = signal_figure(
        tiempo, valores, valores, valores, activation_segments=[]
    )

    assert len(sin_argumento.layout.shapes) == 0
    assert len(lista_vacia.layout.shapes) == 0


def test_signal_figure_varias_activaciones():
    tiempo = [0.0, 1.0, 2.0, 3.0, 4.0]
    valores = [0.0] * 5
    activaciones = [
        {"start_s": 0.5, "end_s": 1.5},
        {"start_s": 2.5, "end_s": 3.5},
    ]

    figura = signal_figure(
        tiempo, valores, valores, valores, activation_segments=activaciones
    )

    limites = [(forma.x0, forma.x1) for forma in figura.layout.shapes]
    assert limites == [(0.5, 1.5), (2.5, 3.5)]


def test_signal_figure_con_numpy_y_etiquetas():
    # La app pasa arreglos de NumPy, no listas.
    tiempo = np.linspace(0.0, 1.0, 5)
    senal = np.array([0.1, -0.2, 0.3, -0.1, 0.0])

    figura = signal_figure(
        tiempo,
        senal,
        senal,
        np.abs(senal),
        raw_label="Cruda (sin DC)",
        y_title="Amplitud (V)",
    )

    np.testing.assert_allclose(figura.data[0].x, tiempo)
    np.testing.assert_allclose(figura.data[2].y, np.abs(senal))
    assert figura.data[0].name == "Cruda (sin DC)"
    assert figura.layout.yaxis.title.text == "Amplitud (V)"


def test_trend_figure():
    tiempo = [1.0, 2.0, 3.0]
    valores = [0.1, 0.3, 0.2]

    figura = trend_figure(tiempo, valores, "RMS móvil", "RMS")

    assert len(figura.data) == 1
    assert list(figura.data[0].x) == tiempo
    assert list(figura.data[0].y) == valores
    assert figura.layout.title.text == "RMS móvil"
    assert figura.layout.xaxis.title.text == "Tiempo (s)"
    assert figura.layout.yaxis.title.text == "RMS"


def test_spectrum_figure():
    frecuencias = [20.0, 60.0, 100.0]
    psd = [0.01, 0.08, 0.02]

    figura = spectrum_figure(frecuencias, psd)

    assert len(figura.data) == 1
    assert list(figura.data[0].x) == frecuencias
    assert list(figura.data[0].y) == psd
    assert figura.layout.title.text == "Densidad espectral de potencia"
    assert figura.layout.xaxis.title.text == "Frecuencia (Hz)"
    assert figura.layout.yaxis.title.text == "PSD"
    assert figura.layout.yaxis.type == "linear"


def test_spectrum_figure_escala_log():
    figura = spectrum_figure(
        [20.0, 60.0], [0.01, 0.08], y_title="PSD (V²/Hz)", log_y=True
    )

    assert figura.layout.yaxis.type == "log"
    assert figura.layout.yaxis.title.text == "PSD (V²/Hz)"


def test_visualizations_tolerate_empty_nan_and_unequal_lengths():
    signal = signal_figure([0, 1, 2], [np.nan, 1], [0], [])
    assert len(signal.data[0].x) == len(signal.data[0].y) == 0
    trend = trend_figure([0, 1], [1], "t", "y")
    spectrum = spectrum_figure([], [np.nan])
    assert len(trend.data[0].x) == len(trend.data[0].y) == 1
    assert len(spectrum.data[0].x) == len(spectrum.data[0].y) == 0
