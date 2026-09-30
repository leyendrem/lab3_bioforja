from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

# Permite ejecutar `streamlit run app.py` sin instalar el paquete.
# Si se instala con `uv sync` o `pip install -e .`, esto no hace nada.
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import welch

# Se importa como paquete: un `import io` suelto cargaría el módulo `io`
# de la librería estándar, no emg_dashboard/io.py.
from emg_dashboard import data_validation as emg_val
from emg_dashboard import io as emg_io
from emg_dashboard import metrics as emg_metrics
from emg_dashboard import preprocessing_metrics as emg_prep
from emg_dashboard import visualization as emg_vis

# Asignación directa de funciones para conservar el resto de app.py intacto
load_uploaded_file = emg_io.load_uploaded_file
validate_contract = emg_io.validate_contract

activation_segments = emg_metrics.activation_segments
activation_threshold = emg_metrics.activation_threshold
iemg = emg_metrics.iemg
median_frequency = emg_metrics.median_frequency
median_frequency_trend = emg_metrics.median_frequency_trend
moving_rms = emg_metrics.moving_rms
percent_mvc = emg_metrics.percent_mvc
rms = emg_metrics.rms
segment_summary = emg_metrics.segment_summary
waveform_length = emg_metrics.waveform_length
window_sensitivity = emg_metrics.window_sensitivity

preprocess_emg = emg_prep.preprocess_emg
channel_quality = emg_val.channel_quality
sampling_report = emg_val.sampling_report

signal_figure = emg_vis.signal_figure
spectrum_figure = emg_vis.spectrum_figure
trend_figure = emg_vis.trend_figure

st.set_page_config(page_title="Monitoreo EMG · Rehabilitación", layout="wide")
st.title("Dashboard reproducible de EMG · monitoreo de actividad muscular")
st.caption("Herramienta académica para explorar patrones de activación durante ejercicio.")

st.info("""
**Pregunta rectora:** ¿Cómo transformar una señal EMG adquirida durante una actividad física
en un conjunto reducido de métricas y visualizaciones que un médico deportivo pueda interpretar,
sin ocultar las decisiones de procesamiento que condicionan esos resultados?

Las métricas son indicadores de monitoreo y no sustituyen una valoración clínica. Un cambio en RMS,
%MVC o frecuencia mediana no establece por sí solo fatiga, sobreesfuerzo o lesión.
""")

with st.sidebar:
    st.header("1 · Datos")
    uploaded = st.file_uploader("Cargar registro EMG (.tdf)", type=["tdf"])
    if uploaded is None:
        demo = st.checkbox("Usar registro demostrativo", value=True)
    else:
        demo = False
        st.caption(f"Usando el archivo cargado: {uploaded.name}")

    st.header("2 · Procesamiento")
    low_hz = st.number_input("Pasa-altas (Hz)", min_value=1.0, value=20.0, step=1.0)
    high_hz = st.number_input("Pasa-bajas (Hz)", min_value=5.0, value=450.0, step=5.0)
    envelope_hz = st.number_input("Envolvente pasa-bajas (Hz)", min_value=0.5, value=5.0, step=0.5)
    use_notch = st.checkbox("Aplicar notch", value=False)
    notch_hz = st.number_input("Frecuencia notch (Hz)", min_value=40.0, max_value=70.0, value=60.0, step=1.0)

    st.header("3 · Ventanas")
    rms_window_ms = st.slider("Ventana RMS (ms)", 50, 500, 200, 10)
    spectral_window_s = st.slider("Ventana espectral (s)", 0.5, 4.0, 2.0, 0.5)
    activation_threshold_multiplier = st.slider("Umbral activación × MAD", 1.0, 8.0, 3.0, 0.5)
    min_activation_s = st.slider("Duración mínima activación (s)", 0.05, 0.50, 0.10, 0.05)

    spectrum_log = st.checkbox("Espectro en escala logarítmica", value=False)

    st.header("4 · Referencia MVC")
    mvc_reference = st.number_input(
        "RMS MVC de referencia (misma unidad que la señal)",
        min_value=0.0,
        value=0.0,
        step=1e-5,
        format="%.6f",
        help="Deje 0 si no tiene referencia. Los registros TDF suelen estar en voltios (RMS del orden de 1e-4).",
    )
    mvc_reference = None if mvc_reference <= 0 else float(mvc_reference)


@st.cache_data(show_spinner="Generando registro demostrativo...")
def demo_data() -> tuple[pd.DataFrame, dict]:
    fs = 1000.0
    duration = 30.0
    t = np.arange(0, duration, 1 / fs)
    rng = np.random.default_rng(42)
    envelope = np.zeros_like(t)
    for start in np.arange(2, 29, 3):
        envelope += 0.25 * ((t >= start) & (t < start + 1.5))
    carrier = rng.normal(0, 0.035, len(t))
    drift = 0.01 * np.sin(2 * np.pi * 0.4 * t)
    # Demonstration only: no clinical meaning.
    signal = envelope * np.sin(2 * np.pi * 90 * t) + carrier + drift
    df = pd.DataFrame({"time_s": t, "emg": signal, "channel": "Demo_Muscle", "session": "DEMO", "event": ""})
    return df, {"source": "demo", "fs": fs, "format": "DEMO"}


@st.cache_data(show_spinner="Leyendo archivo TDF...")
def cargar_archivo_subido(nombre: str, contenido: bytes) -> tuple[pd.DataFrame, dict]:
    """Carga en caché: solo se relee el TDF si cambia el archivo."""
    archivo = SimpleNamespace(name=nombre, getbuffer=lambda: contenido)
    return load_uploaded_file(archivo)


@st.cache_data(show_spinner="Filtrando canal...")
def procesar_canal(x, fs, low_hz, high_hz, envelope_hz, use_notch, notch_hz):
    """Preprocesamiento en caché: solo se recalcula si cambian canal o filtros."""
    return preprocess_emg(x, fs, low_hz, high_hz, envelope_hz, use_notch, notch_hz)


if not demo and uploaded is None:
    st.info("Carga un archivo .tdf para comenzar.")
    st.stop()

try:
    if demo:
        df, meta = demo_data()
    else:
        df, meta = cargar_archivo_subido(uploaded.name, uploaded.getvalue())
except Exception as exc:
    st.error(f"No fue posible cargar el registro: {exc}")
    st.stop()

if demo:
    st.warning(
        "Estás viendo datos artificiales de demostración, "
        "no un registro real."
    )

errores = validate_contract(df)
if errores:
    for error in errores:
        st.error(error)
    st.stop()

fs = float(meta.get("fs"))
nyquist = fs / 2
if not 0 < low_hz < high_hz < nyquist:
    st.error(
        f"Filtros no válidos para fs = {fs:g} Hz: debe cumplirse "
        f"0 < pasa-altas < pasa-bajas < {nyquist:g} Hz."
    )
    st.stop()
if envelope_hz >= nyquist:
    st.error(f"La envolvente debe estar por debajo de {nyquist:g} Hz.")
    st.stop()
sessions = sorted(df["session"].unique())
selected_session = st.sidebar.selectbox("Sesión", sessions)
sub = df[df["session"] == selected_session]
channels = sorted(sub["channel"].unique())
selected_channel = st.sidebar.selectbox("Músculo / canal", channels)
ch = sub[sub["channel"] == selected_channel].sort_values("time_s").copy()

min_time = float(ch["time_s"].min())
max_time = float(ch["time_s"].max())

if max_time <= min_time:
    st.warning("El canal no tiene un intervalo de tiempo válido.")
    st.stop()

duracion = max_time - min_time

start, end = st.sidebar.slider(
    "Intervalo de inspección (s)",
    min_value=min_time,
    max_value=max_time,
    value=(min_time, min(min_time + 10.0, max_time)),
    step=min(0.01, duracion / 100)
)
segment = ch[(ch["time_s"] >= start) & (ch["time_s"] <= end)].copy()

if len(segment) < max(100, int(fs * 0.25)):
    st.warning("El intervalo seleccionado es demasiado corto para algunas métricas. Amplíelo.")
    st.stop()

time = segment["time_s"].to_numpy(dtype=float)
inicio_fragmento = float(time[0])

# Obtener todas las muestras del canal seleccionado.
x_canal = ch["emg"].to_numpy(dtype=float)

try:
    # Procesar el canal completo.
    etapas_completas = procesar_canal(
        x_canal, fs, low_hz, high_hz,
        envelope_hz, use_notch, notch_hz
    )

    # Identificar las muestras del intervalo seleccionado.
    mascara = (
        (ch["time_s"] >= start) &
        (ch["time_s"] <= end)
    ).to_numpy()

    # Recortar cada etapa usando la misma selección.
    stages = {
        nombre: valores[mascara]
        for nombre, valores in etapas_completas.items()
    }

except Exception as exc:
    st.error(f"No fue posible procesar el canal: {exc}")
    st.stop()

filtered = stages["filtered"]
envelope = stages["envelope"]

try:
    umbral = activation_threshold(
        etapas_completas["envelope"],
        fs,
        baseline_seconds=1.0,
        threshold_multiplier=activation_threshold_multiplier
    )

    activations = activation_segments(
        envelope,
        fs,
        threshold=umbral,
        min_duration_s=min_activation_s,
        t0=inicio_fragmento,
    )

except Exception as exc:
    st.error(f"No se pudieron calcular las activaciones: {exc}")
    st.stop()

# Summary
st.header("Resumen")
try:
    resumen = segment_summary(
        filtered,
        envelope,
        fs,
        mvc_ref=mvc_reference,
        spectral_window_s=spectral_window_s
    )
except ValueError as exc:
    st.error(f"No se pudieron calcular las métricas: {exc}")
    st.stop()

rms_value = resumen["RMS"]
iemg_value = resumen["iEMG"]
mdf_value = resumen["MDF_Hz"]
wl_value = resumen["WL"]

if resumen["MDF_note"]:
    st.info(
        f"Frecuencia mediana no disponible: "
        f"{resumen['MDF_note']}"
    )

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("RMS", f"{rms_value:.4g}" if np.isfinite(rms_value) else "—")
c2.metric("%MVC", f"{percent_mvc(rms_value, mvc_reference):.1f}%" if mvc_reference and np.isfinite(rms_value) else "Sin referencia")
c3.metric("MDF", f"{mdf_value:.1f} Hz" if np.isfinite(mdf_value) else "—")
c4.metric("iEMG", f"{iemg_value:.4g}" if np.isfinite(iemg_value) else "—")
c5.metric("Activaciones", str(len(activations)))

st.caption(
    "WL (longitud de la forma de onda) del intervalo seleccionado: "
    + (f"{wl_value:.4g}" if np.isfinite(wl_value) else "—")
)

st.header("Señal y activación")
st.plotly_chart(
    signal_figure(
        time,
        stages["dc_removed"],
        filtered,
        envelope,
        activation_segments=activations,
        raw_label="Cruda (sin DC)",
    ),
    width="stretch",
)

st.header("Tendencias")
col1, col2 = st.columns(2)

try:
    rms_series = moving_rms(
        filtered, fs, rms_window_ms
    )
    col1.plotly_chart(
        trend_figure(
            time, rms_series, "RMS móvil", "RMS"
        ),
        width="stretch"
    )
except ValueError as exc:
    col1.warning(
        f"No se pudo calcular el RMS móvil: {exc}"
    )

try:
    t_mdf, y_mdf = median_frequency_trend(
        filtered, fs, spectral_window_s, t0=inicio_fragmento
    )
    col2.plotly_chart(
        trend_figure(
            t_mdf,
            y_mdf,
            "Frecuencia mediana",
            "MDF (Hz)"
        ),
        width="stretch"
    )
except Exception as exc:
    col2.warning(f"No se pudo calcular la tendencia espectral: {exc}")

st.header("Espectro exploratorio")
f, pxx = welch(filtered, fs=fs, nperseg=min(len(filtered), max(16, int(spectral_window_s * fs))))
st.plotly_chart(
    spectrum_figure(f, pxx, y_title="PSD (unidad²/Hz)", log_y=spectrum_log),
    width="stretch",
)

st.header("Sensibilidad a ventanas")
rms_sens, mdf_sens = window_sensitivity(filtered, fs)
sens_rms = pd.DataFrame.from_dict(rms_sens, orient="index").reset_index().rename(columns={"index": "ventana_RMS_ms"})
sens_mdf = pd.DataFrame.from_dict(mdf_sens, orient="index").reset_index().rename(columns={"index": "ventana_MDF_s"})
cs1, cs2 = st.columns(2)
cs1.dataframe(sens_rms, width="stretch", hide_index=True)
cs2.dataframe(sens_mdf, width="stretch", hide_index=True)
st.caption("La ventana final debe elegirse por el fenómeno observado, equilibrando resolución temporal, resolución frecuencial y estacionariedad local.")

st.header("Activaciones detectadas")
if activations:
    st.dataframe(pd.DataFrame(activations), width="stretch", hide_index=True)
else:
    st.write("No se detectaron activaciones con el umbral actual.")

with st.expander("Parámetros de análisis y justificación", expanded=False):
    params = pd.DataFrame([
        ["fs", fs, "Frecuencia del bloque EMG; condiciona el rango filtrable y las ventanas."],
        ["Pasa-altas", low_hz, "Reduce componentes lentas y artefactos de movimiento; debe revisarse con el espectro."],
        ["Pasa-bajas", high_hz, "Conserva el contenido mioeléctrico útil dentro del ancho de banda de adquisición."],
        ["Notch", f"{use_notch} · {notch_hz} Hz", "Solo debe activarse si el espectro evidencia interferencia estrecha de red."],
        ["Envolvente", envelope_hz, "Suaviza la señal rectificada para seguir la dinámica de activación."],
        ["Ventana RMS", rms_window_ms, "Compromiso entre seguimiento temporal y estabilidad de la estimación."],
        ["Ventana espectral", spectral_window_s, "Necesita suficientes muestras y una estacionariedad local razonable."],
        ["Umbral activación", f"mediana + {activation_threshold_multiplier}×1.4826×MAD", "Heurística transparente de detección; no es umbral clínico."],
        ["Duración mínima", min_activation_s, "Evita contar fluctuaciones muy breves como activaciones."],
    ], columns=["Parámetro", "Valor", "Razón fisiológica/metodológica"])
    # Columna con números y textos mezclados: se pasa a texto para Arrow.
    params["Valor"] = params["Valor"].astype(str)
    st.dataframe(params, width="stretch", hide_index=True)

with st.expander("Calidad y trazabilidad"):
    st.json(sampling_report(ch, fs))
    st.json(channel_quality(ch["emg"].to_numpy()))
    st.write("**Origen:**", meta.get("source"), "· **Formato:**", meta.get("format"))
    if meta.get("events"):
        st.write("**Eventos presentes en el TDF:**")
        st.json(meta["events"])

st.header("Vector de características para clasificación")
feature_row = {
    "session": selected_session,
    "channel": selected_channel,
    "start_s": start,
    "end_s": end,
    "RMS": rms_value,
    "%MVC": percent_mvc(rms_value, mvc_reference),
    "MDF_Hz": mdf_value,
    "iEMG": iemg_value,
    "WL": wl_value,
    "n_activations": len(activations),
    "mean_activation_duration_s": float(np.mean([a["duration_s"] for a in activations])) if activations else np.nan,
}
st.dataframe(pd.DataFrame([feature_row]), width="stretch", hide_index=True)
st.caption("Estas variables pueden alimentar una clasificación posterior de actividades cuando existan etiquetas de ejercicio y suficientes repeticiones. El dashboard no entrena un clasificador con una sola sesión.")

st.header("Interpretación fisiológica")
st.markdown("""
- **RMS:** resume la magnitud de la actividad eléctrica en una ventana. Un aumento describe mayor amplitud EMG, pero no equivale por sí solo a mayor fuerza ni a lesión.
- **%MVC:** expresa la amplitud respecto a una referencia MVC. Solo se muestra cuando se proporciona una referencia adquirida con el mismo protocolo/unidades.
- **MDF:** resume la distribución espectral. Un desplazamiento hacia frecuencias menores durante una contracción sostenida y controlada puede ser compatible con manifestaciones mioeléctricas de fatiga; en tareas dinámicas debe interpretarse con cautela.
- **Inicio, fin y duración:** describen cuándo el músculo cruza el umbral de activación y cuánto permanece activo. Sirven para explorar coordinación y posibles activaciones prolongadas, pero requieren contexto biomecánico.
- **iEMG:** cuantifica actividad acumulada y depende de la duración del intervalo; por eso no debe compararse entre segmentos de duraciones diferentes sin control.
- **WL:** cuantifica el recorrido absoluto de la señal y puede aumentar con cambios rápidos o ruido; debe interpretarse junto con la calidad de la señal.
""")

st.warning("Este dashboard no diagnostica fatiga, lesión ni riesgo clínico. Su propósito es hacer visibles patrones cuantitativos y las decisiones de procesamiento para revisión por un especialista.")