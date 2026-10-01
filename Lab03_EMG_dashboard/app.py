from pathlib import Path
import sys
from types import SimpleNamespace

# Permite ejecutar `streamlit run app.py` sin instalar el paquete.
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import welch

# Importaciones del paquete emg_dashboard
from emg_dashboard import data_compilation as emg_comp
from emg_dashboard import data_filtration as emg_filt
from emg_dashboard import data_validation as emg_val
from emg_dashboard import io as emg_io
from emg_dashboard import metrics as emg_metrics
from emg_dashboard import preprocessing_metrics as emg_prep
from emg_dashboard import visualization as emg_vis

# Asignación directa de funciones
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

process_and_save_signals = emg_filt.process_and_save_signals
process_and_export_summary = emg_comp.process_and_export_summary

# Directorio de datos crudos
RAW_DATA_DIR = Path("data/raw")

# Parámetros de preprocesamiento y análisis fijos
FIXED_LOW_HZ = 20.0
FIXED_HIGH_HZ = 450.0
FIXED_ENVELOPE_HZ = 5.0
FIXED_USE_NOTCH = False
FIXED_NOTCH_HZ = 60.0
FIXED_ACTIVATION_MULTIPLIER = 2.5  # Subido de 3.0 a 4.0 para ser más estricto con el ruido

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
    st.header("1 · Muestras TDF (data/raw)")
    raw_files = sorted(list(RAW_DATA_DIR.glob("*.tdf")))
    
    demo = st.checkbox("Usar registro demostrativo", value=not raw_files)
    selected_raw_path = None

    if not demo:
        if not raw_files:
            st.warning(f"No se encontraron archivos .tdf en `{RAW_DATA_DIR}`.")
        else:
            selected_raw_path = st.selectbox(
                "Selecciona una muestra TDF",
                raw_files,
                format_func=lambda x: x.name,
            )

    st.header("2 · Ventanas y Análisis")
    rms_window_ms = st.slider("Ventana RMS (ms)", 50, 500, 200, 10)
    spectral_window_s = st.slider("Ventana espectral (s)", 0.5, 4.0, 2.0, 0.5)
    min_activation_s = st.slider("Duración mínima activación (s)", 0.05, 0.50, 0.10, 0.05)
    spectrum_log = st.checkbox("Espectro en escala logarítmica", value=False)


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
    signal = envelope * np.sin(2 * np.pi * 90 * t) + carrier + drift
    df = pd.DataFrame({"time_s": t, "emg": signal, "channel": "Demo_Muscle", "session": "DEMO", "event": ""})
    return df, {"source": "demo", "fs": fs, "format": "DEMO"}


@st.cache_data(show_spinner="Leyendo archivo TDF local...")
def cargar_archivo_tdf_local(file_path: Path) -> tuple[pd.DataFrame, dict]:
    archivo = SimpleNamespace(name=file_path.name, getbuffer=file_path.read_bytes)
    return load_uploaded_file(archivo)


@st.cache_data(show_spinner="Filtrando canal...")
def procesar_canal(x, fs, low_hz, high_hz, envelope_hz, use_notch, notch_hz):
    return preprocess_emg(x, fs, low_hz, high_hz, envelope_hz, use_notch, notch_hz)


try:
    if demo:
        df, meta = demo_data()
    elif selected_raw_path is not None:
        df, meta = cargar_archivo_tdf_local(selected_raw_path)
    else:
        st.info("Por favor, selecciona una muestra TDF o activa el modo demostrativo.")
        st.stop()
except Exception as exc:
    st.error(f"No fue posible cargar el registro: {exc}")
    st.stop()

if demo:
    st.warning("Estás viendo datos artificiales de demostración, no un registro real.")

errores = validate_contract(df)
if errores:
    for error in errores:
        st.error(error)
    st.stop()

fs = float(meta.get("fs"))
nyquist = fs / 2

sessions = sorted(df["session"].unique())
selected_session = st.sidebar.selectbox("Sesión", sessions)
sub = df[df["session"] == selected_session]
channels = sorted(sub["channel"].unique())
selected_channel = st.sidebar.selectbox("Músculo / canal", channels)
ch = sub[sub["channel"] == selected_channel].sort_values("time_s").copy()

# --- Referencia MVC dinámica ---
max_signal_amp = float(ch["emg"].abs().max()) if not ch.empty else 1.0

with st.sidebar:
    st.header("3 · Referencia MVC")
    usar_mvc_auto = st.checkbox("Estimar MVC con el pico de la señal", value=False)
    
    if usar_mvc_auto:
        mvc_reference = max_signal_amp
        st.info(f"Usando MVC estimada: {mvc_reference:.5f}")
    else:
        mvc_reference = st.number_input(
            "RMS MVC de referencia",
            min_value=0.0,
            max_value=max_signal_amp * 2.0 if max_signal_amp > 0 else 1.0,
            value=0.0,
            step=max_signal_amp / 100.0 if max_signal_amp > 0 else 1e-4,
            format="%.5f",
            help="Ingresa el valor real de tu prueba o déjalo en 0 para omitir.",
        )
        mvc_reference = None if mvc_reference <= 0 else float(mvc_reference)

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
    step=min(0.01, duracion / 100),
)
segment = ch[(ch["time_s"] >= start) & (ch["time_s"] <= end)].copy()

if len(segment) < max(100, int(fs * 0.25)):
    st.warning("El intervalo seleccionado es demasiado corto.")
    st.stop()

time = segment["time_s"].to_numpy(dtype=float)
inicio_fragmento = float(time[0])
x_canal = ch["emg"].to_numpy(dtype=float)

try:
    etapas_completas = procesar_canal(
        x_canal, fs, FIXED_LOW_HZ, FIXED_HIGH_HZ, FIXED_ENVELOPE_HZ, FIXED_USE_NOTCH, FIXED_NOTCH_HZ
    )
    mascara = ((ch["time_s"] >= start) & (ch["time_s"] <= end)).to_numpy()
    stages = {nombre: valores[mascara] for nombre, valores in etapas_completas.items()}
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
        threshold_multiplier=FIXED_ACTIVATION_MULTIPLIER,
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

st.header("Resumen")
try:
    resumen = segment_summary(
        filtered,
        envelope,
        fs,
        mvc_ref=mvc_reference,
        spectral_window_s=spectral_window_s,
    )
except ValueError as exc:
    st.error(f"No se pudieron calcular las métricas: {exc}")
    st.stop()

rms_value = resumen["RMS"]
iemg_value = resumen["iEMG"]
mdf_value = resumen["MDF_Hz"]
wl_value = resumen["WL"]

if resumen["MDF_note"]:
    st.info(f"Frecuencia mediana no disponible: {resumen['MDF_note']}")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("RMS", f"{rms_value:.4g}" if np.isfinite(rms_value) else "—")
c2.metric("%MVC", f"{percent_mvc(rms_value, mvc_reference):.1f}%" if mvc_reference and np.isfinite(rms_value) else "Sin referencia")
c3.metric("MDF", f"{mdf_value:.1f} Hz" if np.isfinite(mdf_value) else "—")
c4.metric("iEMG", f"{iemg_value:.4g}" if np.isfinite(iemg_value) else "—")
c5.metric("Activaciones", str(len(activations)))

st.caption("WL (longitud de la forma de onda): " + (f"{wl_value:.4g}" if np.isfinite(wl_value) else "—"))

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
    use_container_width=True,
)

st.header("Tendencias")
col1, col2 = st.columns(2)

try:
    rms_series = moving_rms(filtered, fs, rms_window_ms)
    col1.plotly_chart(trend_figure(time, rms_series, "RMS móvil", "RMS"), use_container_width=True)
except ValueError as exc:
    col1.warning(f"No se pudo calcular el RMS móvil: {exc}")

try:
    t_mdf, y_mdf = median_frequency_trend(filtered, fs, spectral_window_s, t0=inicio_fragmento)
    col2.plotly_chart(trend_figure(t_mdf, y_mdf, "Frecuencia mediana", "MDF (Hz)"), use_container_width=True)
except Exception as exc:
    col2.warning(f"No se pudo calcular la tendencia espectral: {exc}")

st.header("Espectro exploratorio")
f, pxx = welch(filtered, fs=fs, nperseg=min(len(filtered), max(16, round(spectral_window_s * fs))))
st.plotly_chart(spectrum_figure(f, pxx, y_title="PSD (unidad²/Hz)", log_y=spectrum_log), use_container_width=True)

st.header("Sensibilidad a ventanas")
rms_sens, mdf_sens = window_sensitivity(filtered, fs)
sens_rms = pd.DataFrame.from_dict(rms_sens, orient="index").reset_index().rename(columns={"index": "ventana_RMS_ms"})
sens_mdf = pd.DataFrame.from_dict(mdf_sens, orient="index").reset_index().rename(columns={"index": "ventana_MDF_s"})
cs1, cs2 = st.columns(2)
cs1.dataframe(sens_rms, use_container_width=True, hide_index=True)
cs2.dataframe(sens_mdf, use_container_width=True, hide_index=True)

st.header("Activaciones detectadas")
if activations:
    st.dataframe(pd.DataFrame(activations), use_container_width=True, hide_index=True)
else:
    st.write("No se detectaron activaciones con el umbral actual.")

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
st.dataframe(pd.DataFrame([feature_row]), use_container_width=True, hide_index=True)

st.header("Interpretación fisiológica")
st.markdown("""
- **RMS:** Magnitud de la actividad eléctrica.
- **%MVC:** Amplitud respecto a referencia.
- **MDF:** Distribución espectral (posible fatiga).
- **Inicio/Fin/Duración:** Dinámica temporal de activación.
- **iEMG:** Actividad acumulada.
- **WL:** Recorrido absoluto de la forma de onda.
""")

st.warning("Este dashboard no diagnostica fatiga, lesión ni riesgo clínico.")

if __name__ == "__main__":
    print("Ejecutando filtrado y guardado de señales...")
    process_and_save_signals()
    
    print("\n--------------------------------------------------\n")
    
    print("Validando contratos y generando resumen consolidado...")
    process_and_export_summary()
    
    print("¡Pipeline ejecutado y exportado con éxito!")
