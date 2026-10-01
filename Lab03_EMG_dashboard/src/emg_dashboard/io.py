from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
from basictdf import Tdf

REQUIRED_COLUMNS = {"time_s", "emg", "channel", "session"}


def build_emg_dataframe(tracks, fs: float,start_time: float, session: str) -> pd.DataFrame:
    """Construye un DataFrame ordenado a partir de las pistas EMG de un TDF."""
    rows = []

    for track in tracks:
        x = np.asarray(track.data, dtype=float)
        # Vector de tiempo basado en la frecuencia de muestreo y el inicio
        time = start_time + np.arange(len(x), dtype=float) / fs

        rows.append(
            pd.DataFrame({
                "time_s": time,
                "emg": x,
                "channel": str(track.label),
                "session": session,
                "event": "",
            })
        )

    return pd.concat(rows, ignore_index=True)


def load_bts_tdf(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Read a BTS Bioengineering TDF EMG block using basictdf.

    The BTS TDF EMG block exposes frequency, startTime, nSamples and one or
    more EMG tracks. Temporal events, when present, are retained separately
    in metadata because event semantics depend on the acquisition protocol.
    """

    path = Path(path)
    with Tdf(path) as tdf:
        if not tdf.has_emg:
            raise ValueError("El TDF no contiene un bloque electromiográfico.")
        emg = tdf.emg
        fs = float(emg.frequency)
        start = float(emg.startTime)
        
        df = build_emg_dataframe(
            tracks=emg, 
            fs=fs, 
            start_time=start, 
            session=path.stem
        )

        event_meta = []
        if getattr(tdf, "has_events", False):
            events = tdf.events
            for event in events:
                event_meta.append({
                    "label": str(event.label),
                    "type": str(event.type),
                    "values_s": [float(v) for v in event.values],
                })

    metadata = {
        "source": path.name,
        "fs": fs,
        "format": "BTS TDF",
        "start_time_s": start,
        "n_channels": len(emg),
        "n_samples": int(emg.nSamples),
        "events": event_meta,
    }
    return df, metadata


def load_file(path: str | Path) -> tuple[pd.DataFrame, dict]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".tdf":
        return load_bts_tdf(path)
    raise ValueError(f"Formato no soportado: {suffix}. Use .tdf")


def load_uploaded_file(uploaded_file) -> tuple[pd.DataFrame, dict]:
    suffix = Path(uploaded_file.name).suffix.lower()
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = Path(tmp.name)
    try:
        return load_file(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def validate_contract(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        errors.append(f"Faltan columnas obligatorias: {sorted(missing)}")
    if errors:
        return errors
    if df.empty:
        errors.append("El registro está vacío.")
    if not df["time_s"].notna().all():
        errors.append("Existen tiempos faltantes.")
    if not df["emg"].notna().any():
        errors.append("No existen muestras EMG válidas.")
    if (df["channel"].str.strip() == "").any():
        errors.append("Existe al menos un canal sin nombre.")
    return errors


def load_directory(data_dir: str | Path = "data/raw") -> tuple[pd.DataFrame, dict[str, dict]]:
    """Carga todos los .tdf de una carpeta como sesiones de un mismo DataFrame.

    Devuelve (df, metadata_por_sesion). ``session`` es el nombre del archivo,
    lo que permite seleccionar registro desde el dashboard. Los archivos
    originales no se modifican.
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.tdf"))
    if not files:
        raise FileNotFoundError(f"No se encontraron archivos .tdf en: {data_dir}")
    frames, metas = [], {}
    for f in files:
        df, meta = load_file(f)
        frames.append(df)
        metas[f.stem] = meta
    return pd.concat(frames, ignore_index=True), metas
