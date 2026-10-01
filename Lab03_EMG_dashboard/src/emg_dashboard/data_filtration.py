from pathlib import Path

import pandas as pd

from emg_dashboard.io import load_file
from emg_dashboard.preprocessing_metrics import preprocess_emg


def process_and_save_signals(
    raw_dir: str | Path = "data/raw",
    processed_dir: str | Path = "data/processed",
) -> None:
    """Recorre los .tdf, aplica preprocesamiento y guarda CSV en processed_dir."""
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)

    tdf_files = sorted(raw_dir.glob("*.tdf"))
    if not tdf_files:
        raise FileNotFoundError(f"No se encontraron archivos .tdf en: {raw_dir}")

    # Solo crear la carpeta de salida si hay archivos que procesar
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("Iniciando preprocesamiento y filtrado inicial de señales...")

    for file_path in tdf_files:
        df, meta = load_file(file_path)
        fs = meta["fs"]

        processed_channels = []
        for canal in df["channel"].unique():
            df_canal = df[df["channel"] == canal].sort_values("time_s")
            x_raw = df_canal["emg"].values
            t = df_canal["time_s"].values

            stages = preprocess_emg(x_raw, fs=fs, use_notch=False)

            df_res = pd.DataFrame({
                "time_s": t,
                "channel": canal,
                "session": (
                    df_canal["session"].iloc[0]
                    if "session" in df_canal.columns
                    else file_path.stem
                ),
                "event": (
                    df_canal["event"].iloc[0]
                    if "event" in df_canal.columns
                    else ""
                ),
                "raw": stages["raw"],
                "dc_removed": stages["dc_removed"],
                "filtered": stages["filtered"],
                "rectified": stages["rectified"],
                "envelope": stages["envelope"],
            })
            processed_channels.append(df_res)

        df_final = pd.concat(processed_channels, ignore_index=True)
        out_path = processed_dir / f"{file_path.stem}_processed.csv"
        df_final.to_csv(out_path, index=False)
        print(f" Guardado archivo procesado: {out_path.name}")

    print("Preprocesamiento de registros finalizado exitosamente...")
