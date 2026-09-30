from pathlib import Path
import pandas as pd
from emg_dashboard.io import load_file
from emg_dashboard.preprocessing_metrics import preprocess_emg


def process_and_save_signals() -> None:
    """Recorre todos los archivos crudos .tdf, aplica el pipeline de preprocesamiento EMG (filtros, rectificación 
    y envolvente) y guarda los resultados limpios en formato CSV dentro de data/processed/."""
    
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    tdf_files = sorted(list(raw_dir.glob("*.tdf")))
    if not tdf_files:
        raise FileNotFoundError(f"No se encontraron archivos .tdf en: {raw_dir}")

    print("Iniciando preprocesamiento y filtrado inicial de señales...")

    for file_path in tdf_files:
        df, meta = load_file(file_path)
        fs = meta["fs"]
        
        processed_channels = []
        for canal in df["channel"].unique():
            df_canal = df[df["channel"] == canal].sort_values("time_s")
            x_raw = df_canal["emg"].values
            t = df_canal["time_s"].values
            
            # Aplicar pipeline de preprocesamiento (use_notch=False según auditoría previa)
            stages = preprocess_emg(x_raw, fs=fs, use_notch=False)
            
            df_res = pd.DataFrame({
                "time_s": t,
                "channel": canal,
                "raw": stages["raw"],
                "dc_removed": stages["dc_removed"],
                "filtered": stages["filtered"],
                "rectified": stages["rectified"],
                "envelope": stages["envelope"]
            })
            processed_channels.append(df_res)
            
        df_final = pd.concat(processed_channels, ignore_index=True)
        out_path = processed_dir / f"{file_path.stem}_processed.csv"
        df_final.to_csv(out_path, index=False)
        print(f" Guardado archivo procesado: {out_path.name}")
        
    print("Preprocesamiento de registros finalizado exitósamente...")