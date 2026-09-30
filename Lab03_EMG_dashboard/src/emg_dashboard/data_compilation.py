from pathlib import Path
import json
from emg_dashboard.io import load_file, validate_contract
from emg_dashboard.data_validation import sampling_report, channel_quality


def export_metadata_to_json(metadata: dict, indent: int = 2) -> str:
    """Exporta el DataFrame de señales EMG y sus metadatos a formato JSON.
    Guarda automáticamente el archivo .json en la carpeta 'results/'"""
    
    # .parents[0] -> src/emg_dashboard
    # .parents[1] -> src
    # .parents[2] -> Raíz del proyecto
    results_dir = Path(__file__).resolve().parents[2] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / "emg_metadata_summary.json"

    json_str = json.dumps(metadata, indent=indent, ensure_ascii=False)
    output_path.write_text(json_str, encoding="utf-8")

    return json_str


def process_and_export_summary(data_dir: str | Path = "data/raw") -> str:
    """Procesa todos los archivos .tdf de una carpeta y exporta un único
    JSON consolidado con la metadata de todos los archivos leídos en 'results/'."""
    
    data_dir = Path(data_dir)
    tdf_files = sorted(list(data_dir.glob("*.tdf")))

    if not tdf_files:
        raise FileNotFoundError(f"No se encontraron archivos .tdf en: {data_dir}")

    summary_list = []
    for file_path in tdf_files:
        df, meta = load_file(file_path)

        errors = validate_contract(df)
        if errors:
            raise ValueError(f"El archivo {file_path.name} no pasó la validación del contrato: {errors}")

        meta["sampling_report"] = sampling_report(df, fs=meta["fs"])

        channels_quality_summary = {}
        for canal in df["channel"].unique():
            x_musculo = df[df["channel"] == canal]["emg"].values
            channels_quality_summary[canal] = channel_quality(x_musculo)
        
        meta["channels_quality"] = channels_quality_summary

        summary_list.append(meta)

    consolidated_metadata = {
        "total_files": len(summary_list),
        "files": summary_list,
    }

    return export_metadata_to_json(consolidated_metadata)