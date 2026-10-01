import json
from pathlib import Path

from emg_dashboard.data_validation import channel_quality, sampling_report
from emg_dashboard.io import load_file, validate_contract


def export_metadata_to_json(
    metadata: dict,
    indent: int = 2,
    output_dir: str | Path | None = None,
) -> str:
    """Exporta metadatos a JSON. Si output_dir es None, usa results/ en la raíz del proyecto."""
    if not isinstance(metadata, dict):
        raise TypeError("metadata debe ser un diccionario")
    if indent < 0:
        raise ValueError("indent debe ser >= 0")

    if output_dir is None:
        results_dir = Path(__file__).resolve().parents[2] / "results"
    else:
        results_dir = Path(output_dir)

    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "emg_metadata_summary.json"

    json_str = json.dumps(metadata, indent=indent, ensure_ascii=False)
    output_path.write_text(json_str, encoding="utf-8")
    return json_str


def process_and_export_summary(
    data_dir: str | Path = "data/raw",
    output_dir: str | Path | None = None,
) -> str:
    """Procesa todos los .tdf de una carpeta y exporta un JSON consolidado de metadatos."""
    data_dir = Path(data_dir)
    tdf_files = sorted(data_dir.glob("*.tdf"))

    if not tdf_files:
        raise FileNotFoundError(f"No se encontraron archivos .tdf en: {data_dir}")

    summary_list = []
    for file_path in tdf_files:
        df, meta = load_file(file_path)

        errors = validate_contract(df)
        if errors:
            raise ValueError(
                f"El archivo {file_path.name} no pasó la validación del contrato: {errors}"
            )

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

    return export_metadata_to_json(consolidated_metadata, output_dir=output_dir)
