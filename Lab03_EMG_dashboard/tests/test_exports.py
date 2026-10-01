import json

import numpy as np
import pandas as pd
import pytest

from emg_dashboard import data_compilation, data_filtration


def _contract_df(n=1000):
    t = np.arange(n, dtype=float) / 1000.0
    return pd.DataFrame(
        {
            "time_s": t,
            "emg": np.sin(2 * np.pi * 50 * t),
            "channel": "Biceps",
            "session": "synthetic-session",
            "event": "rep-1",
        }
    )


def test_process_and_save_signals_preserves_traceability(monkeypatch, tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    source = raw_dir / "synthetic.tdf"
    source.write_bytes(b"placeholder")

    monkeypatch.setattr(
        data_filtration,
        "load_file",
        lambda path: (_contract_df(), {"fs": 1000.0}),
    )

    data_filtration.process_and_save_signals(raw_dir, processed_dir)
    result = pd.read_csv(processed_dir / "synthetic_processed.csv")

    assert {"session", "event", "raw", "filtered", "envelope"} <= set(result.columns)
    assert result["session"].eq("synthetic-session").all()
    assert result["event"].eq("rep-1").all()
    assert len(result) == len(_contract_df())


def test_process_and_save_signals_empty_input_does_not_create_output(monkeypatch, tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="No se encontraron"):
        data_filtration.process_and_save_signals(raw_dir, processed_dir)
    assert not processed_dir.exists()


def test_process_and_export_summary_writes_json_to_requested_directory(monkeypatch, tmp_path):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "results"
    raw_dir.mkdir()
    source = raw_dir / "synthetic.tdf"
    source.write_bytes(b"placeholder")

    monkeypatch.setattr(
        data_compilation,
        "load_file",
        lambda path: (_contract_df(20), {"source": "synthetic.tdf", "fs": 1000.0}),
    )

    text = data_compilation.process_and_export_summary(raw_dir, output_dir)
    saved = json.loads(text)
    output_file = output_dir / "emg_metadata_summary.json"

    assert output_file.exists()
    assert saved["total_files"] == 1
    assert saved["files"][0]["sampling_report"]["n_unique_times"] == 20
    assert "channels_quality" in saved["files"][0]


def test_export_metadata_validates_arguments(tmp_path):
    with pytest.raises(TypeError, match="diccionario"):
        data_compilation.export_metadata_to_json([], output_dir=tmp_path)
    with pytest.raises(ValueError, match="indent"):
        data_compilation.export_metadata_to_json({}, indent=-1, output_dir=tmp_path)
