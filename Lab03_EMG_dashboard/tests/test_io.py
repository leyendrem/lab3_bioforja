import pandas as pd
import pytest
from emg_dashboard.io import validate_contract, build_emg_dataframe, load_file


def test_validate_contract_valid():
    """Valida que un DataFrame correcto pase sin errores."""
    df = pd.DataFrame({
        "time_s": [0.0, 0.1, 0.2],
        "emg": [0.5, -0.2, 0.1],
        "channel": ["Biceps", "Biceps", "Biceps"],
        "session": ["s1", "s1", "s1"],
        "event": ["", "", ""]
    })
    errors = validate_contract(df)
    assert errors == []


def test_validate_contract_missing_columns():
    """Valida que detecte columnas obligatorias faltantes."""
    df = pd.DataFrame({"time_s": [0.0], "emg": [0.1]})
    errors = validate_contract(df)
    assert len(errors) > 0


def test_load_file_error(tmp_path):
    """Verifica que falle al intentar leer extensiones no soportadas."""
    bad_file = tmp_path / "datos.txt"
    bad_file.write_text("prueba")
    
    with pytest.raises(ValueError):
        load_file(bad_file)