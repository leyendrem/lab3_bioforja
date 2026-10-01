from pathlib import Path

from emg_dashboard.data_validation import detect_powerline_interference
from emg_dashboard.io import load_file


def audit_all_files_for_notch(data_dir: str | Path = "data/raw") -> None:
    """
    Recorre todos los archivos TDF y canales, imprimiendo un reporte 
    de cuáles canales necesitan obligatoriamente el filtro Notch de 60 Hz.
    """
    data_dir = Path(data_dir)
    tdf_files = sorted(data_dir.glob("*.tdf"))
    
    if not tdf_files:
        print(f"No se encontraron archivos en {data_dir}")
        return

    print("--- REPORTE DE AUDITORÍA DE RUIDO ELÉCTRICO (60 Hz) ---")
    for file_path in tdf_files:
        df, meta = load_file(file_path)
        fs = meta["fs"]
        
        print(f"\nArchivo: {file_path.name}")
        for canal in df["channel"].unique():
            x_musculo = df[df["channel"] == canal]["emg"].values
            
            # Evaluamos ruido de 60 Hz
            resultado = detect_powerline_interference(x_musculo, fs, target_hz=60.0)
            
            if resultado["has_noise"]:
                print(f"  [!] Músculo '{canal}': ⚠️ REQUIERE NOTCH (Ratio 60Hz/Fondo: {resultado['ratio']:.2f})")
            else:
                print(f"  [✓] Músculo '{canal}': Limpio de 60 Hz (Ratio: {resultado['ratio']:.2f})")

if __name__ == "__main__":
    audit_all_files_for_notch()
