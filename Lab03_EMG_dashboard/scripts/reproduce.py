from emg_dashboard.data_filtration import process_and_save_signals
from emg_dashboard.data_compilation import process_and_export_summary

if __name__ == "__main__":

    # Filtro y procesamiento de las señales crudas.
    process_and_save_signals()
    
    print("\n--------------------------------------------------\n")
    
    # Validación de contratos y generación del resumen consolidado de metadatos.
    process_and_export_summary()
    
    print("¡Pipeline ejecutado y exportado con éxito!")