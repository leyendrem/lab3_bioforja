# Dashboard EMG · BioForja

Proyecto académico reproducible para transformar registros de electromiografía de superficie (sEMG) en métricas y visualizaciones interpretables durante actividad física.

## Problema de monitoreo

Existe incertidumbre sobre la correcta ejecución técnica de los ejercicios y sobre la presencia de patrones de activación potencialmente asociados con sobreesfuerzo, fatiga o compensación. El dashboard busca reducir esa incertidumbre mediante una caracterización cuantitativa y trazable de la señal EMG, sin convertir las métricas en diagnósticos automáticos.

## Pregunta rectora

**¿Cómo transformar una señal EMG adquirida durante una actividad física en un conjunto reducido de métricas y visualizaciones que un médico deportivo pueda interpretar, sin ocultar las decisiones de procesamiento que condicionan esos resultados?**

## Flujo

`carga → validación → inspección → preprocesamiento → segmentación → métricas → visualización → interpretación`

## Estructura

```text
app.py                          # dashboard Streamlit (solo orquesta)
pyproject.toml / uv.lock        # dependencias bloqueadas
data/raw/                       # 3 registros .tdf originales (inmutables)
data/processed/                 # salidas de scripts/reproduce.py
src/emg_dashboard/
  io.py                         # load_file, load_bts_tdf, build_emg_dataframe, load_directory, validate_contract
  data_validation.py            # sampling_report, channel_quality, detect_powerline_interference
  preprocessing_metrics.py      # remove_dc, bandpass_emg, notch_filter, rectify, lowpass_envelope, preprocess_emg
  metrics.py                    # moving_rms, rms, iemg, waveform_length, percent_mvc, median_frequency(+trend), activation_segments, segment_summary, window_sensitivity
  visualization.py              # signal_figure, trend_figure, spectrum_figure
  data_filtration.py / data_compilation.py   # exportan CSV procesados y JSON de metadatos
tests/                          # pytest (28 pruebas)
scripts/reproduce.py
reports/informe_base.md
```

## Datos

- **Origen:** 3 registros BTS Bioengineering TDF (`0033~aa~flexiones_brazos{,2,3}.tdf`), tarea de **flexiones de brazos**.
- **Estructura:** 8 canales sEMG por registro (bíceps, deltoides anterior, pectoral mayor y tríceps, izquierdo y derecho); 61–65 s por registro.
- **Frecuencia de muestreo:** **1000 Hz** (leída del bloque EMG del TDF; Nyquist = 500 Hz).
- **Unidad:** el bloque TDF no declara unidad; los valores (~±1 mV) son consistentes con **voltios**. Debe confirmarse con el equipo.
- **Eventos:** los TDF no traen eventos; si existieran se dibujan en la señal.
- Cada archivo es una **sesión**; el dashboard los carga de `data/raw/` y permite elegir sesión, canal e intervalo.

### Contrato mínimo

| Variable | Significado |
|---|---|
| `time_s` | tiempo en segundos |
| `emg` | amplitud EMG |
| `channel` | músculo/canal |
| `session` | sesión/repetición |
| `event` | fase/tarea, opcional |

## Instalación

Requiere Python 3.12 y `uv`.

```bash
uv sync
uv run pytest
```

## Ejecutar

```bash
uv run streamlit run app.py
```

Por defecto carga los TDF de `data/raw/`. También permite subir un `.tdf` o usar un registro demostrativo sintético.

Pipeline por lotes (CSV procesados y resumen JSON): `uv run python scripts/reproduce.py`.

## Procesamiento y evidencia

| Parámetro | Valor | Evidencia en los registros |
|---|---|---|
| DC | resta de la media | Media ≈ −4·10⁻⁷ V antes y ≈ 0 después; la forma no cambia (std 1.82·10⁻⁴ → 1.76·10⁻⁴ V tras el pasa-banda). |
| Pasa-altas | 20 Hz | Entre 2 % y 27 % de la potencia está < 20 Hz (movimiento/deriva). Con 10/20/30 Hz la MDF de un canal pasa de 46.5/48.5/53 Hz: 30 Hz ya recorta contenido útil. |
| Pasa-bajas | 450 Hz | En todos los canales el 99 % de la potencia queda por debajo de ~330 Hz; 450 Hz < Nyquist (500 Hz). |
| Notch | **no** | Razón pico 60 Hz / fondo = 0.79–1.42 (umbral 3.5): no hay pico estrecho. Se puede activar desde la interfaz. |
| Envolvente | 5 Hz | Las repeticiones duran ~2 s (≈ 0.5 Hz). Con 1 Hz se fusionan repeticiones (27 picos frente a 33–35 con 2–10 Hz); 5 Hz las sigue sin conservar fluctuaciones rápidas. |
| Filtrado | `sosfiltfilt` | Fase cero, solo **offline**. |

El filtrado se aplica al canal completo y **después** se segmenta, para evitar transitorios de borde.

### Ventanas y análisis de sensibilidad

El dashboard incluye el experimento *menor · elegida · mayor* (sección «Sensibilidad a ventanas»). Ejemplo, registro 1, pectoral derecho (registro completo):

| Métrica | Ventana | Variabilidad (cv) | Puntos |
|---|---|---|---|
| RMS | 100 ms | 0.67 | 61063 |
| RMS | **200 ms** | 0.63 | 61063 |
| RMS | 400 ms | 0.59 | 61063 |
| MDF | 1 s | 11.6 % | 121 |
| MDF | **2 s** | 9.1 % | 60 |
| MDF | 4 s | 7.4 % | 29 |

- **RMS 200 ms** (≈ 1/10 de una repetición): 100 ms sigue la actividad pero es más ruidoso; 400 ms empieza a fusionar fases concéntrica/excéntrica.
- **MDF 2 s** con solapamiento 50 %: 1 s da una estimación más variable y menor resolución frecuencial; 4 s deja solo ~2 repeticiones por ventana y viola la estacionariedad local.
- **Activación:** umbral = mediana + 3·1.4826·MAD del primer segundo del registro, duración mínima 0.10 s. Es una heurística; si el primer segundo no es reposo el umbral se eleva.

## Métricas y clasificación

El dashboard construye un vector de características con RMS, %MVC, MDF, iEMG, WL, número de activaciones y duración media de activación. Estas variables son candidatas para clasificar actividades cuando el dataset tenga etiquetas y suficientes repeticiones. No se entrena un modelo automático en esta versión porque la guía se centra en la trazabilidad de las métricas y no aporta un conjunto etiquetado para validar un clasificador.

## Métricas

### RMS

Descriptor de amplitud eléctrica dentro de una ventana. Un aumento indica mayor amplitud EMG, pero no implica linealmente mayor fuerza ni lesión.

### %MVC

`100 × RMS_segmento / RMS_MVC`.

Se calcula solo cuando se introduce una referencia MVC real y compatible. El proyecto no inventa una MVC.

### MDF

Frecuencia que divide en dos el área de la densidad espectral de potencia. Una reducción progresiva puede ser compatible con manifestaciones mioeléctricas de fatiga en contracciones sostenidas y controladas, pero su interpretación es sensible a movimiento, cambios de fuerza y no estacionariedad.

### Inicio, fin y duración

Se detectan sobre la envolvente usando un umbral transparente basado en mediana + múltiplo de MAD y una duración mínima. Es una heurística de análisis, no un umbral clínico.

### iEMG

Actividad absoluta acumulada en el intervalo. Depende de la duración, por lo que no debe compararse sin controlar la longitud del segmento.

### WL

`sum(abs(x[n+1]-x[n]))`. Resume cuánto cambia la señal en el intervalo y puede ser sensible al ruido.

## Interpretación fisiológica

La sEMG superficial registra la superposición espacial y temporal de potenciales de acción de unidades motoras activas. La amplitud observada depende, entre otros factores, de activación neural, propagación, geometría del volumen conductor, colocación de electrodos, tejido subcutáneo, movimiento y cross-talk.

Por eso el dashboard presenta **tendencias e indicadores** y no etiquetas “normal/anormal”, semáforos o diagnósticos.

## Qué puede observar el especialista

- aumento o disminución de la amplitud;
- cambios de RMS a lo largo de la sesión;
- activaciones más prolongadas;
- cambios de inicio/fin entre repeticiones;
- desplazamientos de MDF;
- diferencias entre músculos/canales;
- segmentos con baja calidad o posible contaminación por movimiento;
- sensibilidad de los resultados a las ventanas y filtros.

## Limitaciones

1. RMS no es sinónimo de fuerza ni de sobrecarga.
2. MDF no es un diagnóstico de fatiga.
3. Una activación prolongada no demuestra mala técnica por sí sola.
4. Sin una referencia MVC válida no debe reportarse %MVC como si fuera absoluto.
5. No se infiere riesgo de lesión mediante un único umbral.
6. Los resultados dependen de adquisición, electrodos, movimiento, cross-talk, tarea y ventanas.
7. `sosfiltfilt` es un procesamiento offline; no representa un filtro causal de tiempo real.

## Pruebas

```bash
uv run pytest
```

Cubren métricas (señal corta, NaN, ventana mayor que la señal, MDF constante), filtros inválidos, validación de datos y contrato.

Además, antes de exponer se deben comprobar: cambio de sesión, cambio de canal, intervalos cortos/largos, registros con pocas muestras, datos faltantes, filtros inválidos y actividad muy baja.

## Reproducibilidad

Los datos originales de `data/raw/` no deben modificarse. Los resultados derivados deben guardarse aparte.

## Limitaciones adicionales

- La tarea (flexiones) es **dinámica**: la MDF no es estacionaria y no debe leerse como fatiga confirmada.
- La unidad (V) es supuesta; el %MVC exige una MVC adquirida con el mismo protocolo (no disponible en estos registros).
- Sin etiquetas de ejercicio no se entrena ningún clasificador.
