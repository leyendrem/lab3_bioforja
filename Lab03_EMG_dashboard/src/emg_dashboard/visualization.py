from __future__ import annotations

from collections.abc import Iterable, Sequence

import plotly.graph_objects as go

# Color de las franjas de activación (verde translúcido).
ACTIVATION_FILL = "rgba(46, 160, 67, 0.18)"


def _matched_or_empty(x: Sequence[float], y: Sequence[float]) -> tuple[list, list]:
    """Para series que deben compartir el mismo eje x (p. ej. cruda/filtrada/
    envolvente sincronizadas en el tiempo): si x e y no tienen la misma longitud,
    no se puede garantizar la correspondencia punto a punto, así que se devuelve
    una traza vacía en vez de datos desalineados."""
    if len(x) != len(y):
        return [], []
    return list(x), list(y)


def _truncate_to_shortest(x: Sequence[float], y: Sequence[float]) -> tuple[list, list]:
    """Para series independientes (una sola curva, sin otras series sincronizadas):
    si x e y no coinciden en longitud, se recorta al tamaño común más corto en vez
    de fallar o desalinear."""
    n = min(len(x), len(y))
    return list(x)[:n], list(y)[:n]


def signal_figure(
    time: Sequence[float],
    raw: Sequence[float],
    filtered: Sequence[float],
    envelope: Sequence[float],
    title: str = "EMG: cruda, filtrada y envolvente",
    activation_segments: Iterable[dict] | None = None,
    raw_label: str = "Cruda",
    y_title: str = "Amplitud EMG",
) -> go.Figure:
    fig = go.Figure()
    raw_x, raw_y = _matched_or_empty(time, raw)
    fig.add_trace(
        go.Scattergl(
            x=raw_x,
            y=raw_y,
            name=raw_label,
            line={"width": 1},
            opacity=0.45,
        )
    )
    filt_x, filt_y = _matched_or_empty(time, filtered)
    fig.add_trace(go.Scattergl(x=filt_x, y=filt_y, name="Filtrada", line={"width": 1}))
    env_x, env_y = _matched_or_empty(time, envelope)
    fig.add_trace(go.Scattergl(x=env_x, y=env_y, name="Envolvente", line={"width": 2}))

    for seg in activation_segments or []:
        fig.add_vrect(
            x0=seg["start_s"],
            x1=seg["end_s"],
            fillcolor=ACTIVATION_FILL,
            line_width=0,
            layer="below",
        )
    fig.update_layout(
        title=title,
        xaxis_title="Tiempo (s)",
        yaxis_title=y_title,
        hovermode="x unified",
    )
    return fig


def trend_figure(
    x: Sequence[float],
    y: Sequence[float],
    title: str,
    y_title: str,
) -> go.Figure:
    """Tendencia temporal de una métrica (RMS móvil, MDF por ventanas...)."""
    x2, y2 = _truncate_to_shortest(x, y)
    fig = go.Figure(go.Scatter(x=x2, y=y2, mode="lines+markers"))
    fig.update_layout(
        title=title,
        xaxis_title="Tiempo (s)",
        yaxis_title=y_title,
        hovermode="x unified",
    )
    return fig


def spectrum_figure(
    freq: Sequence[float],
    power: Sequence[float],
    title: str = "Densidad espectral de potencia",
    y_title: str = "PSD",
    log_y: bool = False,
) -> go.Figure:
    freq2, power2 = _truncate_to_shortest(freq, power)
    fig = go.Figure(go.Scatter(x=freq2, y=power2, mode="lines"))
    fig.update_layout(
        title=title,
        xaxis_title="Frecuencia (Hz)",
        yaxis_title=y_title,
        yaxis_type="log" if log_y else "linear",
    )
    return fig
