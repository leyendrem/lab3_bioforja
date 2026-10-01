from __future__ import annotations

from collections.abc import Iterable, Sequence

import plotly.graph_objects as go

# Color de las franjas de activación (verde translúcido).
ACTIVATION_FILL = "rgba(46, 160, 67, 0.18)"


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
    #fig.add_trace(go.Scattergl(x=time, y=raw, name=raw_label, line=dict(width=1), opacity=0.45))
    fig.add_trace(
        go.Scattergl(
            x=time,
            y=raw,
            name=raw_label,
            line={"width": 1},
            opacity=0.45,
        )
    )
    fig.add_trace(go.Scattergl(x=time, y=filtered, name="Filtrada", line={"width": 1}))
    fig.add_trace(go.Scattergl(x=time, y=envelope, name="Envolvente", line={"width": 2}))

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
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines+markers"))
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
    fig = go.Figure(go.Scatter(x=freq, y=power, mode="lines"))
    fig.update_layout(
        title=title,
        xaxis_title="Frecuencia (Hz)",
        yaxis_title=y_title,
        yaxis_type="log" if log_y else "linear",
    )
    return fig
