# core/utils.py
import plotly.graph_objects as go
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def plot_forecast_focus(history_tail: pd.DataFrame, future_fcst: pd.DataFrame, cutoff, title: str):
    """
    history_tail: DataFrame con columnas ['ds','y'] (solo últimos N días reales)
    future_fcst : DataFrame con columnas ['ds','yhat','yhat_lower','yhat_upper'] (solo futuro)
    cutoff      : timestamp/fecha del último dato real
    """
    fig = go.Figure()

    # Banda de confianza del futuro
    if {"yhat_lower","yhat_upper"}.issubset(future_fcst.columns):
        fig.add_trace(go.Scatter(
            x=future_fcst["ds"], y=future_fcst["yhat_upper"],
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter(
            x=future_fcst["ds"], y=future_fcst["yhat_lower"],
            mode="lines", fill="tonexty",
            fillcolor="rgba(239, 68, 68, 0.15)",  # rojo suave
            line=dict(width=0), name="Banda", hoverinfo="skip"
        ))

    # Predicción
    fig.add_trace(go.Scatter(
        x=future_fcst["ds"], y=future_fcst["yhat"],
        mode="lines", name="Predicción",
        line=dict(color="#ef4444", width=3),  # rojo
        hovertemplate="%{x|%Y-%m-%d}<br>Predicción: %{y:.3f} €<extra></extra>"
    ))

    # Histórico (últimos N días)
    if not history_tail.empty:
        fig.add_trace(go.Scatter(
            x=history_tail["ds"], y=history_tail["y"],
            mode="lines+markers", name="Histórico (últimos días)",
            line=dict(color="#2563eb", width=2),  # azul
            marker=dict(size=5),
            hovertemplate="%{x|%Y-%m-%d}<br>Real: %{y:.3f} €<extra></extra>"
        ))

    # Línea vertical de corte (último real)
    fig.add_vline(x=cutoff, line_width=2, line_dash="dot", line_color="#64748b")  # gris
    fig.update_layout(
        title=title,
        xaxis_title="Fecha", yaxis_title="Precio (EUR/kg)",
        hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    return fig

def compute_metrics(df_real: pd.DataFrame, df_pred: pd.DataFrame):
    merged = df_real.merge(df_pred, on="ds", how="inner")
    if len(merged) < 3:
        return None, None
    mae = mean_absolute_error(merged["y"], merged["yhat"])
    rmse = mean_squared_error(merged["y"], merged["yhat"], squared=False)
    return mae, rmse
