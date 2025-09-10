# app.py
import os
import importlib.util
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from core.data_loader import (
    normalize_columns, parse_date_auto, detect_format,
    candidates_types, prepare_series
)
from core.prophet_module import ProphetModel
from core.utils import plot_forecast_focus
from core.stats_module import ClassicTSModel, ClassicConfig  # módulo clásico SARIMAX/ETS

# Opcional: silenciar logs de TF
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ========================
# Configuración principal
# ========================
st.set_page_config(page_title="TFM Aceite — Prophet / LSTM / Clásicos", layout="wide")

# ---------------- Título general (centrado) ----------------
st.markdown(
    """
    <h1 style="text-align:center; margin: 0 0 6px 0;">
        🫒 Predicción del precio del aceite 🫒
    </h1>
    """,
    unsafe_allow_html=True
)

# ========================
# Función de gráfico histórico (ejes mejorados)
# ========================
def plot_history_series(df: pd.DataFrame, tipo: str):
    if df.empty or "ds" not in df.columns or "y" not in df.columns:
        fig = go.Figure()
        fig.update_layout(height=300, title="Serie vacía o columnas no válidas")
        return fig

    ymin = 1.0
    ymax = float(df["y"].max()) if pd.api.types.is_numeric_dtype(df["y"]) else 2.0
    if ymax <= ymin:
        ymax = ymin + 1.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["ds"], y=df["y"],
        mode="lines+markers", name="Histórico",
        line=dict(color="#2563eb", width=2),
        marker=dict(size=4, color="#1d4ed8"),
        hovertemplate="%{x|%Y-%m-%d}<br>Precio: %{y:.3f} €<extra></extra>"
    ))

    fig.update_layout(
        template="plotly_white",   # plantilla clara para que los ticks se vean en tema oscuro
        title=f"Evolución histórica — {tipo}",
        xaxis_title="Fecha",
        yaxis_title="Precio (EUR/kg)",
        height=450,
        margin=dict(l=50, r=20, t=60, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(
        showgrid=True, gridcolor="rgba(0,0,0,0.08)",
        tickfont=dict(size=12, color="#222"),
        showticklabels=True, ticks="outside", ticklen=6
    )
    fig.update_yaxes(
        range=[ymin, ymax + 0.5],
        showgrid=True, gridcolor="rgba(0,0,0,0.08)",
        tickfont=dict(size=12, color="#222"),
        showticklabels=True, ticks="outside", ticklen=6
    )
    return fig

# ========================
# Descubrir LSTM opcional
# ========================
spec_lstm = importlib.util.find_spec("core.lstm_module")
HAS_LSTM = spec_lstm is not None
if HAS_LSTM:
    from core.lstm_module import LSTMModel

# ---------------- Sidebar: título y carga ----------------

st.sidebar.header("Datos de entrada")

up = st.sidebar.file_uploader("Sube CSV (ancho o tidy)", type=["csv"])
sep = st.sidebar.selectbox("Separador", [",",";","\\t"], index=0,
                           format_func=lambda s: {";":"Punto y coma","\\t":"Tabulador"}.get(s, "Coma"))
dec = st.sidebar.selectbox("Decimal", [".",","], index=0)
interp = st.sidebar.checkbox("Interpolar huecos diarios", value=True)
ffill = st.sidebar.checkbox("Forward-fill", value=False)
h = st.sidebar.number_input("Horizonte (días)", 1, 365, 30)
context_days = st.sidebar.number_input("Días de contexto (histórico mostrado)", 3, 60, 7, step=1)

# Modelos
model_options = ["Prophet", "Clásico (SARIMAX/ETS)"]
if HAS_LSTM:
    model_options.append("LSTM")
modelo = st.sidebar.selectbox("Modelo", model_options, index=0)

# Config extras para Clásicos
classic_type = None
if modelo == "Clásico (SARIMAX/ETS)":
    classic_type = st.sidebar.selectbox("Tipo clásico", ["SARIMAX", "ETS"], index=0)
    sp = st.sidebar.number_input("Periodo estacional (días)", 1, 365, 7)
    st.sidebar.caption("SARIMAX: orden (1,1,1)(1,1,1,s) por defecto. ETS: tendencia aditiva; marca estacional si procede.")
    use_seasonal_ets = st.sidebar.checkbox("ETS con estacionalidad", value=False)

with st.sidebar.expander("Validación / Backtest"):
    do_backtest = st.checkbox("Hacer backtest (cortar histórico)", value=False)
    h_bt = st.number_input("Horizonte backtest (días)", 1, 365, min(h, 30))

DEFAULT_CSV = os.environ.get("CSV_PATH")

# ========================
# Carga CSV
# ========================
if up is not None:
    df_raw = pd.read_csv(up, sep=sep, decimal=dec)
elif DEFAULT_CSV and os.path.exists(DEFAULT_CSV):
    df_raw = pd.read_csv(DEFAULT_CSV)
    st.sidebar.info(f"Usando CSV por defecto: {DEFAULT_CSV}")
else:
    df_raw = None
    st.info("Sube un CSV para continuar.")

# ========================
# Si hay CSV cargado
# ========================
if df_raw is not None:
    # Normalizar + fecha robusta
    df_raw = normalize_columns(df_raw)
    try:
        df_raw = parse_date_auto(df_raw)
    except Exception as e:
        st.error(str(e)); st.stop()

    fmt = detect_format(df_raw)
    tipos = candidates_types(df_raw, fmt)
    if not tipos:
        st.error("No se detectaron columnas/categorías de precio."); st.stop()

    tipo = st.sidebar.selectbox("Tipo / columna", tipos, index=0)

    # Vista previa
    st.subheader("Vista previa del CSV")
    st.dataframe(df_raw.head(20), use_container_width=True)

    # Serie limpia (antes de filtrar por rango)
    is_tidy = (fmt == "tidy")
    serie_all = prepare_series(df_raw, tipo, is_tidy, interp=interp, ffill=ffill)

    # -------- Selector de RANGO DE FECHAS (filtrado) --------
    min_d = pd.to_datetime(serie_all["ds"].min()).date()
    max_d = pd.to_datetime(serie_all["ds"].max()).date()
    rango = st.sidebar.date_input(
        "Rango de fechas a usar",
        (min_d, max_d),
        min_value=min_d,
        max_value=max_d,
        help="Filtra el histórico para entrenar/validar sin tocar el CSV original."
    )
    if isinstance(rango, tuple) and len(rango) == 2:
        start_d, end_d = rango
    else:
        # compat: un único valor seleccionado
        start_d, end_d = min_d, rango

    serie = serie_all[
        (serie_all["ds"] >= pd.Timestamp(start_d)) &
        (serie_all["ds"] <= pd.Timestamp(end_d))
    ].copy()

    if len(serie) < 10:
        st.warning("El rango seleccionado tiene muy pocos datos. Amplía el rango para entrenar.")
    st.markdown(f"**Filas tras limpieza** (rango aplicado): {len(serie)}")

    # Gráfico histórico filtrado
    st.plotly_chart(plot_history_series(serie, tipo), use_container_width=True)

    # -------- Ejecutar --------
    if st.button("Calcular predicción"):
        last = serie["ds"].max()
        hist_tail = serie[serie["ds"] >= (last - pd.Timedelta(days=int(context_days)))][["ds","y"]].copy()

        # ============== Prophet ==============
        if modelo == "Prophet":
            m = ProphetModel()
            m.fit(serie.rename(columns={"ds":"ds","y":"y"}))
            fcst = m.forecast(h)
            future = fcst.loc[fcst["ds"] > last, ["ds","yhat","yhat_lower","yhat_upper"]].copy()

            st.subheader("Predicción (Prophet)")
            st.plotly_chart(plot_forecast_focus(hist_tail, future, last, f"Prophet — {tipo}"),
                            use_container_width=True)

            out = future.rename(columns={"ds":"fecha","yhat":"prediccion","yhat_lower":"lo","yhat_upper":"hi"})
            st.download_button("Descargar predicciones (CSV)", out.to_csv(index=False), "pred_prophet.csv")

            if do_backtest:
                try:
                    hbt = int(h_bt)
                    if len(serie) <= hbt + 5:
                        raise ValueError("Serie demasiado corta para ese backtest.")
                    train = serie.iloc[:-hbt].copy()
                    test = serie.iloc[-hbt:].copy()
                    m_bt = ProphetModel(); m_bt.fit(train.rename(columns={"ds":"ds","y":"y"}))
                    fcst_bt = m_bt.forecast(hbt)
                    fcst_bt_tail = fcst_bt[fcst_bt["ds"].isin(test["ds"])]
                    if len(fcst_bt_tail) != len(test): fcst_bt_tail = fcst_bt_tail.tail(len(test))
                    from sklearn.metrics import mean_absolute_error, mean_squared_error
                    mae = mean_absolute_error(test["y"], fcst_bt_tail["yhat"])
                    rmse = mean_squared_error(test["y"], fcst_bt_tail["yhat"], squared=False)
                    st.success(f"Backtest Prophet {hbt} días → MAE={mae:.3f} | RMSE={rmse:.3f}")
                except Exception as e:
                    st.warning(f"No se pudo ejecutar backtest Prophet: {e}")

        # ============== Clásico (SARIMAX/ETS) ==============
        elif modelo == "Clásico (SARIMAX/ETS)":
            cfg = ClassicConfig(
                model_type="ets" if (classic_type or "").lower().startswith("ets") else "sarimax",
                seasonal_periods=int(sp) if model_options else 7,
                seasonal="add" if (classic_type or "").lower().startswith("ets") and use_seasonal_ets else None,
                trend="add" if (classic_type or "").lower().startswith("ets") else None,
                alpha_band=0.2,
            )
            try:
                cm = ClassicTSModel(cfg)
                cm.fit(serie)
                fc = cm.forecast(h)  # ds, yhat, yhat_lower, yhat_upper
                future = fc[fc["ds"] > last].copy()

                st.subheader(f"Predicción ({'ETS' if cfg.model_type=='ets' else 'SARIMAX'})")
                st.plotly_chart(
                    plot_forecast_focus(hist_tail, future, last, f"{'ETS' if cfg.model_type=='ets' else 'SARIMAX'} — {tipo}"),
                    use_container_width=True
                )

                out = future.rename(columns={"ds":"fecha","yhat":"prediccion","yhat_lower":"lo","yhat_upper":"hi"})
                st.download_button("Descargar predicciones (CSV)", out.to_csv(index=False), "pred_clasico.csv")

                if do_backtest:
                    try:
                        hbt = int(h_bt)
                        if len(serie) <= hbt + 5:
                            raise ValueError("Serie demasiado corta para ese backtest.")
                        train = serie.iloc[:-hbt].copy()
                        test = serie.iloc[-hbt:].copy()
                        cm_bt = ClassicTSModel(cfg)
                        cm_bt.fit(train)
                        fc_bt = cm_bt.forecast(hbt)
                        eval_bt = fc_bt[fc_bt["ds"].isin(test["ds"])]
                        if len(eval_bt) != len(test): eval_bt = eval_bt.tail(len(test))
                        from sklearn.metrics import mean_absolute_error, mean_squared_error
                        mae = mean_absolute_error(test["y"], eval_bt["yhat"])
                        rmse = mean_squared_error(test["y"], eval_bt["yhat"], squared=False)
                        st.success(f"Backtest {('ETS' if cfg.model_type=='ets' else 'SARIMAX')} {hbt} días → MAE={mae:.3f} | RMSE={rmse:.3f}")
                    except Exception as e:
                        st.warning(f"No se pudo ejecutar backtest clásico: {e}")
            except Exception as e:
                st.error(f"Error en modelo clásico: {e}")

        # ============== LSTM ==============
        else:
            if not HAS_LSTM:
                st.error("LSTM no disponible. Instala 'tensorflow-cpu' y confirma que 'core/lstm_module.py' existe.")
            else:
                try:
                    lstm = LSTMModel(lookback=60)
                    lstm.fit(serie, epochs=20, verbose=0)
                    preds = lstm.forecast(serie, horizon=h)  # ds, yhat
                    future = preds.copy()
                    future["yhat_lower"] = None
                    future["yhat_upper"] = None

                    st.subheader("Predicción (LSTM)")
                    st.plotly_chart(
                        plot_forecast_focus(hist_tail, future, last, f"LSTM — {tipo}"),
                        use_container_width=True
                    )

                    out = future.rename(columns={"ds":"fecha","yhat":"prediccion"})
                    st.download_button("Descargar predicciones (CSV)", out.to_csv(index=False), "pred_lstm.csv")

                    if do_backtest:
                        try:
                            hbt = int(h_bt)
                            if len(serie) <= hbt + 60 + 1:
                                raise ValueError("Serie demasiado corta para backtest con LSTM (revisa lookback/horizonte).")
                            train = serie.iloc[:-hbt].copy()
                            test = serie.iloc[-hbt:].copy()
                            lstm_bt = LSTMModel(lookback=60)
                            lstm_bt.fit(train, epochs=20, verbose=0)
                            preds_bt = lstm_bt.forecast(train, horizon=hbt)
                            eval_bt = preds_bt[preds_bt["ds"].isin(test["ds"])]
                            if len(eval_bt) != len(test): eval_bt = eval_bt.tail(len(test))
                            from sklearn.metrics import mean_absolute_error, mean_squared_error
                            mae = mean_absolute_error(test["y"], eval_bt["yhat"])
                            rmse = mean_squared_error(test["y"], eval_bt["yhat"], squared=False)
                            st.success(f"Backtest LSTM {hbt} días → MAE={mae:.3f} | RMSE={rmse:.3f}")
                        except Exception as e:
                            st.warning(f"No se pudo ejecutar backtest LSTM: {e}")
                except Exception as e:
                    st.error(f"Error al ejecutar LSTM: {e}")
