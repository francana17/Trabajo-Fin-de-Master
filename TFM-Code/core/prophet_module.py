# core/prophet_module.py
import pandas as pd
from prophet import Prophet


class ProphetModel:
    """
    Wrapper de Prophet con parámetros afinados para series diarias y
    futuro continuo (arranca en last+1 día, sin “punto suelto”).
    """
    def __init__(
        self,
        weekly_seasonality: bool = True,
        yearly_seasonality: bool = False,
        daily_seasonality: bool = False,
        seasonality_mode: str = "additive",       # "additive" o "multiplicative"
        changepoint_prior_scale: float = 0.05,    # menor => tendencia más suave
        changepoint_range: float = 0.9,           # proporción del histórico con cambios
        seasonality_prior_scale: float = 10.0,
        interval_width: float = 0.80              # banda al 80%
    ):
        self.cfg = dict(
            weekly_seasonality=weekly_seasonality,
            yearly_seasonality=yearly_seasonality,
            daily_seasonality=daily_seasonality,
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=changepoint_prior_scale,
            changepoint_range=changepoint_range,
            seasonality_prior_scale=seasonality_prior_scale,
            interval_width=interval_width,
        )
        self._m: Prophet | None = None
        self._last = None

    def fit(self, df_ds_y: pd.DataFrame):
        """
        df_ds_y: DataFrame con columnas ['ds','y'] (diario).
        """
        df = df_ds_y.rename(columns={"ds": "ds", "y": "y"}).copy()
        df["ds"] = pd.to_datetime(df["ds"])
        df = df.sort_values("ds").dropna(subset=["y"])
        self._last = df["ds"].max()

        m = Prophet(
            weekly_seasonality=self.cfg["weekly_seasonality"],
            yearly_seasonality=self.cfg["yearly_seasonality"],
            daily_seasonality=self.cfg["daily_seasonality"],
            seasonality_mode=self.cfg["seasonality_mode"],
            changepoint_prior_scale=self.cfg["changepoint_prior_scale"],
            changepoint_range=self.cfg["changepoint_range"],
            seasonality_prior_scale=self.cfg["seasonality_prior_scale"],
            interval_width=self.cfg["interval_width"],
        )
        # Estacionalidad opcional adicional (ej. mensual):
        # m.add_seasonality(name="mensual", period=30.5, fourier_order=5)

        m.fit(df[["ds", "y"]])
        self._m = m

    def forecast(self, horizon_days: int) -> pd.DataFrame:
        if self._m is None or self._last is None:
            raise RuntimeError("Llama a fit() antes de forecast().")

        h = int(horizon_days)
        # Futuro que empieza EXACTAMENTE en el día siguiente
        future = pd.DataFrame({
            "ds": pd.date_range(self._last + pd.Timedelta(days=1), periods=h, freq="D")
        })
        fcst = self._m.predict(future)
        return fcst[["ds", "yhat", "yhat_lower", "yhat_upper"]]
