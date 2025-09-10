# core/stats_module.py
from __future__ import annotations
import warnings
from dataclasses import dataclass
import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing


@dataclass
class ClassicConfig:
    model_type: str = "sarimax"          # "sarimax" | "ets"
    order: tuple[int, int, int] = (1, 1, 1)
    seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 7)  # (P,D,Q,s)
    trend: str | None = "add"            # ETS: "add" | "mul" | None
    seasonal: str | None = None          # ETS: "add" | "mul" | None
    seasonal_periods: int = 7            # ETS/SARIMAX
    alpha_band: float = 0.2              # banda ~80%


class ClassicTSModel:
    """
    Wrapper sencillo para modelos clásicos:
      - SARIMAX (ARIMA + estacionalidad)
      - ETS (Holt-Winters)

    Espera un DataFrame con columnas ['ds','y'].
    """
    def __init__(self, cfg: ClassicConfig | None = None):
        self.cfg = cfg or ClassicConfig()
        self._fitted_model = None
        self._last_index = None
        self._is_ets = self.cfg.model_type.lower() == "ets"

    @staticmethod
    def _prep(df: pd.DataFrame) -> pd.Series:
        if not {"ds", "y"}.issubset(df.columns):
            raise ValueError("El DataFrame debe tener columnas ['ds','y'].")
        s = df.copy()
        s["ds"] = pd.to_datetime(s["ds"])
        s = s.sort_values("ds").dropna(subset=["y"])
        y = s.set_index("ds")["y"].asfreq("D")
        # rellenar huecos pequeños de días con interpolación temporal/ffill
        y = y.interpolate("time").ffill().bfill()
        return y

    def fit(self, df: pd.DataFrame):
        y = self._prep(df)
        self._last_index = y.index.max()

        if self._is_ets:
            sp = max(1, int(self.cfg.seasonal_periods))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ExponentialSmoothing(
                    y,
                    trend=self.cfg.trend,
                    seasonal=self.cfg.seasonal,
                    seasonal_periods=sp if self.cfg.seasonal else None,
                    initialization_method="estimated",
                ).fit(optimized=True)
        else:
            # SARIMAX
            P, D, Q, s = self.cfg.seasonal_order
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = SARIMAX(
                    y,
                    order=self.cfg.order,
                    seasonal_order=(P, D, Q, s),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)

        self._fitted_model = model

    def forecast(self, horizon: int) -> pd.DataFrame:
        if self._fitted_model is None:
            raise RuntimeError("Modelo no entrenado. Llama a fit() primero.")

        h = int(horizon)
        if h < 1:
            raise ValueError("horizon debe ser >= 1")

        if self._is_ets:
            # ETS no devuelve PI nativo; generamos una banda simple usando residuales
            fc = self._fitted_model.forecast(h)
            idx = pd.date_range(self._last_index + pd.Timedelta(days=1), periods=h, freq="D")
            yhat = pd.Series(fc, index=idx)
            resid = getattr(self._fitted_model, "resid", None)
            sigma = float(np.nanstd(resid)) if resid is not None else 0.0
            z = 1.28  # ~80%
            lo = yhat - z * sigma
            hi = yhat + z * sigma
        else:
            # SARIMAX sí da PI
            res = self._fitted_model.get_forecast(steps=h)
            idx = pd.date_range(self._last_index + pd.Timedelta(days=1), periods=h, freq="D")
            yhat = pd.Series(res.predicted_mean, index=idx)
            alpha = float(self.cfg.alpha_band)
            conf = res.conf_int(alpha=alpha)
            # statsmodels devuelve columnas ['lower y','upper y'] (o similar)
            lo = pd.Series(conf.iloc[:, 0].values, index=idx)
            hi = pd.Series(conf.iloc[:, 1].values, index=idx)

        out = pd.DataFrame(
            {"ds": idx, "yhat": yhat.values, "yhat_lower": lo.values, "yhat_upper": hi.values}
        )
        return out
