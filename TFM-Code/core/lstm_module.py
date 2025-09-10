# core/lstm_module.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    import tensorflow as tf
except Exception:
    tf = None


class LSTMModel:
    """
    Modelo LSTM minimalista y robusto para series univariantes.
    Espera un DataFrame con columnas ['ds','y'] ordenadas por fecha.
    """

    def __init__(self, lookback: int = 60, scaler: StandardScaler | None = None):
        if tf is None:
            raise ImportError(
                "TensorFlow no está instalado. Instala 'tensorflow-cpu' para usar LSTM."
            )
        self.lookback = int(lookback)
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.model: tf.keras.Model | None = None
        self._fitted = False

    # ----------------- utils -----------------
    @staticmethod
    def _check_df(df_y: pd.DataFrame):
        if not {"ds", "y"}.issubset(df_y.columns):
            raise ValueError("df_y debe tener columnas ['ds','y'].")
        if df_y.empty:
            raise ValueError("La serie está vacía.")

        # asegurar orden temporal y sin NaN en y
        df = df_y.copy()
        df["ds"] = pd.to_datetime(df["ds"])
        df = df.sort_values("ds").dropna(subset=["y"])
        return df

    def _make_sequences(self, arr: np.ndarray):
        X, y = [], []
        L = len(arr)
        for i in range(L - self.lookback):
            X.append(arr[i : i + self.lookback])
            y.append(arr[i + self.lookback])
        if len(X) == 0 or len(y) == 0:
            raise ValueError(
                f"No se pudieron formar secuencias: longitud={L}, lookback={self.lookback}."
            )
        X = np.array(X).reshape((-1, self.lookback, 1))
        y = np.array(y).reshape((-1, 1))
        return X, y

    # ----------------- API -----------------
    def fit(self, df_y: pd.DataFrame, epochs: int = 20, batch_size: int = 32, verbose: int = 0):
        df = self._check_df(df_y)

        # Si la serie es más corta que lookback+1, ampliamos repitiendo el último valor
        if len(df) < self.lookback + 1:
            deficit = self.lookback + 1 - len(df)
            tail_val = float(df["y"].iloc[-1])
            tail_dates = pd.date_range(
                start=df["ds"].iloc[-1] + pd.Timedelta(days=1),
                periods=deficit,
                freq="D",
            )
            df_pad = pd.DataFrame({"ds": tail_dates, "y": [tail_val] * deficit})
            df = pd.concat([df, df_pad], ignore_index=True)

        y = df["y"].to_numpy().reshape(-1, 1)
        y_sc = self.scaler.fit_transform(y).flatten()

        X, yseq = self._make_sequences(y_sc)

        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(self.lookback, 1)),
                tf.keras.layers.LSTM(64, return_sequences=False),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(1),
            ]
        )
        self.model.compile(optimizer="adam", loss="mse")
        self.model.fit(X, yseq, epochs=int(epochs), batch_size=int(batch_size), verbose=verbose)
        self._fitted = True

    def forecast(self, df_y: pd.DataFrame, horizon: int):
        if not self._fitted or self.model is None:
            raise RuntimeError("El modelo LSTM no está entrenado. Llama a fit() primero.")

        df = self._check_df(df_y)

        # Asegurar longitud mínima para extraer la ventana
        y = df["y"].to_numpy().reshape(-1, 1)
        y_sc_full = self.scaler.transform(y).flatten().tolist()

        if len(y_sc_full) < self.lookback:
            # completar al principio con el primer valor
            first_val = y_sc_full[0]
            pad = [first_val] * (self.lookback - len(y_sc_full))
            y_sc_full = pad + y_sc_full

        preds_sc = []
        hist = y_sc_full.copy()

        horizon = int(horizon)
        for _ in range(horizon):
            window = np.array(hist[-self.lookback :]).reshape(1, self.lookback, 1)
            yhat_sc = float(self.model.predict(window, verbose=0).flatten()[0])
            preds_sc.append(yhat_sc)
            hist.append(yhat_sc)

        preds = self.scaler.inverse_transform(np.array(preds_sc).reshape(-1, 1)).flatten()

        # Fechas futuras: desde el día siguiente al último real, con longitud == horizon
        last = pd.to_datetime(df["ds"].max())
        future_idx = pd.date_range(start=last + pd.Timedelta(days=1), periods=horizon, freq="D")

        # Construcción segura del DataFrame (longitudes iguales)
        if len(future_idx) != len(preds):
            raise ValueError(
                f"Longitudes distintas al crear predicciones: fechas={len(future_idx)} vs preds={len(preds)}"
            )

        return pd.DataFrame({"ds": future_idx, "yhat": preds})
