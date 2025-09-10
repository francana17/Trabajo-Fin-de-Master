# core/data_loader.py
import re
import pandas as pd
import numpy as np

# ---------- normalización y detección ----------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    def norm(s):
        s = str(s).strip()
        tr = str.maketrans("áéíóúÁÉÍÓÚñÑ", "aeiouAEIOUnN")
        s = s.translate(tr)
        s = re.sub(r"[^0-9A-Za-z_]+", "_", s).strip("_").lower()
        return s
    out = df.copy()
    out.columns = [norm(c) for c in out.columns]
    return out

def parse_date_auto(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # localizar columna de fecha
    cand = [c for c in df.columns if c in ("fecha","date")]
    if not cand:
        for c in df.columns:
            if re.search(r"(fecha|date|dia|day|fecha_valor|fecha_venta)$", c):
                cand.append(c); break
    if not cand:
        raise ValueError("No se encuentra columna de fecha ('fecha' o 'date').")
    col = cand[0]

    # detectar patrón de formato con la primera no nula
    sample = next((str(v) for v in df[col] if pd.notna(v)), "")
    iso_like = bool(re.match(r"^\d{4}-\d{2}-\d{2}$", sample))
    euro_like = bool(re.match(r"^\d{2}/\d{2}/\d{4}$", sample)) or bool(re.match(r"^\d{2}-\d{2}-\d{4}$", sample))

    if iso_like:
        dt = pd.to_datetime(df[col], errors="coerce", dayfirst=False)
    elif euro_like:
        dt = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
    else:
        # intento robusto
        dt = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
        if dt.isna().mean() > 0.5:
            dt = pd.to_datetime(df[col], errors="coerce", dayfirst=False)

    if dt.isna().all():
        raise ValueError("No se pudieron parsear las fechas del CSV.")
    df["Date"] = dt
    return df

def detect_format(df: pd.DataFrame) -> str:
    cols = set(df.columns)
    if {"date","tipo","precio_eur_kg"}.issubset(cols) or {"Date","tipo","precio_eur_kg"}.issubset(cols):
        return "tidy"
    wide_candidates = {"virgen_extra_picual","virgen_picual","lampante_picual"}
    if wide_candidates.intersection(cols):
        return "wide"
    # si hay al menos 1 columna numérica además de la fecha, lo consideramos ancho
    num_cols = [c for c in df.columns if c not in ("date","Date") and pd.api.types.is_numeric_dtype(df[c])]
    return "wide" if len(num_cols) >= 1 else "unknown"

def candidates_types(df: pd.DataFrame, fmt: str):
    if fmt == "tidy":
        return sorted(df["tipo"].dropna().astype(str).str.lower().unique().tolist())
    # wide
    cols = [c for c in df.columns if c not in ("date","Date")]
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    return numeric

# ---------- preparación de serie ----------
def prepare_series(df_raw: pd.DataFrame, tipo: str, is_tidy: bool, interp: bool, ffill: bool) -> pd.DataFrame:
    """Devuelve df con columnas ['ds','y'] listo para Prophet/LSTM."""
    if is_tidy:
        df_s = (
            df_raw.loc[df_raw["tipo"].astype(str).str.lower() == tipo, ["Date","precio_eur_kg"]]
                  .dropna(subset=["Date"])
                  .groupby("Date", as_index=False)["precio_eur_kg"].mean()
                  .sort_values("Date")
                  .rename(columns={"Date":"ds","precio_eur_kg":"y"})
        )
    else:
        df_s = (
            df_raw[["Date", tipo]]
                .rename(columns={"Date":"ds", tipo:"y"})
                .dropna(subset=["ds"])
                .sort_values("ds")
        )

    # quitar duplicados por fecha
    df_s = df_s[~df_s["ds"].duplicated(keep="last")]

    if interp or ffill:
        full_idx = pd.date_range(df_s["ds"].min(), df_s["ds"].max(), freq="D")
        df_s = df_s.set_index("ds").reindex(full_idx)
        if ffill:
            df_s["y"] = df_s["y"].ffill()
        if interp:
            df_s["y"] = df_s["y"].interpolate(method="time")
        df_s = df_s.rename_axis("ds").reset_index()

    df_s = df_s.dropna(subset=["y"])
    return df_s

# ---------- utilidades de evaluación ----------
def join_future_with_real(future_df: pd.DataFrame, hist_df: pd.DataFrame, is_tidy: bool, tipo: str):
    last_hist = hist_df["ds"].max()
    out = (
        future_df.loc[future_df["ds"] > last_hist, ["ds","yhat","yhat_lower","yhat_upper"]]
                 .rename(columns={"ds":"fecha","yhat":"prediccion","yhat_lower":"lo","yhat_upper":"hi"})
    )
    # reales
    if is_tidy:
        real = hist_df.rename(columns={"ds":"fecha","y":"real"})[["fecha","real"]]
    else:
        real = hist_df.rename(columns={"ds":"fecha","y":"real"})[["fecha","real"]]
    merged = out.merge(real, on="fecha", how="left")
    return out, merged

def backtest_split(df_y: pd.DataFrame, horizon: int):
    """Separa últimos h puntos como test y devuelve (train_df, test_df)."""
    if len(df_y) <= horizon + 5:
        raise ValueError("Serie demasiado corta para backtest con ese horizonte.")
    train = df_y.iloc[:-horizon].copy()
    test = df_y.iloc[-horizon:].copy()
    return train, test
