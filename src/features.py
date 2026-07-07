from __future__ import annotations

import numpy as np
import pandas as pd


def add_return_target(df: pd.DataFrame, horizon: int, target_name: str) -> pd.DataFrame:
    out = df.copy()
    out["log_price"] = np.log(out["price"])
    out["return_1d"] = out["log_price"].diff()
    out[target_name] = out["log_price"].shift(-horizon) - out["log_price"]
    out["direction_next_1d"] = (out[target_name] > 0).astype(int)
    return out


def add_technical_features(
    df: pd.DataFrame,
    return_lags: list[int],
    rolling_windows: list[int],
    rsi_windows: list[int],
    add_calendar: bool = True,
) -> pd.DataFrame:
    out = df.copy()

    for lag in return_lags:
        out[f"return_lag_{lag}"] = out["return_1d"].shift(lag)

    for window in rolling_windows:
        out[f"return_mean_{window}"] = out["return_1d"].rolling(window).mean()
        out[f"return_vol_{window}"] = out["return_1d"].rolling(window).std()
        out[f"price_ma_{window}"] = out["price"].rolling(window).mean()
        out[f"price_to_ma_{window}"] = out["price"] / out[f"price_ma_{window}"] - 1.0
        out[f"volume_mean_{window}"] = out["volume"].rolling(window).mean()
        out[f"volume_to_mean_{window}"] = out["volume"] / out[f"volume_mean_{window}"] - 1.0

    out["hl_range"] = np.log(out["high"] / out["low"])
    out["oc_return"] = np.log(out["close"] / out["open"])
    out["close_to_high"] = out["close"] / out["high"] - 1.0
    out["close_to_low"] = out["close"] / out["low"] - 1.0
    out["macd"] = _ema(out["price"], 12) - _ema(out["price"], 26)
    out["macd_signal"] = _ema(out["macd"], 9)
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    out["atr_14"] = _atr(out, 14)

    for window in rsi_windows:
        out[f"rsi_{window}"] = _rsi(out["price"], window)

    if add_calendar:
        out["day_of_week"] = out["date"].dt.dayofweek
        out["month"] = out["date"].dt.month
        out["is_month_end"] = out["date"].dt.is_month_end.astype(int)

    return out


def add_market_features(stock: pd.DataFrame, market: pd.DataFrame | None) -> pd.DataFrame:
    if market is None:
        return stock

    market_features = market[["date", "price"]].copy()
    market_features["market_return_1d"] = np.log(market_features["price"]).diff()
    market_features["market_vol_20"] = market_features["market_return_1d"].rolling(20).std()
    market_features["market_momentum_20"] = np.log(market_features["price"]).diff(20)
    market_features = market_features.drop(columns=["price"])
    return stock.merge(market_features, on="date", how="left")


def build_feature_matrix(
    df: pd.DataFrame,
    target_name: str,
    drop_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    drop_columns = drop_columns or []
    excluded = {
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "price",
        "log_price",
        target_name,
        "direction_next_1d",
        *drop_columns,
    }
    feature_cols = [
        col
        for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]
    modeling_df = df[["date", target_name, "direction_next_1d", *feature_cols]].replace(
        [np.inf, -np.inf], np.nan
    )
    modeling_df = modeling_df.dropna(subset=[target_name, *feature_cols]).reset_index(drop=True)
    return modeling_df, feature_cols


def build_inference_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_name: str = "target_return_next_1d",
) -> pd.DataFrame:
    required = ["date", *feature_cols]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Inference data is missing model features: {missing}")

    optional_cols = [col for col in [target_name, "direction_next_1d"] if col in df.columns]
    inference_df = df[[*required, *optional_cols]].replace([np.inf, -np.inf], np.nan)
    return inference_df.dropna(subset=feature_cols).reset_index(drop=True)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(price: pd.Series, window: int) -> pd.Series:
    delta = price.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(df: pd.DataFrame, window: int) -> pd.Series:
    previous_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window).mean()
