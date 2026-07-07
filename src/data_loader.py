from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"date", "open", "high", "low", "close", "volume"}


def load_ohlcv_csv(
    path: str | Path,
    date_col: str = "date",
    price_col: str = "adj_close",
    fallback_price_col: str = "close",
) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Cannot find data file: {csv_path}. Put a VIC OHLCV CSV there or update config/config.yaml."
        )

    df = pd.read_csv(csv_path)
    df = normalize_ohlcv_columns(df, date_col=date_col)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

    resolved_price_col = price_col if price_col in df.columns else fallback_price_col
    if resolved_price_col not in df.columns:
        raise ValueError(
            f"Neither price_col={price_col!r} nor fallback_price_col={fallback_price_col!r} exists in {csv_path}."
        )

    numeric_cols = ["open", "high", "low", "close", "volume", resolved_price_col]
    for col in dict.fromkeys(numeric_cols):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    df["price"] = df[resolved_price_col]
    return df.dropna(subset=["date", "open", "high", "low", "close", "volume", "price"])


def normalize_ohlcv_columns(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [
        str(col).strip().lower().replace(" ", "_").replace("-", "_") for col in normalized.columns
    ]
    normalized = normalized.rename(columns={date_col.lower(): "date"})
    alias_map = {
        "datetime": "date",
        "time": "date",
        "adj_close": "adj_close",
        "adjusted_close": "adj_close",
        "adjclose": "adj_close",
    }
    normalized = normalized.rename(columns={k: v for k, v in alias_map.items() if k in normalized.columns})
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    return normalized


def download_yahoo_ohlcv(ticker: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("Install yfinance first: pip install yfinance") from exc

    data = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if data.empty:
        raise ValueError(f"No Yahoo Finance data returned for ticker {ticker!r}.")

    data = data.reset_index()
    data.columns = [str(col[0] if isinstance(col, tuple) else col).lower().replace(" ", "_") for col in data.columns]
    data = data.rename(
        columns={
            "date": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "adj_close": "adj_close",
            "volume": "volume",
        }
    )
    return normalize_ohlcv_columns(data)
