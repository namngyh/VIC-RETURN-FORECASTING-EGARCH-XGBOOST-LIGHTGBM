#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


TCBS_BARS_URL = "https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term"
VN_TZ = timezone(timedelta(hours=7))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Vietnamese OHLCV data for VIC.")
    parser.add_argument("--symbol", default="VIC", help="Ticker symbol, default: VIC.")
    parser.add_argument("--start", default="2015-01-01", help="Start date YYYY-MM-DD.")
    parser.add_argument("--end", default=date.today().isoformat(), help="End date YYYY-MM-DD.")
    parser.add_argument("--output", default="data/raw/VIC.csv", help="Output CSV path.")
    parser.add_argument(
        "--provider",
        default="vnstock",
        choices=["vnstock", "tcbs-direct", "yahoo"],
        help="Data provider. vnstock is recommended for Vietnamese equities.",
    )
    parser.add_argument(
        "--source",
        default="auto",
        help="vnstock source. Use auto, VCI, or KBS.",
    )
    parser.add_argument(
        "--asset-type",
        default="stock",
        choices=["stock", "index"],
        help="TCBS direct asset type. Use stock for VIC, index for VNINDEX.",
    )
    args = parser.parse_args()

    if args.provider == "vnstock":
        rows = download_vnstock_bars(args.symbol, args.start, args.end, source=args.source)
    elif args.provider == "tcbs-direct":
        rows = download_tcbs_bars(
            symbol=args.symbol,
            start=args.start,
            end=args.end,
            asset_type=args.asset_type,
        )
    else:
        rows = download_yahoo_bars(args.symbol, args.start, args.end)

    rows = filter_date_range(rows, args.start, args.end)
    write_ohlcv_csv(rows, args.output)
    print(f"Saved {len(rows)} rows for {args.symbol} from {args.provider} to {args.output}")


def download_vnstock_bars(symbol: str, start: str, end: str, source: str = "auto") -> list[dict[str, Any]]:
    try:
        from vnstock.api.quote import Quote
    except ImportError as exc:
        raise ImportError("Install vnstock first: pip install vnstock") from exc

    sources = ["VCI", "KBS"] if source.lower() == "auto" else [source.upper()]
    errors: list[str] = []

    for candidate in sources:
        try:
            quote = Quote(symbol=symbol.upper(), source=candidate, show_log=False)
            df = quote.history(symbol=symbol.upper(), start=start, end=end, interval="1D")
            if df is None or df.empty:
                errors.append(f"{candidate}: empty response")
                continue

            df = df.copy()
            df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]
            rows = [_normalize_bar(row) for row in df.to_dict(orient="records")]
            rows = [row for row in rows if row is not None]
            if rows:
                return sorted(rows, key=lambda row: row["date"])
            errors.append(f"{candidate}: response did not contain OHLCV columns {list(df.columns)}")
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")

    raise ValueError("vnstock download failed. " + " | ".join(errors))


def download_tcbs_bars(symbol: str, start: str, end: str, asset_type: str = "stock") -> list[dict[str, Any]]:
    params = {
        "ticker": symbol.upper(),
        "type": asset_type,
        "resolution": "D",
        "from": _to_epoch_seconds(start),
        "to": _to_epoch_seconds(end, end_of_day=True),
    }
    url = f"{TCBS_BARS_URL}?{urlencode(params)}"
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json,text/plain,*/*",
        },
    )

    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    raw_rows = _extract_rows(payload)
    rows = [_normalize_bar(row) for row in raw_rows]
    rows = [row for row in rows if row is not None]
    rows = sorted(rows, key=lambda row: row["date"])

    if not rows:
        raise ValueError(
            f"No OHLCV rows returned for {symbol}. Try a different date range or provider."
        )
    return rows


def download_yahoo_bars(symbol: str, start: str, end: str) -> list[dict[str, Any]]:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("Install yfinance first: pip install yfinance") from exc

    ticker = symbol if "." in symbol else f"{symbol}.VN"
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"No Yahoo Finance data returned for {ticker}.")

    df = df.reset_index()
    if isinstance(df.columns, list):
        pass
    df.columns = [
        str(col[0] if isinstance(col, tuple) else col).strip().lower().replace(" ", "_")
        for col in df.columns
    ]
    rows = [_normalize_bar(row) for row in df.to_dict(orient="records")]
    rows = [row for row in rows if row is not None]
    if not rows:
        raise ValueError(f"Yahoo returned data for {ticker}, but OHLCV columns could not be parsed.")
    return sorted(rows, key=lambda row: row["date"])


def write_ohlcv_csv(rows: list[dict[str, Any]], output: str | Path) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "open", "high", "low", "close", "volume"])
        writer.writeheader()
        writer.writerows(rows)


def filter_date_range(rows: list[dict[str, Any]], start: str, end: str) -> list[dict[str, Any]]:
    return [row for row in rows if start <= row["date"] <= end]


def _extract_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected response type: {type(payload).__name__}")

    for key in ("data", "bars", "items", "rows"):
        value = payload.get(key)
        if isinstance(value, list):
            return value

    # Some APIs return column arrays: {"t": [...], "o": [...], ...}.
    array_keys = {"t", "o", "h", "l", "c", "v"}
    if array_keys.issubset(payload.keys()) and all(isinstance(payload[key], list) for key in array_keys):
        return [
            {"t": payload["t"][idx], "o": payload["o"][idx], "h": payload["h"][idx],
             "l": payload["l"][idx], "c": payload["c"][idx], "v": payload["v"][idx]}
            for idx in range(len(payload["t"]))
        ]

    raise ValueError(f"Cannot find OHLCV rows in response keys: {sorted(payload.keys())}")


def _normalize_bar(row: dict[str, Any]) -> dict[str, Any] | None:
    date_value = _pick(row, "date", "datetime", "tradingDate", "trading_date", "time", "t")
    open_value = _pick(row, "open", "o")
    high_value = _pick(row, "high", "h")
    low_value = _pick(row, "low", "l")
    close_value = _pick(row, "close", "c")
    volume_value = _pick(row, "volume", "vol", "v")

    if any(value is None for value in [date_value, open_value, high_value, low_value, close_value, volume_value]):
        return None

    return {
        "date": _normalize_date(date_value),
        "open": _to_number(open_value),
        "high": _to_number(high_value),
        "low": _to_number(low_value),
        "close": _to_number(close_value),
        "volume": int(float(str(volume_value).replace(",", ""))),
    }


def _pick(row: dict[str, Any], *keys: str) -> Any:
    lowered = {str(key).lower(): value for key, value in row.items()}
    for key in keys:
        if key in row:
            return row[key]
        if key.lower() in lowered:
            return lowered[key.lower()]
    return None


def _normalize_date(value: Any) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()

    if isinstance(value, (int, float)):
        timestamp = float(value)
        if timestamp > 10_000_000_000:
            timestamp /= 1000.0
        return datetime.fromtimestamp(timestamp, tz=VN_TZ).date().isoformat()

    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(text[:10], fmt).date().isoformat()
        except ValueError:
            pass
    return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()


def _to_number(value: Any) -> float:
    return float(str(value).replace(",", ""))


def _to_epoch_seconds(value: str, end_of_day: bool = False) -> int:
    day = datetime.strptime(value, "%Y-%m-%d").date()
    clock = time.max if end_of_day else time.min
    return int(datetime.combine(day, clock, tzinfo=VN_TZ).timestamp())


if __name__ == "__main__":
    main()
