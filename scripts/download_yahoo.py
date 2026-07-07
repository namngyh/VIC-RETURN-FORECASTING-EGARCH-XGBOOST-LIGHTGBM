#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from src.data_loader import download_yahoo_ohlcv


def main() -> None:
    parser = argparse.ArgumentParser(description="Download OHLCV data from Yahoo Finance.")
    parser.add_argument("--ticker", default="VIC.VN", help="Yahoo Finance ticker.")
    parser.add_argument("--start", default=None, help="Start date, e.g. 2015-01-01.")
    parser.add_argument("--end", default=None, help="End date, e.g. 2026-01-01.")
    parser.add_argument("--output", default="data/raw/VIC.csv", help="Output CSV path.")
    args = parser.parse_args()

    df = download_yahoo_ohlcv(args.ticker, start=args.start, end=args.end)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
