# VIC Return Forecasting with EGARCH + XGBoost/LightGBM

Project scaffold for a research pipeline that forecasts next-day VIC stock returns using a hybrid econometrics and machine-learning design:

```text
VIC OHLCV data
-> return and technical features
-> EGARCH volatility features
-> XGBoost / LightGBM
-> forecast r(t+1), evaluate, and backtest a simple signal
```

## Notebook Version

The full research workflow is also available as a single Jupyter notebook:

```text
VIC_EGARCH_GBM.ipynb
```

Open it after activating your conda environment:

```bash
conda activate eda
jupyter notebook VIC_EGARCH_GBM.ipynb
```

## Project Structure

```text
config/config.yaml          Main experiment configuration
data/raw/                   Put raw CSV files here
data/processed/             Generated feature datasets
models/                     Trained model artifacts
reports/                    Metrics, predictions, importances, backtest output
src/config.py               Config loading and dataclasses
src/data_loader.py          CSV loading and optional Yahoo download
src/features.py             Return, market, and technical indicators
src/egarch.py               EGARCH feature generation
src/split.py                Time-series train/validation/test split
src/modeling.py             XGBoost/LightGBM training and prediction
src/evaluation.py           Regression, direction, and backtest metrics
src/train.py                End-to-end training pipeline
src/predict.py              Batch prediction with saved models
scripts/download_yahoo.py   Optional data download helper
```

## Data Format

Create a CSV file at `data/raw/VIC.csv` with at least:

```csv
date,open,high,low,close,volume
2020-01-02,100000,101000,99000,100500,1234567
```

Optional column:

```text
adj_close
```

If `adj_close` exists, the pipeline uses it for returns. Otherwise it uses `close`.

You can also add a market index file such as `data/raw/VNINDEX.csv` with the same format and configure it in `config/config.yaml`.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Download

Download VIC directly into the expected raw-data path:

```bash
python scripts/download_vic.py --symbol VIC --start 2015-01-01 --output data/raw/VIC.csv
```

Optional market index data:

```bash
python scripts/download_vic.py --symbol VNINDEX --asset-type index --start 2015-01-01 --output data/raw/VNINDEX.csv
```

Yahoo tickers for Vietnam can vary by provider. You can also try:

```bash
python scripts/download_yahoo.py --ticker VIC.VN --output data/raw/VIC.csv
```

## Train

```bash
python -m src.train --config config/config.yaml
```

Outputs:

```text
data/processed/features.csv
models/xgboost.pkl
models/lightgbm.pkl
reports/metrics.json
reports/predictions.csv
reports/feature_importance_*.csv
reports/backtest_*.csv
```

## Predict

```bash
python -m src.predict --config config/config.yaml --model models/lightgbm.pkl --input data/raw/VIC.csv --output reports/latest_predictions.csv
```

## Research Notes

The default target is:

```text
target_return_next_1d = log(P(t+1) / P(t))
```

The default EGARCH mode is `walk_forward`, which is slower but avoids fitting volatility parameters using future observations. For quick exploratory runs, you can set `egarch.method: in_sample` in `config/config.yaml`, but this is less appropriate for final research results.
