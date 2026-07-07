#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import ensure_output_dirs, load_config
from src.data_loader import load_ohlcv_csv
from src.egarch import add_egarch_features
from src.evaluation import backtest_long_flat, backtest_summary, evaluate_predictions
from src.features import add_market_features, add_return_target, add_technical_features, build_feature_matrix
from src.modeling import (
    feature_importance_frame,
    predict_frame,
    save_model_bundle,
    train_lightgbm,
    train_xgboost,
)
from src.split import time_series_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EGARCH + XGBoost/LightGBM models for VIC returns.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dirs = ensure_output_dirs(cfg)
    data_cfg = cfg.raw["data"]

    stock = load_ohlcv_csv(
        data_cfg["stock_csv"],
        date_col=data_cfg.get("date_col", "date"),
        price_col=data_cfg.get("price_col", "adj_close"),
        fallback_price_col=data_cfg.get("fallback_price_col", "close"),
    )

    market = None
    if data_cfg.get("market_csv"):
        market = load_ohlcv_csv(
            data_cfg["market_csv"],
            date_col=data_cfg.get("date_col", "date"),
            price_col=data_cfg.get("price_col", "adj_close"),
            fallback_price_col=data_cfg.get("fallback_price_col", "close"),
        )

    features_cfg = cfg.raw.get("features", {})
    df = add_return_target(stock, horizon=cfg.horizon, target_name=cfg.target_name)
    df = add_market_features(df, market if features_cfg.get("add_market_features", True) else None)
    df = add_technical_features(
        df,
        return_lags=features_cfg.get("return_lags", [1, 2, 3, 5, 10, 20]),
        rolling_windows=features_cfg.get("rolling_windows", [5, 10, 20, 60]),
        rsi_windows=features_cfg.get("rsi_windows", [14]),
        add_calendar=features_cfg.get("add_calendar", True),
    )

    egarch_cfg = cfg.raw.get("egarch", {})
    if egarch_cfg.get("enabled", True):
        df = add_egarch_features(
            df,
            method=egarch_cfg.get("method", "walk_forward"),
            p=int(egarch_cfg.get("p", 1)),
            o=int(egarch_cfg.get("o", 1)),
            q=int(egarch_cfg.get("q", 1)),
            dist=egarch_cfg.get("dist", "normal"),
            min_obs=int(egarch_cfg.get("min_obs", 500)),
            max_train_window=egarch_cfg.get("max_train_window", 1000),
        )

    modeling_df, feature_cols = build_feature_matrix(df, target_name=cfg.target_name)
    feature_path = dirs["processed"] / "features.csv"
    modeling_df.to_csv(feature_path, index=False)

    split_cfg = cfg.raw["split"]
    train_df, valid_df, test_df = time_series_split(
        modeling_df,
        train_size=float(split_cfg["train_size"]),
        valid_size=float(split_cfg["valid_size"]),
        test_size=float(split_cfg["test_size"]),
    )

    predictions: list[pd.DataFrame] = []
    metrics: dict[str, object] = {
        "n_rows": len(modeling_df),
        "n_features": len(feature_cols),
        "feature_path": str(feature_path),
        "splits": {
            "train": [str(train_df["date"].min().date()), str(train_df["date"].max().date())],
            "valid": [str(valid_df["date"].min().date()), str(valid_df["date"].max().date())],
            "test": [str(test_df["date"].min().date()), str(test_df["date"].max().date())],
        },
    }

    model_cfg = cfg.raw.get("models", {})
    if model_cfg.get("xgboost", {}).get("enabled", True):
        model = train_xgboost(
            train_df,
            valid_df,
            feature_cols,
            cfg.target_name,
            params=model_cfg["xgboost"].get("params", {}),
            random_state=cfg.random_state,
        )
        save_model_bundle(model, dirs["models"] / "xgboost.pkl", feature_cols, cfg.target_name)
        feature_importance_frame(model, feature_cols).to_csv(
            dirs["reports"] / "feature_importance_xgboost.csv", index=False
        )
        predictions.append(predict_frame(model, test_df, feature_cols, cfg.target_name, "xgboost"))

    if model_cfg.get("lightgbm", {}).get("enabled", True):
        model = train_lightgbm(
            train_df,
            valid_df,
            feature_cols,
            cfg.target_name,
            params=model_cfg["lightgbm"].get("params", {}),
            random_state=cfg.random_state,
        )
        save_model_bundle(model, dirs["models"] / "lightgbm.pkl", feature_cols, cfg.target_name)
        feature_importance_frame(model, feature_cols).to_csv(
            dirs["reports"] / "feature_importance_lightgbm.csv", index=False
        )
        predictions.append(predict_frame(model, test_df, feature_cols, cfg.target_name, "lightgbm"))

    if not predictions:
        raise ValueError("No model is enabled in config/config.yaml.")

    predictions_df = pd.concat(predictions, ignore_index=True)
    predictions_path = dirs["reports"] / "predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    metrics["test_metrics"] = evaluate_predictions(predictions_df)

    backtest_cfg = cfg.raw.get("backtest", {})
    if backtest_cfg.get("enabled", True):
        metrics["backtest"] = {}
        for model_name, group in predictions_df.groupby("model"):
            bt = backtest_long_flat(
                group,
                signal_threshold=float(backtest_cfg.get("signal_threshold", 0.0)),
                transaction_cost_bps=float(backtest_cfg.get("transaction_cost_bps", 15.0)),
            )
            bt.to_csv(dirs["reports"] / f"backtest_{model_name}.csv", index=False)
            metrics["backtest"][model_name] = backtest_summary(bt)

    metrics_path = dirs["reports"] / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Saved features to {feature_path}")
    print(f"Saved predictions to {predictions_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
