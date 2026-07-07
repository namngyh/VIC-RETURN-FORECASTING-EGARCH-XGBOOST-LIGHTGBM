#!/usr/bin/env python3

from __future__ import annotations

import argparse

from src.config import load_config
from src.data_loader import load_ohlcv_csv
from src.egarch import add_egarch_features
from src.features import add_return_target, add_technical_features, build_inference_matrix
from src.modeling import load_model_bundle, predict_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch prediction with a saved model bundle.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config.")
    parser.add_argument("--model", required=True, help="Path to .joblib model bundle.")
    parser.add_argument("--input", required=True, help="Input OHLCV CSV.")
    parser.add_argument("--output", required=True, help="Output predictions CSV.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.raw["data"]
    df = load_ohlcv_csv(
        args.input,
        date_col=data_cfg.get("date_col", "date"),
        price_col=data_cfg.get("price_col", "adj_close"),
        fallback_price_col=data_cfg.get("fallback_price_col", "close"),
    )
    df = add_return_target(df, horizon=cfg.horizon, target_name=cfg.target_name)
    features_cfg = cfg.raw.get("features", {})
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

    bundle = load_model_bundle(args.model)
    modeling_df = build_inference_matrix(
        df,
        feature_cols=bundle["feature_cols"],
        target_name=bundle["target_name"],
    )
    predictions = predict_frame(
        bundle["model"],
        modeling_df,
        bundle["feature_cols"],
        bundle["target_name"],
        model_name=args.model,
    )
    predictions.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
