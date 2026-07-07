from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = actual - predicted
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    denominator = np.sum((actual - np.mean(actual)) ** 2)
    r2 = float(1.0 - np.sum(error**2) / denominator) if denominator != 0 else 0.0
    actual_direction = actual > 0
    predicted_direction = predicted > 0
    directional_accuracy = float(np.mean(actual_direction == predicted_direction))
    f1 = _binary_f1(actual_direction, predicted_direction)
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "directional_accuracy": directional_accuracy,
        "f1_direction": f1,
    }


def evaluate_predictions(predictions: pd.DataFrame) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for model_name, group in predictions.groupby("model"):
        group = group.dropna(subset=["actual_return", "predicted_return"])
        metrics[model_name] = regression_metrics(
            group["actual_return"].to_numpy(),
            group["predicted_return"].to_numpy(),
        )
    return metrics


def backtest_long_flat(
    prediction_df: pd.DataFrame,
    signal_threshold: float = 0.0,
    transaction_cost_bps: float = 15.0,
) -> pd.DataFrame:
    out = prediction_df.sort_values("date").copy()
    out["position"] = (out["predicted_return"] > signal_threshold).astype(float)
    out["trade"] = out["position"].diff().abs().fillna(out["position"].abs())
    out["cost"] = out["trade"] * transaction_cost_bps / 10000.0
    out["strategy_return"] = out["position"] * out["actual_return"] - out["cost"]
    out["buy_hold_return"] = out["actual_return"]
    out["strategy_equity"] = np.exp(out["strategy_return"].cumsum())
    out["buy_hold_equity"] = np.exp(out["buy_hold_return"].cumsum())
    return out


def backtest_summary(backtest_df: pd.DataFrame) -> dict[str, float]:
    returns = backtest_df["strategy_return"]
    buy_hold = backtest_df["buy_hold_return"]
    sharpe = _annualized_sharpe(returns)
    max_dd = _max_drawdown(backtest_df["strategy_equity"])
    return {
        "strategy_total_return": float(backtest_df["strategy_equity"].iloc[-1] - 1.0),
        "buy_hold_total_return": float(backtest_df["buy_hold_equity"].iloc[-1] - 1.0),
        "strategy_sharpe": float(sharpe),
        "strategy_max_drawdown": float(max_dd),
        "mean_daily_return": float(returns.mean()),
        "buy_hold_mean_daily_return": float(buy_hold.mean()),
    }


def _annualized_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    std = returns.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return np.sqrt(periods_per_year) * returns.mean() / std


def _max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return drawdown.min()


def _binary_f1(actual: np.ndarray, predicted: np.ndarray) -> float:
    true_positive = np.sum(actual & predicted)
    false_positive = np.sum(~actual & predicted)
    false_negative = np.sum(actual & ~predicted)
    denominator = 2 * true_positive + false_positive + false_negative
    if denominator == 0:
        return 0.0
    return float(2 * true_positive / denominator)
