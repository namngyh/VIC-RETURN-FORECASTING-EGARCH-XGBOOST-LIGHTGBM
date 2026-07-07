from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pandas as pd


def train_xgboost(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
    target_name: str,
    params: dict[str, Any],
    random_state: int,
):
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError("Install xgboost first: pip install xgboost") from exc

    model = XGBRegressor(random_state=random_state, n_jobs=-1, **params)
    model.fit(
        train_df[feature_cols],
        train_df[target_name],
        eval_set=[(valid_df[feature_cols], valid_df[target_name])],
        verbose=False,
    )
    return model


def train_lightgbm(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
    target_name: str,
    params: dict[str, Any],
    random_state: int,
):
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("Install lightgbm first: pip install lightgbm") from exc

    train_data = lgb.Dataset(train_df[feature_cols], label=train_df[target_name], feature_name=feature_cols)
    valid_data = lgb.Dataset(
        valid_df[feature_cols],
        label=valid_df[target_name],
        feature_name=feature_cols,
        reference=train_data,
    )
    train_params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "seed": random_state,
        **params,
    }
    num_boost_round = int(train_params.pop("n_estimators", 100))
    model = lgb.train(
        train_params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[valid_data],
        valid_names=["valid"],
    )
    return model


def predict_frame(
    model,
    df: pd.DataFrame,
    feature_cols: list[str],
    target_name: str,
    model_name: str,
) -> pd.DataFrame:
    pred = model.predict(df[feature_cols])
    output = pd.DataFrame(
        {
            "date": df["date"].to_numpy(),
            "model": model_name,
            "predicted_return": pred,
            "predicted_direction": (pred > 0).astype(int),
        }
    )
    if target_name in df.columns:
        actual = df[target_name].to_numpy()
        output["actual_return"] = actual
        output["actual_direction"] = pd.Series(np.where(pd.notna(actual), actual > 0, pd.NA)).astype("Int64")
    return output


def save_model_bundle(model, path: str | Path, feature_cols: list[str], target_name: str) -> None:
    bundle = {"model": model, "feature_cols": feature_cols, "target_name": target_name}
    with Path(path).open("wb") as f:
        pickle.dump(bundle, f)


def load_model_bundle(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def feature_importance_frame(model, feature_cols: list[str]) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
    elif hasattr(model, "feature_importance"):
        values = np.asarray(model.feature_importance(importance_type="gain"))
    else:
        values = np.zeros(len(feature_cols))
    return (
        pd.DataFrame({"feature": feature_cols, "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
