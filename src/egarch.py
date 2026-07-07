from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def add_egarch_features(
    df: pd.DataFrame,
    method: str = "walk_forward",
    p: int = 1,
    o: int = 1,
    q: int = 1,
    dist: str = "normal",
    min_obs: int = 500,
    max_train_window: int | None = 1000,
) -> pd.DataFrame:
    if method == "none":
        return df
    if method == "in_sample":
        return add_in_sample_egarch_features(df, p=p, o=o, q=q, dist=dist)
    if method == "walk_forward":
        return add_walk_forward_egarch_features(
            df, p=p, o=o, q=q, dist=dist, min_obs=min_obs, max_train_window=max_train_window
        )
    raise ValueError(f"Unknown EGARCH method: {method!r}")


def add_in_sample_egarch_features(
    df: pd.DataFrame,
    p: int = 1,
    o: int = 1,
    q: int = 1,
    dist: str = "normal",
) -> pd.DataFrame:
    arch_model = _load_arch_model()
    out = df.copy()
    returns_pct = out["return_1d"].dropna() * 100.0
    if len(returns_pct) < 50:
        raise ValueError("Need at least 50 non-null returns for in-sample EGARCH.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = arch_model(returns_pct, mean="Constant", vol="EGARCH", p=p, o=o, q=q, dist=dist)
        result = model.fit(disp="off", show_warning=False)

    out["egarch_cond_vol"] = np.nan
    out["egarch_std_resid"] = np.nan
    out.loc[returns_pct.index, "egarch_cond_vol"] = result.conditional_volatility / 100.0
    out.loc[returns_pct.index, "egarch_std_resid"] = result.std_resid
    out["egarch_abs_std_resid"] = out["egarch_std_resid"].abs()
    out["egarch_negative_shock"] = np.minimum(out["egarch_std_resid"], 0.0)
    return out


def add_walk_forward_egarch_features(
    df: pd.DataFrame,
    p: int = 1,
    o: int = 1,
    q: int = 1,
    dist: str = "normal",
    min_obs: int = 500,
    max_train_window: int | None = 1000,
) -> pd.DataFrame:
    arch_model = _load_arch_model()
    out = df.copy()
    returns_pct = out["return_1d"] * 100.0
    out["egarch_cond_vol"] = np.nan
    out["egarch_std_resid"] = np.nan
    out["egarch_forecast_vol_1d"] = np.nan

    valid_positions = np.flatnonzero(returns_pct.notna().to_numpy())
    if len(valid_positions) < min_obs:
        raise ValueError(
            f"Need at least egarch.min_obs={min_obs} non-null returns, got {len(valid_positions)}. "
            "Lower min_obs in config/config.yaml for short datasets."
        )

    for count, pos in enumerate(valid_positions):
        if count + 1 < min_obs:
            continue
        train_start_count = 0 if max_train_window is None else max(0, count + 1 - max_train_window)
        train_positions = valid_positions[train_start_count : count + 1]
        train_returns = returns_pct.iloc[train_positions].dropna()

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = arch_model(train_returns, mean="Constant", vol="EGARCH", p=p, o=o, q=q, dist=dist)
                result = model.fit(disp="off", show_warning=False)
                forecast = result.forecast(horizon=1, reindex=False)
        except Exception:
            continue

        out.iloc[pos, out.columns.get_loc("egarch_cond_vol")] = result.conditional_volatility.iloc[-1] / 100.0
        out.iloc[pos, out.columns.get_loc("egarch_std_resid")] = result.std_resid.iloc[-1]
        out.iloc[pos, out.columns.get_loc("egarch_forecast_vol_1d")] = (
            np.sqrt(forecast.variance.iloc[-1, 0]) / 100.0
        )

    out["egarch_abs_std_resid"] = out["egarch_std_resid"].abs()
    out["egarch_negative_shock"] = np.minimum(out["egarch_std_resid"], 0.0)
    return out


def _load_arch_model():
    try:
        from arch import arch_model
    except ImportError as exc:
        raise ImportError("Install arch first: pip install arch") from exc
    return arch_model
