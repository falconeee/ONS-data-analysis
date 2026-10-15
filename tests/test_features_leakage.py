"""No-leakage guarantees of the day-ahead backtest.

1. Forecasts issued at a cutoff must not change when the target AFTER the cutoff is altered
   (checked through the same Nixtla engines used by the backtest).
2. Rolling-conformal intervals for day D may only use residuals of target days <= D-2.
3. Future-known covariates are a function of the timestamp only (never of y).
"""

from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ons_energy.forecast.intervals import add_rolling_conformal


def _synthetic_panel(n_days: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ds = pd.date_range("2024-01-01", periods=24 * n_days, freq="h")
    t = np.arange(len(ds))
    y = 1000 + 100 * np.sin(2 * np.pi * t / 24) + 30 * np.sin(2 * np.pi * t / 168) + rng.normal(0, 5, len(ds))
    return pd.DataFrame({"unique_id": "SE", "ds": ds, "y": y, "hour_sin": np.sin(2 * np.pi * ds.hour / 24)})


def _perturb_future(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    out = df.copy()
    out.loc[out.ds > cutoff, "y"] *= 10  # wildly different future
    return out


def _cv_at(engine, df: pd.DataFrame, cutoff: pd.Timestamp, **kw) -> pd.DataFrame:
    # truncate so the last window starts right after `cutoff` (h=36, cutoff 11:00 -> panel ends 23:00)
    sub = df[df.ds <= cutoff + pd.Timedelta(hours=36)]
    res = engine.cross_validation(df=sub, h=36, step_size=24, n_windows=1, **kw)
    res = res.reset_index() if "unique_id" not in res.columns else res
    return res[res.cutoff == cutoff]


def test_seasonal_naive_ignores_future_target():
    from statsforecast import StatsForecast
    from statsforecast.models import SeasonalNaive

    df = _synthetic_panel()
    cutoff = pd.Timestamp("2024-03-10 11:00")
    sf = StatsForecast(models=[SeasonalNaive(season_length=168)], freq="h")
    a = _cv_at(sf, df[["unique_id", "ds", "y"]], cutoff)
    b = _cv_at(sf, _perturb_future(df, cutoff)[["unique_id", "ds", "y"]], cutoff)
    np.testing.assert_allclose(a["SeasonalNaive"], b["SeasonalNaive"])
    assert not np.allclose(a["y"], b["y"])  # the perturbation did hit the evaluated targets


def test_lightgbm_recursive_ignores_future_target():
    from ons_energy.forecast.config import ModelSpec
    from ons_energy.forecast.models import lgbm

    df = _synthetic_panel()
    cutoff = pd.Timestamp("2024-03-10 11:00")
    spec = ModelSpec(name="LightGBM", kind="lgbm", params={"n_estimators": 30, "n_jobs": 1})
    a = _cv_at(lgbm([spec], 0), df, cutoff, static_features=[])
    b = _cv_at(lgbm([spec], 0), _perturb_future(df, cutoff), cutoff, static_features=[])
    np.testing.assert_allclose(a["LightGBM"], b["LightGBM"])


def test_cutoffs_fall_on_origin_hour():
    from statsforecast import StatsForecast
    from statsforecast.models import SeasonalNaive

    df = _synthetic_panel()  # ends at 23:00
    res = StatsForecast(models=[SeasonalNaive(season_length=168)], freq="h").cross_validation(
        df=df[["unique_id", "ds", "y"]], h=36, step_size=24, n_windows=10
    )
    assert set(res["cutoff"].dt.hour) == {11}
    steps = ((res["ds"] - res["cutoff"]) / pd.Timedelta(hours=1)).astype(int)
    assert steps.min() == 1 and steps.max() == 36


def test_rolling_conformal_uses_only_past_residuals():
    rng = np.random.default_rng(1)
    days = pd.date_range("2024-01-01", periods=120, freq="D")
    rows = []
    for d in days:
        cutoff = d - pd.Timedelta(hours=13)  # D-1 11:00
        for step in range(13, 37):
            ds = cutoff + pd.Timedelta(hours=step)
            rows.append({"unique_id": "SE", "ds": ds, "cutoff": cutoff, "step": step, "y": rng.normal(), "M": 0.0})
    cv = pd.DataFrame(rows)
    target = pd.Timestamp("2024-04-15")
    base = add_rolling_conformal(cv, "M", [0.1, 0.9], window_days=60)
    bumped = cv.copy()
    # residuals of D-1 and later are NOT known when issuing D's forecast -> must not matter
    bumped.loc[bumped.ds >= target - pd.Timedelta(days=1), "y"] += 100
    alt = add_rolling_conformal(bumped, "M", [0.1, 0.9], window_days=60)
    sel = base.ds.dt.normalize() == target
    np.testing.assert_allclose(base.loc[sel, "M-q0.9"], alt.loc[sel, "M-q0.9"])
    # ... but residuals of D-2 do
    bumped2 = cv.copy()
    bumped2.loc[bumped2.ds.dt.normalize() == target - pd.Timedelta(days=2), "y"] += 100
    alt2 = add_rolling_conformal(bumped2, "M", [0.1, 0.9], window_days=60)
    assert not np.allclose(base.loc[sel, "M-q0.9"], alt2.loc[sel, "M-q0.9"])


def test_calendar_features_depend_only_on_time():
    from ons_energy.features.forecast import CALENDAR_FEATURES, calendar_features

    events = pl.DataFrame(
        {"event_id": ["m"], "kind": ["worldcup_match"], "start": [datetime(2022, 12, 9, 12)],
         "end": [datetime(2022, 12, 9, 14)], "subsystems": [["SE", "S", "NE", "N"]], "description": [""]}
    )
    ts = pl.datetime_range(datetime(2022, 12, 1), datetime(2022, 12, 31, 23), "1h", eager=True)
    f = calendar_features(ts, "SE", events)
    assert f.columns == ["ds", *CALENDAR_FEATURES]
    match = f.filter(pl.col("worldcup_match") == 1)["ds"]
    assert match.min() == datetime(2022, 12, 9, 11) and match.max() == datetime(2022, 12, 9, 15)
    assert f.filter(pl.col("ds") == datetime(2022, 12, 25, 10))["is_national_holiday"].item() == 1
    assert f["regime_2023_04"].sum() == 0 and f["regime_2021_03"].min() == 1


@pytest.mark.parametrize("hour", [11])
def test_origin_constant_matches_config(hour):
    from ons_energy.forecast.config import BacktestConfig, ModelSpec

    cfg = BacktestConfig(name="x", models=[ModelSpec(name="a", kind="seasonal_naive")])
    assert (23 - cfg.horizon) % 24 == hour == cfg.origin_hour - 1
