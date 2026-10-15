"""Rolling-origin day-ahead backtest shared by every model.

All models see the same panel and the same daily cutoffs (D-1 ``origin_hour - 1``:00) through the
Nixtla ``cross_validation`` engines, with refits every ``refit_every_days`` windows. Results are
stored long-format (unique_id, ds, cutoff, step, y, <model columns>) under
``data/interim/forecasts/<name>.parquet``.
"""

import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import polars as pl

from ons_energy.features.forecast import build_panel
from ons_energy.forecast import models as factory
from ons_energy.forecast.config import BacktestConfig
from ons_energy.forecast.intervals import add_rolling_conformal
from ons_energy.utils.io import INTERIM_DIR, PROCESSED_DIR
from ons_energy.utils.logging import get_logger
from ons_energy.utils.seed import set_seed

log = get_logger(__name__)

FORECAST_DIR = INTERIM_DIR / "forecasts"
NEURAL_KINDS = {"nhits", "nbeats_i", "nbeatsx", "tft"}


def last_full_day() -> date:
    ts = pl.read_parquet(PROCESSED_DIR / "load_hourly.parquet", columns=["ts_local"])["ts_local"]
    last = ts.max()
    return last.date() if last.hour == 23 else last.date() - timedelta(days=1)


def _ons_programmed(cv: pd.DataFrame, window_days: int = 28) -> tuple[np.ndarray, np.ndarray]:
    """ONS programmed load and a level-adjusted version.

    The API "carga global" runs ~1% above curva-carga (definition). ``ONS_adj`` rescales the
    programmed load by the curva-carga / verified-API ratio over the ``window_days`` days ending at
    D-2 (known when the D-1 forecast is issued), so it is still a causal benchmark.
    """
    api = pl.read_parquet(PROCESSED_DIR / "ons_load_api_hourly.parquet").select(
        unique_id="subsystem", ds="ts_local", ONS="programmed_load", verified="verified_load"
    ).to_pandas()
    load = pl.read_parquet(PROCESSED_DIR / "load_hourly.parquet").select(
        unique_id="subsystem", ds="ts_local", load="load_mw"
    ).to_pandas()
    daily = api.merge(load, on=["unique_id", "ds"], how="inner").dropna(subset=["verified", "load"])
    daily = daily.assign(day=daily.ds.dt.normalize()).groupby(["unique_id", "day"])[["load", "verified"]].sum()
    roll = daily.groupby(level=0).rolling(window_days, min_periods=7).sum().droplevel(0)
    factor = (roll["load"] / roll["verified"]).groupby(level=0).shift(2)  # ratio known at D-1 (target D)
    factor = factor.rename("factor").reset_index()
    merged = cv[["unique_id", "ds"]].merge(api[["unique_id", "ds", "ONS"]], on=["unique_id", "ds"], how="left")
    merged["day"] = merged.ds.dt.normalize()
    merged = merged.merge(factor, on=["unique_id", "day"], how="left")
    return merged["ONS"].to_numpy(), (merged["ONS"] * merged["factor"]).to_numpy()


def run_backtest(cfg: BacktestConfig) -> pd.DataFrame:
    set_seed(cfg.seed)
    if (23 - cfg.horizon) % 24 != cfg.origin_hour - 1:
        raise ValueError("horizon must make the last cutoff fall on origin_hour - 1 (e.g. 36 for 12:00)")

    eval_start = cfg.eval_start
    eval_end = cfg.eval_end(last_full_day())
    first_target = eval_start - timedelta(days=cfg.calib_days)
    n_windows = (eval_end - first_target).days + 1
    panel = build_panel(cfg.subsystems, cfg.train_start, eval_end, cfg.weather, weather_switch=first_target)
    df = panel.df
    log.info(
        "%s: %d series, %d rows, %d windows (%s -> %s), weather=%s (fallback share %.1f%%)",
        cfg.name, df.unique_id.nunique(), len(df), n_windows, first_target, eval_end, cfg.weather,
        100 * panel.weather_gap_share,
    )
    refit = cfg.refit_every_days

    kinds = {s.kind for s in cfg.models}
    if "lgbm" in kinds and kinds & NEURAL_KINDS:
        raise ValueError("Put LightGBM and neural models in separate configs (libomp clash on macOS)")
    by_kind: dict[str, list] = {}
    for spec in cfg.models:
        by_kind.setdefault("neural" if spec.kind in NEURAL_KINDS else spec.kind, []).append(spec)

    parts: list[pd.DataFrame] = []
    timings: dict[str, float] = {}
    for kind, specs in by_kind.items():
        t0 = time.perf_counter()
        if kind == "seasonal_naive":
            sf = factory.seasonal_naive(specs)
            res = sf.cross_validation(df=df[["unique_id", "ds", "y"]], h=cfg.horizon, step_size=24, n_windows=n_windows)
        elif kind == "lgbm":
            mlf = factory.lgbm(specs, cfg.seed)
            res = mlf.cross_validation(
                df=df, h=cfg.horizon, n_windows=n_windows, step_size=24, refit=refit, static_features=[]
            )
        elif kind == "neural":
            nf = factory.neural(specs, cfg, panel.futr_exog)
            set_seed(cfg.seed)  # torch is imported now
            res = nf.cross_validation(df=df, n_windows=n_windows, step_size=24, refit=refit)
            res = _rename_quantiles(res, specs)
        elif kind == "ons_programmed":
            continue
        else:
            raise ValueError(kind)
        timings[kind] = time.perf_counter() - t0
        log.info("%s: %s done in %.1fs", cfg.name, kind, timings[kind])
        parts.append(res.reset_index(drop=True) if "unique_id" in res.columns else res.reset_index())

    cv = parts[0]
    for p in parts[1:]:
        cv = cv.merge(p.drop(columns=["y"]), on=["unique_id", "ds", "cutoff"], how="outer")
    cv["step"] = ((cv["ds"] - cv["cutoff"]) / pd.Timedelta(hours=1)).astype(int)
    if any(s.kind == "ons_programmed" for s in cfg.models):
        cv["ONS"], cv["ONS_adj"] = _ons_programmed(cv)

    lo, hi = cfg.eval_steps
    cv = cv[(cv.step >= lo) & (cv.step <= hi)].copy()
    point_models = [s.name for s in cfg.models if s.kind not in ("tft",)]
    if "ONS_adj" in cv:
        point_models.append("ONS_adj")
    for m in point_models:
        cv = add_rolling_conformal(cv, m, cfg.quantiles, cfg.conformal_window_days)
    cv["target_date"] = cv["ds"].dt.date
    cv = cv[cv["ds"] >= pd.Timestamp(eval_start)].reset_index(drop=True)

    FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    cv.to_parquet(FORECAST_DIR / f"{cfg.name}.parquet", index=False)
    cv.attrs["timings"] = timings
    cv.attrs["weather_gap_share"] = panel.weather_gap_share
    return cv


def _rename_quantiles(res: pd.DataFrame, specs) -> pd.DataFrame:
    """TFT MQLoss columns -> ``{name}`` (median) and ``{name}-q{level}``."""
    ren = {}
    for s in specs:
        if s.kind != "tft":
            continue
        for suffix, q in factory.NF_LEVEL_COLUMNS.items():
            col = f"{s.name}-{suffix}"
            if col in res.columns:
                ren[col] = s.name if suffix == "median" else f"{s.name}-q{q:g}"
    out = res.rename(columns=ren)
    for s in specs:
        if s.kind == "tft" and f"{s.name}" in out.columns:
            out[f"{s.name}-q0.5"] = out[s.name]
    return out


KEY = ["unique_id", "ds", "cutoff", "y", "step", "target_date"]


def run_or_load(cfg: BacktestConfig, retrain: bool = False) -> pd.DataFrame:
    """Stored backtest ``cfg.name`` if it exists (and ``retrain`` is False), else :func:`run_backtest`."""
    path = FORECAST_DIR / f"{cfg.name}.parquet"
    if path.exists() and not retrain:
        log.info("%s: loading stored backtest %s (retrain=True to refit)", cfg.name, path)
        return pd.read_parquet(path)
    return run_backtest(cfg)


def load_forecasts(name: str) -> pd.DataFrame:
    """Backtest stored by :func:`run_backtest` (lets one notebook compare models trained in another)."""
    return pd.read_parquet(FORECAST_DIR / f"{name}.parquet")


def merge_forecasts(names: list[str]) -> pd.DataFrame:
    """Stored backtests side by side on their common windows (first name wins duplicated columns)."""
    cv = load_forecasts(names[0])
    for name in names[1:]:
        b = load_forecasts(name)
        b = b.drop(columns=[c for c in b.columns if c in cv.columns and c not in KEY])
        cv = cv.merge(b, on=KEY, how="inner")
    return cv


def mase_scales(cfg: BacktestConfig, m: int = 168) -> dict[str, float]:
    """Per-series in-sample seasonal-naive MAE on [train_start, eval_start)."""
    load = pl.read_parquet(PROCESSED_DIR / "load_hourly.parquet")
    t0 = datetime.combine(cfg.train_start, datetime.min.time())
    t1 = datetime.combine(cfg.eval_start, datetime.min.time())
    out = {}
    for sub in cfg.subsystems:
        y = load.filter((pl.col("subsystem") == sub) & (pl.col("ts_local") >= t0) & (pl.col("ts_local") < t1))["load_mw"].to_numpy()
        out[sub] = float(np.mean(np.abs(y[m:] - y[:-m])))
    return out
