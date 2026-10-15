"""Metric tables of a backtest: overall, by subsystem, by day type, by step; DM tests; CIs."""

from datetime import date

import numpy as np
import pandas as pd

from ons_energy.calendar.segments import DAY_TYPES, day_types
from ons_energy.forecast import metrics as M
from ons_energy.forecast.stats_tests import diebold_mariano

QUANTILES = [0.05, 0.1, 0.5, 0.9, 0.95]


def model_columns(cv: pd.DataFrame) -> list[str]:
    fixed = {"unique_id", "ds", "cutoff", "y", "step", "target_date", "day_type"}
    return [c for c in cv.columns if c not in fixed and "-q" not in c]


def attach_day_types(cv: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for uid, g in cv.groupby("unique_id"):
        start, end = g["target_date"].min(), g["target_date"].max()
        dt = day_types(start, end, uid).to_pandas().rename(columns={"date": "target_date"})
        dt["target_date"] = pd.to_datetime(dt["target_date"]).dt.date
        frames.append(g.merge(dt, on="target_date", how="left"))
    return pd.concat(frames, ignore_index=True)


def _row(g: pd.DataFrame, model: str, scale: dict[str, float]) -> dict:
    g = g[g[model].notna()]
    if g.empty:
        return {}
    y, yhat = g["y"].to_numpy(), g[model].to_numpy()
    sc = np.array([scale[u] for u in g["unique_id"]])
    row = {
        "n_days": g["target_date"].nunique(),
        "MAPE": M.mape(y, yhat),
        "sMAPE": M.smape(y, yhat),
        "RMSE": M.rmse(y, yhat),
        "MAE": M.mae(y, yhat),
        "MASE": float(np.mean(np.abs(y - yhat) / sc)),
    }
    qcols = {q: f"{model}-q{q:g}" for q in QUANTILES}
    if all(c in g.columns for c in qcols.values()) and g[qcols[0.05]].notna().all():
        row["pinball"] = M.mean_pinball(y, {q: g[c].to_numpy() for q, c in qcols.items()})
        row["cov80"] = M.coverage(y, g[qcols[0.1]].to_numpy(), g[qcols[0.9]].to_numpy())
        row["cov90"] = M.coverage(y, g[qcols[0.05]].to_numpy(), g[qcols[0.95]].to_numpy())
        row["width90"] = M.interval_width(g[qcols[0.05]].to_numpy(), g[qcols[0.95]].to_numpy())
    return row


def metrics_table(cv: pd.DataFrame, scale: dict[str, float], by: list[str] | None = None) -> pd.DataFrame:
    """One row per (model, *by). Rows where a model has no forecast (e.g. ONS gaps) are skipped."""
    rows = []
    groups = cv.groupby(by) if by else [((), cv)]
    for key, g in groups:
        key = key if isinstance(key, tuple) else (key,)
        for m in model_columns(cv):
            r = _row(g, m, scale)
            if r:
                rows.append({"model": m, **dict(zip(by or [], key)), **r})
    out = pd.DataFrame(rows)
    if by and "day_type" in by:
        out["day_type"] = pd.Categorical(out["day_type"], DAY_TYPES, ordered=True)
        out = out.sort_values(["day_type", "MAPE"])
    return out.reset_index(drop=True)


def daily_loss(cv: pd.DataFrame, model: str) -> pd.Series:
    """Daily absolute percentage error (one forecast occasion = one target day per series)."""
    ape = (cv["y"] - cv[model]).abs() / cv["y"]
    return ape.groupby([cv["unique_id"], cv["target_date"]]).mean()


def dm_table(cv: pd.DataFrame, reference: str) -> pd.DataFrame:
    rows = []
    ref = daily_loss(cv, reference)
    for m in model_columns(cv):
        if m == reference:
            continue
        loss = daily_loss(cv, m)
        both = pd.concat([loss, ref], axis=1, keys=["m", "r"]).dropna()
        res = diebold_mariano(both["m"].to_numpy(), both["r"].to_numpy())
        rows.append({"model": m, "vs": reference, "DM": res.statistic, "p_value": res.p_value,
                     "mean_diff_pp": 100 * res.mean_diff, "n_days": res.n})
    return pd.DataFrame(rows)


def bootstrap_mape_ci(cv: pd.DataFrame, model: str, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Day-block bootstrap CI for MAPE (hours within a day are resampled together)."""
    daily = daily_loss(cv, model).dropna().to_numpy()
    if daily.size < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boots = rng.choice(daily, size=(n_boot, daily.size), replace=True).mean(axis=1) * 100
    return tuple(np.quantile(boots, [alpha / 2, 1 - alpha / 2]))


def segment_table(cv: pd.DataFrame, scale: dict[str, float], n_boot: int = 1000) -> pd.DataFrame:
    """MAPE by day type with day-bootstrap 95% CI and number of days."""
    tab = metrics_table(cv, scale, by=["day_type"])
    lo, hi = [], []
    for _, r in tab.iterrows():
        g = cv[cv["day_type"] == r["day_type"]]
        ci = bootstrap_mape_ci(g, r["model"], n_boot=n_boot)
        lo.append(ci[0])
        hi.append(ci[1])
    tab["MAPE_lo"], tab["MAPE_hi"] = lo, hi
    return tab


def evaluate_backtest(cv: pd.DataFrame, scale: dict[str, float], reference: str | None = None) -> dict[str, pd.DataFrame]:
    cv = attach_day_types(cv)
    out = {
        "overall": metrics_table(cv, scale),
        "by_subsystem": metrics_table(cv, scale, by=["unique_id"]),
        "by_day_type": segment_table(cv, scale),
        "by_step": metrics_table(cv, scale, by=["step"])[["model", "step", "MAPE", "n_days"]],
    }
    if reference:
        out["dm"] = dm_table(cv, reference)
    return out


def summarize_period(cv: pd.DataFrame) -> tuple[date, date]:
    return cv["target_date"].min(), cv["target_date"].max()

