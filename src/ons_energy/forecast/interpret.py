"""Interpretability of the forecasters, segmented by day type where it matters.

* TFT: variable-selection weights (past / future inputs) and temporal attention. NeuralForecast keeps
  them for the LAST predicted batch only, so each origin is predicted separately and the weights are
  collected per origin, then averaged per day type (normal vs. holidays and other special days).
* N-BEATS (interpretable): trend / seasonality stack outputs for selected origins (``decompose``).
* LightGBM: SHAP values via ``pred_contrib=True`` (exact TreeSHAP built into LightGBM).

The neural and LightGBM parts must run in separate processes (libomp clash, see package docs).
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from ons_energy.calendar.segments import day_types
from ons_energy.features.forecast import build_panel
from ons_energy.forecast.config import BacktestConfig
from ons_energy.utils.logging import get_logger

log = get_logger(__name__)


def _origins(cfg: BacktestConfig, end: datetime.date, n_normal: int = 25, seed: int = 0) -> pd.DataFrame:
    """Target days in the evaluation period: every special day plus a random sample of normal days."""
    sub = cfg.subsystems[0]
    dt = day_types(cfg.eval_start, end, sub).to_pandas()
    special = dt[dt.day_type != "normal"]
    normal = dt[dt.day_type == "normal"].sample(n=min(n_normal, (dt.day_type == "normal").sum()), random_state=seed)
    return pd.concat([special, normal]).sort_values("date").reset_index(drop=True)


def _fit_until(nf, df: pd.DataFrame, cutoff: pd.Timestamp):
    nf.fit(df=df[df.ds <= cutoff], val_size=0)
    return nf


def neural_interpretability(cfg: BacktestConfig, end: datetime.date) -> dict[str, pd.DataFrame]:
    from ons_energy.forecast import models as factory

    panel = build_panel(cfg.subsystems, cfg.train_start, end, cfg.weather, weather_switch=cfg.eval_start - timedelta(days=cfg.calib_days))
    df = panel.df
    specs = [s for s in cfg.models if s.kind in ("tft", "nbeats_i")]
    if not specs:
        return {}
    nf = factory.neural(specs, cfg, panel.futr_exog)
    fit_cut = pd.Timestamp(cfg.eval_start) - pd.Timedelta(hours=13)
    _fit_until(nf, df, fit_cut)
    log.info("interpret: models fitted up to %s", fit_cut)

    origins = _origins(cfg, end)
    vsn_rows, att_rows, dec_rows = [], [], []
    for r in origins.itertuples():
        cutoff = pd.Timestamp(r.date) - pd.Timedelta(hours=13)  # D-1 11:00
        hist = df[df.ds <= cutoff]
        futr = df[(df.ds > cutoff) & (df.ds <= cutoff + pd.Timedelta(hours=cfg.horizon))].drop(columns="y")
        nf.predict(df=hist, futr_df=futr)
        for model in nf.models:
            name = model.alias or type(model).__name__
            if type(model).__name__ == "TFT":
                imp = model.feature_importances()
                fut = imp.get("Future variable importance over time")
                past = imp.get("Past variable importance over time")
                for kind, tab in (("future", fut), ("past", past)):
                    if tab is not None:
                        for var, w in tab.mean(axis=0).items():
                            vsn_rows.append({"model": name, "date": r.date, "day_type": r.day_type, "block": kind, "variable": var, "weight": float(w)})
                att = model.attention_weights()
                att = np.asarray(att.detach().cpu() if hasattr(att, "detach") else att)
                # attention of the decoder steps of day D (steps 13..36) over encoder positions
                att_rows.append({"model": name, "date": r.date, "day_type": r.day_type, "attention": att})
            elif type(model).__name__ == "NBEATS":
                from neuralforecast.tsdataset import TimeSeriesDataset

                ds_obj, *_ = TimeSeriesDataset.from_df(hist[["unique_id", "ds", "y"]])
                comp = model.decompose(dataset=ds_obj)  # (n_series, 1 + n_stacks, h) [, quantiles]
                comp = np.squeeze(comp)
                if comp.ndim == 2 and comp.shape[0] >= 3:
                    for step in range(comp.shape[1]):
                        dec_rows.append({"date": r.date, "day_type": r.day_type, "step": step + 1,
                                         "level": float(comp[0, step]), "trend": float(comp[1, step]),
                                         "seasonality": float(comp[2, step])})

    out: dict[str, pd.DataFrame] = {"origins": origins}
    if vsn_rows:
        vsn = pd.DataFrame(vsn_rows)
        vsn["segment"] = np.where(vsn.day_type == "normal", "normal", "special")
        out["tft_vsn"] = vsn.groupby(["model", "block", "segment", "variable"], as_index=False)["weight"].mean()
        out["tft_vsn_by_day_type"] = vsn.groupby(["model", "block", "day_type", "variable"], as_index=False)["weight"].mean()
    if att_rows:
        out["tft_attention"] = _attention_table(att_rows, cfg)
    if dec_rows:
        out["nbeats_decomposition"] = pd.DataFrame(dec_rows)
    return out


def _attention_table(rows: list[dict], cfg: BacktestConfig) -> pd.DataFrame:
    """Mean attention by relative encoder lag (hours before the origin), for normal vs special days."""
    recs = []
    for r in rows:
        a = r["attention"]
        while a.ndim > 2:  # average heads / extra dims
            a = a.mean(axis=0)
        # a: (decoder+encoder positions, encoder+decoder) -> take the rows of day-D decoder steps
        n_enc = a.shape[1] - cfg.horizon
        lo, hi = cfg.eval_steps
        dec_rows = a[n_enc + lo - 1 : n_enc + hi] if a.shape[0] > n_enc else a
        prof = dec_rows[:, :n_enc].mean(axis=0)
        seg = "normal" if r["day_type"] == "normal" else "special"
        for i, w in enumerate(prof):
            recs.append({"model": r["model"], "segment": seg, "lag_h": n_enc - i, "weight": float(w)})
    return pd.DataFrame(recs).groupby(["model", "segment", "lag_h"], as_index=False)["weight"].mean()


def lgbm_shap(cfg: BacktestConfig, end: datetime.date) -> dict[str, pd.DataFrame]:
    """Mean |SHAP| per feature on the evaluation period, normal vs special days (one-step features)."""
    from ons_energy.forecast import models as factory

    panel = build_panel(cfg.subsystems, cfg.train_start, end, cfg.weather, weather_switch=cfg.eval_start - timedelta(days=cfg.calib_days))
    df = panel.df
    spec = next(s for s in cfg.models if s.kind == "lgbm")
    mlf = factory.lgbm([spec], cfg.seed)
    prep = mlf.preprocess(df, static_features=[])
    X = prep.drop(columns=["unique_id", "ds", "y"])
    train = prep.ds < pd.Timestamp(cfg.eval_start)
    model = mlf.models[spec.name]
    model.fit(X[train], prep.loc[train, "y"])
    test = ~train
    contrib = model.predict(X[test], pred_contrib=True)[:, :-1]  # last column = expected value
    dts = []
    for uid in cfg.subsystems:
        d = day_types(cfg.eval_start, end, uid).to_pandas()
        d["unique_id"] = uid
        dts.append(d)
    dt = pd.concat(dts)
    dt["date"] = pd.to_datetime(dt["date"])
    meta = prep.loc[test, ["unique_id", "ds"]].assign(date=lambda x: x.ds.dt.normalize()).merge(dt, on=["unique_id", "date"], how="left")
    seg = np.where(meta.day_type.to_numpy() == "normal", "normal", "special")
    rows = []
    for s in ("normal", "special"):
        m = seg == s
        for j, f in enumerate(X.columns):
            rows.append({"segment": s, "feature": f, "mean_abs_shap": float(np.abs(contrib[m, j]).mean())})
    return {"lgbm_shap": pd.DataFrame(rows).sort_values(["segment", "mean_abs_shap"], ascending=[True, False])}
