"""Rolling (causal) conformal intervals for point forecasters.

For a forecast issued at origin D-1 ``origin_hour``:00, the residuals already observable are those
of target days <= D-2. Quantiles of the residuals by forecast step over the previous
``window_days`` such days are added to the point forecast. Every model without native quantiles
gets intervals this way, so coverage and pinball loss are comparable with the TFT's MQLoss.
"""

import numpy as np
import pandas as pd


def add_rolling_conformal(
    cv: pd.DataFrame, model: str, quantiles: list[float], window_days: int = 90, min_days: int = 30
) -> pd.DataFrame:
    """Add ``{model}-q{q}`` columns. ``cv`` needs unique_id, ds, cutoff, step, y, ``model``."""
    out = cv.copy()
    out["_target_day"] = out["ds"].dt.normalize()
    out["_resid"] = out["y"] - out[model]
    cols = {q: f"{model}-q{q:g}" for q in quantiles}
    for c in cols.values():
        out[c] = np.nan

    for uid, g in out.groupby("unique_id"):
        days = np.sort(g["_target_day"].unique())
        # residual matrix: rows = target day, cols = step  (one forecast per target day and step)
        res = g.pivot_table(index="_target_day", columns="step", values="_resid", aggfunc="first")
        res = res.reindex(days)
        qtab: dict[pd.Timestamp, pd.DataFrame] = {}
        for i, day in enumerate(days):
            # issued at D-1 -> residuals of target days <= D-2 are known
            hist = res.iloc[max(0, i - 1 - window_days) : max(0, i - 1)]
            if len(hist.dropna(how="all")) < min_days:
                continue
            qtab[day] = hist.quantile(quantiles)  # index = quantiles, columns = steps
        idx = g.index
        for day, qt in qtab.items():
            rows = idx[(g["_target_day"] == day).to_numpy()]
            steps = out.loc[rows, "step"].to_numpy()
            for q, c in cols.items():
                out.loc[rows, c] = out.loc[rows, model].to_numpy() + qt.loc[q].reindex(steps).to_numpy()
    return out.drop(columns=["_target_day", "_resid"])
