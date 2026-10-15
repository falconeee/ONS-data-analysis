"""Diebold-Mariano test with the Harvey-Leybourne-Newbold small-sample correction."""

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class DMResult:
    statistic: float
    p_value: float
    mean_diff: float
    n: int


def _newey_west_var(d: np.ndarray, lags: int) -> float:
    n = d.size
    dc = d - d.mean()
    var = dc @ dc / n
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        var += 2 * w * (dc[k:] @ dc[:-k]) / n
    return var


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 1, lags: int | None = None) -> DMResult:
    """Two-sided DM test of equal predictive accuracy; negative statistic = model A better.

    Losses should be one value per forecast occasion (e.g. daily MAE of each model): hourly
    errors within a day are strongly dependent. ``lags`` defaults to max(h-1, n^(1/3)).
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[~np.isnan(d)]
    n = d.size
    lags = lags if lags is not None else max(h - 1, int(np.floor(n ** (1 / 3))))
    var = _newey_west_var(d, lags)
    dm = d.mean() / np.sqrt(var / n)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = hln * dm
    p = 2 * stats.t.sf(np.abs(stat), df=n - 1)
    return DMResult(float(stat), float(p), float(d.mean()), n)
