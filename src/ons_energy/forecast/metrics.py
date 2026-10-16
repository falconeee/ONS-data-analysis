"""Point and probabilistic forecast metrics and the Diebold-Mariano test (NumPy; NaN-safe inputs
are the caller's job)."""

import numpy as np
from scipy import stats


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - yhat)))


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mape(y: np.ndarray, yhat: np.ndarray) -> float:
    """Mean absolute percentage error, in %."""
    return float(100 * np.mean(np.abs((y - yhat) / y)))


def smape(y: np.ndarray, yhat: np.ndarray) -> float:
    """Symmetric MAPE in % (0-200 scale)."""
    return float(100 * np.mean(2 * np.abs(y - yhat) / (np.abs(y) + np.abs(yhat))))


def mase_scale(y_train: np.ndarray, m: int = 168) -> float:
    """In-sample MAE of the seasonal-naive forecast with period ``m`` (MASE denominator)."""
    return float(np.mean(np.abs(y_train[m:] - y_train[:-m])))


def mase(y: np.ndarray, yhat: np.ndarray, scale: float) -> float:
    return float(np.mean(np.abs(y - yhat)) / scale)


def pinball(y: np.ndarray, qhat: np.ndarray, q: float) -> float:
    """Quantile (pinball) loss for quantile level ``q``."""
    diff = y - qhat
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def mean_pinball(y: np.ndarray, qhats: dict[float, np.ndarray]) -> float:
    return float(np.mean([pinball(y, v, q) for q, v in qhats.items()]))


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean((y >= lo) & (y <= hi)))


def interval_width(lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean(hi - lo))


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 1) -> dict[str, float]:
    """Two-sided Diebold-Mariano test with the Harvey-Leybourne-Newbold correction; negative
    statistic = model A better. Use one loss per forecast occasion (e.g. daily MAPE): hourly errors
    within a day are strongly dependent. Newey-West variance with n^(1/3) lags."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[~np.isnan(d)]
    n = d.size
    lags = max(h - 1, int(np.floor(n ** (1 / 3))))
    dc = d - d.mean()
    var = dc @ dc / n + sum(2 * (1 - k / (lags + 1)) * (dc[k:] @ dc[:-k]) / n for k in range(1, lags + 1))
    stat = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n) * d.mean() / np.sqrt(var / n)
    return {"statistic": float(stat), "p_value": float(2 * stats.t.sf(np.abs(stat), df=n - 1)), "mean_diff": float(d.mean()), "n": n}
