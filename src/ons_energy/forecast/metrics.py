"""Point and probabilistic forecast metrics (NumPy, NaN-safe inputs are the caller's job)."""

import numpy as np


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
