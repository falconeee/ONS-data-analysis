"""Common detector interface so PCA and MSCVAE go through the same evaluation.

Data convention: ``pandas.DataFrame`` indexed by a regular ``DatetimeIndex``, one column per
monitored variable named ``<subsystem>_<quantity>`` (e.g. ``NE_load``).
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Detector(ABC):
    name: str = "detector"

    @abstractmethod
    def fit(self, reference: list[pd.DataFrame]) -> "Detector":
        """Fit on one or more disjoint normal-operation periods."""

    @abstractmethod
    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score a continuous period. Returns a frame indexed like ``df`` with, per statistic S,
        columns ``S`` and ``S_lim``, plus ``ratio`` (max S/S_lim) and ``alarm_raw`` (ratio > 1).
        Rows without enough history (lags/windows) are NaN / False."""

    @abstractmethod
    def contributions(self, df: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
        """Per-variable contribution averaged over the rows of ``df`` selected by ``mask``.
        Columns: variable, subsystem, score, pct (sorted descending)."""


def finish_scores(out: pd.DataFrame, stats: list[str]) -> pd.DataFrame:
    ratios = pd.concat([out[s] / out[f"{s}_lim"] for s in stats], axis=1)
    out["ratio"] = ratios.max(axis=1, skipna=False)
    out["alarm_raw"] = (out["ratio"] > 1).fillna(False)
    return out


def contribution_frame(variables: list[str], score: np.ndarray) -> pd.DataFrame:
    score = np.clip(np.asarray(score, float), 0, None)
    total = score.sum()
    df = pd.DataFrame({
        "variable": variables,
        "subsystem": [v.split("_", 1)[0] for v in variables],
        "score": score,
        "pct": 100 * score / total if total > 0 else 0.0,
    })
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def persistent_alarm(alarm_raw: pd.Series, m: int, n: int) -> pd.Series:
    """Alarm when at least ``m`` of the last ``n`` points exceed the limit (reduces isolated alarms)."""
    return (alarm_raw.astype(float).rolling(n, min_periods=n).sum() >= m).fillna(False)
