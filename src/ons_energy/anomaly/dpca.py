"""Dynamic PCA (Ku, Storer & Georgakis 1995): PCA on the lag-augmented matrix [x_t, …, x_{t-l}]."""

import numpy as np
import pandas as pd

from ons_energy.anomaly.pca import PCAMonitor


def lag_matrix(df: pd.DataFrame, lags: int) -> pd.DataFrame:
    parts = [df] + [df.shift(i).add_suffix(f"@{i}") for i in range(1, lags + 1)]
    return pd.concat(parts, axis=1)


def score_autocorrelation(t2: pd.Series, lag: int = 1) -> float:
    s = t2.dropna()
    return float(s.autocorr(lag)) if len(s) > lag + 2 else np.nan


class DPCAMonitor(PCAMonitor):
    """PCA monitor over lagged copies. Contributions are summed back per original variable."""

    def __init__(self, lags: int = 1, name: str = "DPCA", **kw):
        super().__init__(name=name, **kw)
        self.lags = lags

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        # each reference period is lagged separately, so lags never cross period boundaries
        return lag_matrix(df, self.lags)

    def _aggregate_vars(self, arr: np.ndarray) -> tuple[list[str], np.ndarray]:
        base = [v.split("@")[0] for v in self.variables_]
        s = pd.Series(arr, index=base).groupby(level=0, sort=False).sum()
        return s.index.tolist(), s.to_numpy()


def select_lags(reference: list[pd.DataFrame], max_lags: int = 3, **kw) -> pd.DataFrame:
    """Lag-1 autocorrelation of T² on the reference for l = 0..max_lags (lower = whiter scores)."""
    rows = []
    for l in range(max_lags + 1):
        m = DPCAMonitor(lags=l, **kw).fit(reference) if l else PCAMonitor(**kw).fit(reference)
        t2 = pd.concat([m.score(r)["T2"] for r in reference])
        rows.append({"lags": l, "k": m.k_, "acf1_T2": score_autocorrelation(t2), "ref_alarm_rate": m.ref_alarm_rate_})
    return pd.DataFrame(rows)
