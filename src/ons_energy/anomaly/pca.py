"""PCA-based multivariate statistical process monitoring (T² and SPE/Q)."""

from typing import Literal

import numpy as np
import pandas as pd

from ons_energy.anomaly import contributions as C
from ons_energy.anomaly import limits as L
from ons_energy.anomaly.base import Detector, contribution_frame, finish_scores

LimitMethod = Literal["parametric", "kde", "spot"]


class PCAMonitor(Detector):
    """Static PCA monitor.

    Parameters
    ----------
    n_components : fixed number of PCs; if None, the smallest k with cumulative explained variance
        >= ``var_explained``.
    alpha : false-alarm level of each limit.
    limit_method : "parametric" (F for T², Box or Jackson-Mudholkar for SPE), "kde" (empirical
        quantile of the reference statistics) or "spot" (EVT, the threshold used by MSCVAE).
    """

    def __init__(
        self,
        n_components: int | None = None,
        var_explained: float = 0.9,
        alpha: float = 0.01,
        limit_method: LimitMethod = "parametric",
        spe_approx: Literal["box", "jm"] = "box",
        min_std: float = 1e-6,
        std_floor: float = 1.0,
        name: str = "PCA",
    ):
        self.n_components = n_components
        self.var_explained = var_explained
        self.alpha = alpha
        self.limit_method = limit_method
        self.spe_approx = spe_approx
        self.min_std = min_std
        # standardization floor (data units: MW or %): a variable almost constant in the
        # reference would otherwise turn any later movement into a huge z-score
        self.std_floor = std_floor
        self.name = name

    # -- preprocessing ------------------------------------------------------------------------
    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Hook for subclasses (DPCA adds lagged copies)."""
        return df

    def _stack_reference(self, reference: list[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat([self._prepare(r) for r in reference]).dropna()

    def _standardize(self, X: pd.DataFrame) -> np.ndarray:
        return ((X[self.variables_] - self.mean_) / self.std_).to_numpy()

    # -- fitting --------------------------------------------------------------------------------
    def fit(self, reference: list[pd.DataFrame]) -> "PCAMonitor":
        X = self._stack_reference(reference)
        std = X.std()
        self.dropped_ = std[std < self.min_std].index.tolist()  # constant in the reference period
        self.variables_ = [c for c in X.columns if c not in self.dropped_]
        self.mean_ = X[self.variables_].mean()
        self.std_ = X[self.variables_].std().clip(lower=self.std_floor)
        Z = self._standardize(X)
        self.n_ref_ = Z.shape[0]

        _, s, vt = np.linalg.svd(Z, full_matrices=False)
        eig = s**2 / (self.n_ref_ - 1)
        k = self.n_components or int(np.searchsorted(np.cumsum(eig) / eig.sum(), self.var_explained) + 1)
        k = min(k, len(eig) - 1)
        self.k_, self.eigenvalues_ = k, eig
        self.P_, self.lam_ = vt[:k].T, eig[:k]
        self._D = C.t2_matrix(self.P_, self.lam_)
        self._Ct = C.spe_matrix(self.P_)

        t2_ref, spe_ref = self._stats(Z)
        if self.limit_method == "parametric":
            self.t2_lim_ = L.t2_limit_f(k, self.n_ref_, self.alpha)
            resid = eig[k:]
            self.spe_lim_ = L.spe_limit_box(resid, self.alpha) if self.spe_approx == "box" else L.spe_limit_jm(resid, self.alpha)
        elif self.limit_method == "kde":
            self.t2_lim_, self.spe_lim_ = L.kde_limit(t2_ref, self.alpha), L.kde_limit(spe_ref, self.alpha)
        else:
            self.t2_lim_, self.spe_lim_ = L.spot_limit(t2_ref, self.alpha), L.spot_limit(spe_ref, self.alpha)
        self.ref_alarm_rate_ = float(np.mean((t2_ref > self.t2_lim_) | (spe_ref > self.spe_lim_)))
        return self

    def _stats(self, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        T = Z @ self.P_
        t2 = np.sum(T**2 / self.lam_, axis=1)
        E = Z - T @ self.P_.T
        return t2, np.sum(E**2, axis=1)

    # -- monitoring -----------------------------------------------------------------------------
    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._prepare(df)
        valid = X[self.variables_].notna().all(axis=1).to_numpy()
        out = pd.DataFrame(index=df.index, columns=["T2", "SPE"], dtype=float)
        if valid.any():
            t2, spe = self._stats(self._standardize(X.loc[valid]))
            out.loc[valid, "T2"], out.loc[valid, "SPE"] = t2, spe
        out["T2_lim"], out["SPE_lim"] = self.t2_lim_, self.spe_lim_
        return finish_scores(out, ["T2", "SPE"])

    def contribution_arrays(self, df: pd.DataFrame, mask: np.ndarray) -> dict[str, np.ndarray]:
        X = self._prepare(df).loc[mask, self.variables_].dropna()
        Z = self._standardize(X)
        return {
            "rbc_t2": C.rbc(Z, self._D).mean(axis=0),
            "rbc_spe": C.rbc(Z, self._Ct).mean(axis=0),
            "cdc_t2": C.cdc(Z, self._D).mean(axis=0),
            "spe_partial": C.spe_partial(Z, self.P_).mean(axis=0),
        }

    def _aggregate_vars(self, arr: np.ndarray) -> tuple[list[str], np.ndarray]:
        return self.variables_, arr

    def contributions(self, df: pd.DataFrame, mask: np.ndarray, kind: str = "rbc") -> pd.DataFrame:
        """RBC by default: T² and SPE RBC, each normalized by its limit, summed."""
        a = self.contribution_arrays(df, mask)
        if kind == "rbc":
            score = a["rbc_t2"] / self.t2_lim_ + a["rbc_spe"] / self.spe_lim_
        else:
            score = np.clip(a["cdc_t2"], 0, None) / self.t2_lim_ + a["spe_partial"] / self.spe_lim_
        names, agg = self._aggregate_vars(score)
        return contribution_frame(names, agg)
