"""PCA monitoring with Hotelling T² and SPE (Q), empirical (KDE) limits and RBC contributions.

On standardized data z, with loadings P (m×k) and retained eigenvalues λ:
  T²  = zᵀ D z,  D = P Λ⁻¹ Pᵀ        (variation inside the PCA model)
  SPE = ‖C̃ z‖²,  C̃ = I − P Pᵀ        (variation the model cannot explain)

Limits are the (1-α) quantile of a Gaussian KDE of each statistic on the reference period: on
autocorrelated, heavy-tailed load residuals the parametric F / Box limits under-cover badly.
Reconstruction-based contributions (RBC, Alcala & Qin 2009): RBC_j = (M z)_j² / M_jj, the drop in
the statistic when variable j is reconstructed; the faulty variable has the largest RBC.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq

from ons_energy.anomaly.base import Detector, contribution_frame, finish_scores


def kde_limit(values: np.ndarray, alpha: float) -> float:
    """(1-α) quantile of a Gaussian-KDE fit of ``values``."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    kde = stats.gaussian_kde(v)
    return float(brentq(lambda x: kde.integrate_box_1d(-np.inf, x) - (1 - alpha), v.min(), v.max() * 3 + 1e-9))


def rbc(Z: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Reconstruction-based contributions of each variable to the quadratic statistic zᵀMz."""
    return (Z @ M.T) ** 2 / np.clip(np.diag(M), 1e-12, None)


class PCAMonitor(Detector):
    """Static PCA monitor.

    n_components : fixed number of PCs; if None, the smallest k with cumulative explained variance
        >= ``var_explained``.
    alpha : false-alarm level of each limit (T² and SPE).
    std_floor : standardization floor in data units (MW): a variable almost constant in the
        reference would otherwise turn any later movement into a huge z-score.
    """

    def __init__(
        self,
        n_components: int | None = None,
        var_explained: float = 0.9,
        alpha: float = 0.01,
        std_floor: float = 1.0,
        name: str = "PCA",
    ):
        self.n_components = n_components
        self.var_explained = var_explained
        self.alpha = alpha
        self.std_floor = std_floor
        self.name = name

    def _standardize(self, X: pd.DataFrame) -> np.ndarray:
        return ((X[self.variables_] - self.mean_) / self.std_).to_numpy()

    def _stats(self, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        T = Z @ self.P_
        E = Z - T @ self.P_.T
        return np.sum(T**2 / self.lam_, axis=1), np.sum(E**2, axis=1)

    def fit(self, reference: list[pd.DataFrame]) -> "PCAMonitor":
        X = pd.concat(reference).dropna()
        self.variables_ = list(X.columns)
        self.mean_ = X.mean()
        self.std_ = X.std().clip(lower=self.std_floor)
        Z = self._standardize(X)

        _, s, vt = np.linalg.svd(Z, full_matrices=False)
        eig = s**2 / (len(Z) - 1)
        k = self.n_components or int(np.searchsorted(np.cumsum(eig) / eig.sum(), self.var_explained) + 1)
        self.k_ = min(k, len(eig) - 1)
        self.P_, self.lam_ = vt[: self.k_].T, eig[: self.k_]
        self.D_ = self.P_ @ np.diag(1 / self.lam_) @ self.P_.T  # T² metric
        self.C_ = np.eye(len(self.variables_)) - self.P_ @ self.P_.T  # residual projector

        t2, spe = self._stats(Z)
        self.t2_lim_, self.spe_lim_ = kde_limit(t2, self.alpha), kde_limit(spe, self.alpha)
        self.ref_alarm_rate_ = float(np.mean((t2 > self.t2_lim_) | (spe > self.spe_lim_)))
        return self

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        valid = df[self.variables_].notna().all(axis=1).to_numpy()
        out = pd.DataFrame(index=df.index, columns=["T2", "SPE"], dtype=float)
        if valid.any():
            out.loc[valid, "T2"], out.loc[valid, "SPE"] = self._stats(self._standardize(df.loc[valid]))
        out["T2_lim"], out["SPE_lim"] = self.t2_lim_, self.spe_lim_
        return finish_scores(out, ["T2", "SPE"])

    def contributions(self, df: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
        """RBC to T² and to SPE, each normalized by its limit and summed, averaged over ``mask``."""
        Z = self._standardize(df.loc[mask, self.variables_].dropna())
        score = rbc(Z, self.D_).mean(axis=0) / self.t2_lim_ + rbc(Z, self.C_).mean(axis=0) / self.spe_lim_
        return contribution_frame(self.variables_, score)
