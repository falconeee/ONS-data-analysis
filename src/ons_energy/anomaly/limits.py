"""Control limits for T² and SPE (Q) statistics."""

import numpy as np
from scipy import stats
from scipy.optimize import brentq


def t2_limit_f(k: int, n: int, alpha: float) -> float:
    """T² limit for a NEW observation, PCA with k components fitted on n samples:
    k (n² - 1) / (n (n - k)) · F_{1-α}(k, n - k)."""
    return float(k * (n**2 - 1) / (n * (n - k)) * stats.f.ppf(1 - alpha, k, n - k))


def t2_limit_chi2(k: int, alpha: float) -> float:
    """Large-sample T² limit: χ²_{1-α}(k)."""
    return float(stats.chi2.ppf(1 - alpha, k))


def spe_limit_box(residual_eigs: np.ndarray, alpha: float) -> float:
    """Box (1954) approximation: SPE ~ g·χ²(h), g = θ2/θ1, h = θ1²/θ2 (θi = Σ λ^i of discarded PCs)."""
    lam = np.asarray(residual_eigs, float)
    theta1, theta2 = lam.sum(), (lam**2).sum()
    g, h = theta2 / theta1, theta1**2 / theta2
    return float(g * stats.chi2.ppf(1 - alpha, h))


def spe_limit_jm(residual_eigs: np.ndarray, alpha: float) -> float:
    """Jackson & Mudholkar (1979) limit for SPE."""
    lam = np.asarray(residual_eigs, float)
    t1, t2, t3 = lam.sum(), (lam**2).sum(), (lam**3).sum()
    h0 = 1 - 2 * t1 * t3 / (3 * t2**2)
    c = stats.norm.ppf(1 - alpha)
    return float(t1 * (c * np.sqrt(2 * t2 * h0**2) / t1 + 1 + t2 * h0 * (h0 - 1) / t1**2) ** (1 / h0))


def kde_limit(values: np.ndarray, alpha: float) -> float:
    """Empirical (1-α) quantile of a Gaussian-KDE fit; robust to non-Gaussian / autocorrelated stats."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    kde = stats.gaussian_kde(v)
    lo, hi = v.min(), v.max() * 3 + 1e-9

    def cdf_gap(x: float) -> float:
        return kde.integrate_box_1d(-np.inf, x) - (1 - alpha)

    return float(brentq(cdf_gap, lo, hi))


def spot_limit(values: np.ndarray, q: float, level: float = 0.98, max_iter: int = 2000) -> float:
    """EVT threshold (SPOT/GPD tail fit) with exceedance probability ``q``; same routine as MSCVAE."""
    from ons_energy.anomaly.mscvae import SPOT

    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    lv = level
    for _ in range(max_iter):
        try:
            s = SPOT(q)
            s.fit(v, v)
            s.initialize(level=lv, min_extrema=False, verbose=False)
            if np.isfinite(s.extreme_quantile):
                return float(s.extreme_quantile)
        except Exception:  # GPD fit failed: lower the peak level, as MSCVAE._pot_eval does
            pass
        lv *= 0.999
    raise RuntimeError("SPOT did not converge")
