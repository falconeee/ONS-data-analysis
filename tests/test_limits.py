import numpy as np
import pytest
from scipy import stats

from ons_energy.anomaly import limits as L


def test_t2_f_limit_matches_monte_carlo():
    # T² of a new observation with estimated mean/cov follows k(n²-1)/(n(n-k)) F(k, n-k)
    rng = np.random.default_rng(0)
    k, n, alpha, reps = 3, 50, 0.05, 4000
    vals = []
    for _ in range(reps):
        ref = rng.normal(size=(n, k))
        mu, cov = ref.mean(0), np.cov(ref, rowvar=False)
        x = rng.normal(size=k) - mu
        vals.append(x @ np.linalg.solve(cov, x))
    assert np.quantile(vals, 1 - alpha) == pytest.approx(L.t2_limit_f(k, n, alpha), rel=0.08)
    assert L.t2_limit_f(k, 10**6, alpha) == pytest.approx(L.t2_limit_chi2(k, alpha), rel=1e-3)


# JM is known to be less accurate with few, very unequal residual eigenvalues (here ~6% at α=1%);
# with equal eigenvalues it is near exact (χ²(6): 16.84 vs 16.81).
@pytest.mark.parametrize("approx, tol", [(L.spe_limit_box, 0.05), (L.spe_limit_jm, 0.08)])
def test_spe_limits_match_monte_carlo(approx, tol):
    # SPE = Σ λ_i z_i² over the discarded components (weighted chi-square)
    rng = np.random.default_rng(1)
    lam = np.array([0.8, 0.5, 0.3, 0.2, 0.1, 0.05])
    spe = (rng.normal(size=(200_000, lam.size)) ** 2) @ lam
    for alpha in (0.05, 0.01):
        assert approx(lam, alpha) == pytest.approx(np.quantile(spe, 1 - alpha), rel=tol)


def test_kde_and_spot_limits_on_chi2_sample():
    rng = np.random.default_rng(2)
    x = rng.chisquare(4, 50_000)
    assert L.kde_limit(x, 0.01) == pytest.approx(stats.chi2.ppf(0.99, 4), rel=0.05)
    assert L.spot_limit(x, 1e-3) == pytest.approx(stats.chi2.ppf(1 - 1e-3, 4), rel=0.1)


def test_jm_exact_for_equal_eigenvalues():
    assert L.spe_limit_jm(np.ones(6), 0.01) == pytest.approx(stats.chi2.ppf(0.99, 6), rel=0.01)
