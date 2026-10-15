"""PCA monitor: false-alarm rate under H0, sensitivity to injected faults, DPCA/RPCA mechanics."""

import numpy as np
import pandas as pd
import pytest

from ons_energy.anomaly.base import persistent_alarm
from ons_energy.anomaly.dpca import DPCAMonitor, lag_matrix
from ons_energy.anomaly.pca import PCAMonitor
from ons_energy.anomaly.rpca import MovingWindowPCA


def _latent_data(n: int, m: int = 8, k: int = 3, seed: int = 0, noise: float = 0.3, w_seed: int | None = None) -> pd.DataFrame:
    """Latent-factor data; ``w_seed`` fixes the loadings so two samples share the same structure."""
    rng = np.random.default_rng(seed)
    W = np.random.default_rng(seed if w_seed is None else w_seed).normal(size=(k, m))
    X = rng.normal(size=(n, k)) @ W + noise * rng.normal(size=(n, m))
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    return pd.DataFrame(X, index=idx, columns=[f"S{i % 4}_v{i}" for i in range(m)])


def test_false_alarm_rate_close_to_alpha_on_gaussian_data():
    ref = _latent_data(5000, seed=1)
    test = _latent_data(20000, seed=99, w_seed=1)  # same loadings, new samples -> H0 holds

    m = PCAMonitor(n_components=3, alpha=0.01).fit([ref])
    s = m.score(test)
    assert (s["T2"] > s["T2_lim"]).mean() == pytest.approx(0.01, abs=0.006)
    assert (s["SPE"] > s["SPE_lim"]).mean() == pytest.approx(0.01, abs=0.006)


def test_spe_detects_correlation_break_and_t2_detects_large_shift():
    ref = _latent_data(4000, seed=3)
    m = PCAMonitor(n_components=3, alpha=0.01).fit([ref])
    test = ref.iloc[:500].copy()
    test.iloc[:, 2] += 4 * ref.iloc[:, 2].std()  # one sensor off the latent structure
    s = m.score(test)
    assert (s["SPE"] > s["SPE_lim"]).mean() > 0.9


def test_variance_explained_rule_and_constant_columns_dropped():
    ref = _latent_data(3000, seed=4)
    ref["N_solar"] = 0.0  # did not exist in the reference period
    m = PCAMonitor(var_explained=0.9).fit([ref])
    assert "N_solar" in m.dropped_ and "N_solar" not in m.variables_
    assert 1 <= m.k_ <= 5
    assert np.cumsum(m.eigenvalues_)[m.k_ - 1] / m.eigenvalues_.sum() >= 0.9


def test_multiple_reference_periods_are_stacked():
    a, b = _latent_data(1500, seed=5), _latent_data(1500, seed=5).shift(freq="365D")
    m = PCAMonitor(n_components=2).fit([a, b])
    assert m.n_ref_ == 3000


def test_dpca_lag_matrix_and_period_boundaries():
    ref = _latent_data(200, seed=6)
    L = lag_matrix(ref, 2)
    assert L.shape[1] == 3 * ref.shape[1] and L.iloc[:2].isna().any(axis=None)
    m = DPCAMonitor(lags=2, n_components=4).fit([ref.iloc[:100], ref.iloc[100:]])
    assert m.n_ref_ == 200 - 2 * 2  # two lag rows lost per period, never bridged
    c = m.contributions(ref, np.ones(len(ref), bool))
    assert set(c["variable"]) == set(ref.columns)  # aggregated back to original variables


def _shifted_followup(ref: pd.DataFrame, n: int = 3000, shift_sd: float = 3.0) -> pd.DataFrame:
    later = _latent_data(n, seed=8, w_seed=7)
    later.index = pd.date_range(ref.index[-1] + pd.Timedelta(hours=1), periods=n, freq="h")
    later.iloc[:, 0] += shift_sd * ref.iloc[:, 0].std()  # structural (permanent) shift
    return later


def test_rpca_accepts_a_permanent_shift_but_static_pca_keeps_alarming():
    ref = _latent_data(2000, seed=7)
    later = _shifted_followup(ref)
    static = PCAMonitor(n_components=3, alpha=0.01).fit([ref])
    rpca = MovingWindowPCA(window=1000, update_every=100, accept_after=200, n_components=3, alpha=0.01).fit([ref])
    assert static.score(later)["alarm_raw"].iloc[-1000:].mean() > 0.5
    assert rpca.score(later)["alarm_raw"].iloc[-1000:].mean() < 0.1


def test_rpca_guard_without_acceptance_never_learns_the_fault():
    ref = _latent_data(2000, seed=7)
    later = _shifted_followup(ref)
    rpca = MovingWindowPCA(window=1000, update_every=100, accept_after=None, n_components=3, alpha=0.01).fit([ref])
    assert rpca.score(later)["alarm_raw"].iloc[-1000:].mean() > 0.5


def test_persistence_rule():
    raw = pd.Series([0, 1, 1, 0, 1, 1, 1, 0, 0, 1], dtype=bool)
    out = persistent_alarm(raw, m=3, n=4).tolist()
    assert out == [False, False, False, False, True, True, True, True, False, False]
