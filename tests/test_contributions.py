import numpy as np
import pandas as pd
import pytest

from ons_energy.anomaly import contributions as C
from ons_energy.anomaly.pca import PCAMonitor


def _model(seed: int = 0, m: int = 10, k: int = 3):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(k, m))
    X = rng.normal(size=(5000, k)) @ W + 0.2 * rng.normal(size=(5000, m))
    cols = [f"{'SE' if i < 5 else 'NE'}_v{i}" for i in range(m)]
    ref = pd.DataFrame(X, columns=cols, index=pd.date_range("2020-01-01", periods=5000, freq="h"))
    return PCAMonitor(n_components=k).fit([ref]), ref


def test_cdc_and_partial_contributions_sum_to_statistics():
    m, ref = _model()
    Z = m._standardize(ref.iloc[:200])
    t2, spe = m._stats(Z)
    np.testing.assert_allclose(C.cdc(Z, m._D).sum(1), t2, rtol=1e-8)
    np.testing.assert_allclose(C.spe_partial(Z, m.P_).sum(1), spe, rtol=1e-8)


@pytest.mark.parametrize("j", [1, 4, 7])
def test_rbc_points_at_the_faulty_variable(j):
    m, ref = _model(seed=j)
    test = ref.iloc[:300].copy()
    test.iloc[:, j] += 5 * ref.iloc[:, j].std()
    c = m.contributions(test, np.ones(len(test), bool))
    assert c.iloc[0]["variable"] == ref.columns[j]
    assert c["pct"].sum() == pytest.approx(100)
    assert c.iloc[0]["subsystem"] == ref.columns[j].split("_")[0]


def test_rbc_equals_statistic_reduction_after_reconstruction():
    # RBC_j = SPE(z) - min_f SPE(z - f ξ_j)
    m, ref = _model(seed=11)
    z = m._standardize(ref.iloc[[0]])[0] + np.eye(10)[3] * 4
    M = m._Ct
    rbc = C.rbc(z[None, :], M)[0, 3]
    xi = np.eye(10)[3]
    f = (xi @ M @ z) / (xi @ M @ xi)
    reduced = (z - f * xi) @ M @ (z - f * xi)
    assert rbc == pytest.approx(z @ M @ z - reduced, rel=1e-8)
