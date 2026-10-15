import numpy as np
import pytest

from ons_energy.forecast import metrics as M
from ons_energy.forecast.stats_tests import diebold_mariano


def test_point_metrics_known_values():
    y = np.array([100.0, 200.0])
    yhat = np.array([110.0, 180.0])
    assert M.mae(y, yhat) == 15
    assert M.rmse(y, yhat) == pytest.approx(np.sqrt((100 + 400) / 2))
    assert M.mape(y, yhat) == pytest.approx(10.0)
    assert M.smape(y, yhat) == pytest.approx(100 * np.mean([20 / 210, 40 / 380]))


def test_mase_of_seasonal_naive_in_sample_is_one():
    rng = np.random.default_rng(0)
    y = rng.normal(size=1000).cumsum() + 100
    m = 24
    scale = M.mase_scale(y, m)
    assert M.mase(y[m:], y[:-m], scale) == pytest.approx(1.0)


def test_pinball_and_coverage():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    assert M.pinball(y, np.zeros(4), 0.9) == pytest.approx(0.9 * np.mean([0, 1, 2, 3]))
    assert M.pinball(y, np.full(4, 3.0), 0.1) == pytest.approx(0.9 * np.mean([3, 2, 1, 0]))
    assert M.coverage(y, np.full(4, 0.5), np.full(4, 2.5)) == 0.5
    # median pinball = MAE / 2
    yhat = np.array([0.5, 0.5, 2.5, 2.5])
    assert M.pinball(y, yhat, 0.5) == pytest.approx(M.mae(y, yhat) / 2)


def test_dm_detects_better_model_and_not_equal_ones():
    rng = np.random.default_rng(3)
    base = rng.gamma(2, 1, 400)
    better = base * 0.8 + rng.normal(0, 0.05, 400)
    r = diebold_mariano(better, base)
    assert r.statistic < 0 and r.p_value < 0.01
    same = diebold_mariano(base + rng.normal(0, 0.01, 400), base + rng.normal(0, 0.01, 400))
    assert same.p_value > 0.05
