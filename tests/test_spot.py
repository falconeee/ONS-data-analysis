"""SPOT (EVT thresholding used by MSCVAE) against analytic quantiles, plus an MSCVAE smoke test."""

import math

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from ons_energy.anomaly.mscvae import MSCVAE, SPOT  # noqa: E402


def _spot_quantile(data: np.ndarray, q: float, level: float = 0.98) -> float:
    s = SPOT(q)
    s.fit(data, data)
    s.initialize(level=level, verbose=False)
    return s.extreme_quantile


def test_spot_exponential_tail():
    # Exp(1): P(X > z) = exp(-z)  =>  z_q = -ln(q). The GPD tail has shape 0, scale 1.
    rng = np.random.default_rng(0)
    data = rng.exponential(1.0, size=50_000)
    q = 1e-3
    assert _spot_quantile(data, q) == pytest.approx(-math.log(q), rel=0.08)


def test_spot_pareto_tail():
    # Lomax/Pareto II with shape a: P(X > z) = (1 + z)^-a  =>  z_q = q^(-1/a) - 1 (GPD shape 1/a).
    rng = np.random.default_rng(1)
    a = 4.0
    data = rng.pareto(a, size=100_000)
    q = 1e-3
    assert _spot_quantile(data, q) == pytest.approx(q ** (-1 / a) - 1, rel=0.15)


_SMOKE = """
import numpy as np, pandas as pd
from ons_energy.anomaly.mscvae import MSCVAE
rng = np.random.default_rng(0)
n, k = 240, 6
t = np.arange(n)
base = np.sin(2 * np.pi * t / 24)[:, None]
df = pd.DataFrame(base + 0.1 * rng.standard_normal((n, k)), columns=[f"x{i}" for i in range(k)])
ts = pd.date_range("2024-01-01", periods=n, freq="h")

model = MSCVAE(window_sizes=(4, 8), stride=1, seq_len=3, device="cpu", seed=0)
model.fit(df.iloc[:160], epochs=1, batch_size=32, verbose=False)
out = model.predict(df.iloc[160:], timestamps=ts[160:])

assert len(out["phi"]) == len(out["timestamp"]) == 80
assert out["timestamp"][0] == ts[160]
assert np.isfinite(out["phi"]).all()
assert np.isfinite(model.threshold)
print("OK")
"""


@pytest.mark.slow
def test_mscvae_fit_predict_smoke():
    # torch training runs in its own process (libomp clash with LightGBM tests on macOS)
    import subprocess
    import sys

    res = subprocess.run([sys.executable, "-c", _SMOKE], capture_output=True, text=True, timeout=600)
    assert res.returncode == 0 and "OK" in res.stdout, res.stderr[-2000:]
