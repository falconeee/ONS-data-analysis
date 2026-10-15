"""MSCVAE adapter: alignment, chunked scoring == full scoring (warm-up with real history), and the
author's RCA pointing at an injected fault. Runs in a subprocess (torch; see conftest.py)."""

import subprocess
import sys

import pytest

_SCRIPT = r"""
import numpy as np, pandas as pd
from ons_energy.anomaly.mscvae_adapter import MSCVAEDetector

rng = np.random.default_rng(0)
n, k = 1400, 6
t = np.arange(n)
latent = np.sin(2 * np.pi * t / 24)[:, None] * np.linspace(0.5, 1.5, k) + 0.3 * np.sin(2 * np.pi * t / 168)[:, None]
X = pd.DataFrame(latent + 0.05 * rng.standard_normal((n, k)), columns=[f"{s}_v{i}" for i, s in enumerate(["SE", "SE", "S", "S", "NE", "N"])],
                 index=pd.date_range("2024-01-01", periods=n, freq="h"))

det = MSCVAEDetector(window_sizes=(4, 12), seq_len=3, stride=1, epochs=8, batch_size=64, device="cpu", seed=0, chunk=10_000)
det.fit([X.iloc[:900]])
assert np.isfinite(det.threshold_) and np.isfinite(det.threshold_in_sample_)

test = X.iloc[900:].copy()
full = det.score(test)
assert len(full) == len(test) and (full.index == test.index).all()
assert full["phi"].iloc[: det.warmup].isna().all() and full["phi"].iloc[det.warmup:].notna().all()

det.chunk = 97  # many chunks with overlapping warm-up must give the same scores
chunked = det.score(test)
np.testing.assert_allclose(full["phi"].iloc[det.warmup:], chunked["phi"].iloc[det.warmup:], rtol=1e-4, atol=1e-6)

j = 2
faulty = test.copy()
faulty.iloc[300:340, j] += 6 * X.iloc[:900, j].std()
mask = np.zeros(len(faulty), bool); mask[305:340] = True
c = det.contributions(faulty, mask)
assert X.columns[j] in c["variable"].head(3).tolist(), c.head()
s = det.score(faulty)
assert s["ratio"].iloc[305:340].max() > s["ratio"].iloc[det.warmup:290].quantile(0.99)
print("OK")
"""


@pytest.mark.slow
def test_mscvae_adapter_end_to_end():
    res = subprocess.run([sys.executable, "-c", _SCRIPT], capture_output=True, text=True, timeout=900)
    assert res.returncode == 0 and "OK" in res.stdout, (res.stdout[-1500:], res.stderr[-3000:])
