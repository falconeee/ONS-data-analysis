"""Adapter: the author's MSCVAE (``anomaly/mscvae.py``, untouched) behind the ``Detector`` interface.

Project-specific behaviour lives here, not in the algorithm:

1. **Warm-up with real history.** ``AttributeMatrixGenerator.generate`` and ``SequenceMatrixDataset``
   left-pad by repeating the first row/matrix, so the first ``max(w) + seq_len`` windows of any frame
   are artificial. Scoring is done on one continuous frame, in chunks that overlap by that warm-up,
   and the padded rows at the very start are returned as NaN.
2. **Out-of-sample threshold.** ``MSCVAE.fit`` calibrates SPOT on the *training* scores, which are
   optimistic. The tail ``val_fraction`` of each reference period is held out; the SPOT threshold
   and the per-variable RCA baseline are recalibrated on it. Both thresholds are kept.
3. **Bounded SPOT.** ``_pot_eval`` loops ``while True``; here the same procedure runs with an
   iteration cap (``limits.spot_limit``).
4. **Missing values.** The generator turns NaN into 0 (the mean). Gaps are interpolated before
   (short) and counted, so missing data is never silently scored as "normal".
5. **Stride.** Training may use ``stride > 1`` (memory); scoring always uses stride 1.
"""

import numpy as np
import pandas as pd
import torch

from ons_energy.anomaly.base import Detector, contribution_frame, finish_scores
from ons_energy.anomaly.limits import spot_limit
from ons_energy.anomaly.mscvae import MSCVAE
from ons_energy.features.mspc import variable_table
from ons_energy.utils.logging import get_logger

log = get_logger(__name__)


def _impute(df: pd.DataFrame, limit: int = 6) -> tuple[pd.DataFrame, int]:
    n_missing = int(df.isna().any(axis=1).sum())
    out = df.interpolate(limit=limit, limit_direction="both").ffill().bfill()
    return out, n_missing


class MSCVAEDetector(Detector):
    def __init__(
        self,
        window_sizes: tuple[int, ...] = (6, 24, 168),
        seq_len: int = 6,
        stride: int = 1,
        latent_dim: int | None = None,
        epochs: int = 20,
        batch_size: int = 128,
        lr: float = 1e-3,
        val_fraction: float = 0.2,
        q: float = 1e-3,
        level: float = 0.98,
        device: str | None = None,
        seed: int = 42,
        chunk: int = 20_000,
        name: str = "MSCVAE",
    ):
        self.window_sizes = tuple(window_sizes)
        self.seq_len = seq_len
        self.stride = stride
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.val_fraction = val_fraction
        self.q = q
        self.level = level
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.seed = seed
        self.chunk = chunk
        self.name = name

    @property
    def warmup(self) -> int:
        return max(self.window_sizes) + self.seq_len

    # -- fit ----------------------------------------------------------------------------------
    def fit(self, reference: list[pd.DataFrame]) -> "MSCVAEDetector":
        reference = [r for r in reference if len(r) > 2 * self.warmup]
        train, val = [], []
        for r in reference:
            r, _ = _impute(r)
            cut = int(len(r) * (1 - self.val_fraction))
            train.append(r.iloc[:cut])
            val.append(r.iloc[max(0, cut - self.warmup) :])  # val keeps real warm-up history
        self.variables_ = list(reference[0].columns)
        self.meta_ = variable_table(self.variables_)

        self.model_ = MSCVAE(
            n_features=len(self.variables_), latent_dim=self.latent_dim, window_sizes=self.window_sizes,
            stride=self.stride, seq_len=self.seq_len, device=self.device, seed=self.seed,
        )
        self.model_.fit(train, epochs=self.epochs, batch_size=self.batch_size, lr=self.lr, verbose=True)
        self.threshold_in_sample_ = float(self.model_.threshold)
        self.model_.generator.step = 1  # score every timestamp from here on

        scores, mats, vals = [], [], []
        for v in val:
            m, x = self.model_.generator.generate(v)
            s = self.model_._get_anomaly_scores(m, x, batch_size=self.batch_size)
            scores.append(s[self.warmup :])
            mats.append(m)
            vals.append(x)
        self.val_scores_ = np.concatenate(scores)
        self.threshold_ = spot_limit(self.val_scores_, self.q, self.level)
        self.model_.threshold = self.threshold_
        # RCA baseline on held-out normal data (author's routine, our data)
        self.model_._calibrate_variable_baseline(torch.cat(mats), torch.cat(vals), batch_size=self.batch_size)
        log.info("MSCVAE thresholds: in-sample %.4g | out-of-sample %.4g", self.threshold_in_sample_, self.threshold_)
        return self

    # -- score --------------------------------------------------------------------------------
    def _phi(self, df: pd.DataFrame) -> np.ndarray:
        m, x = self.model_.generator.generate(df[self.variables_])
        return self.model_._get_anomaly_scores(m, x, batch_size=self.batch_size)

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        X, self.n_imputed_ = _impute(df[self.variables_])
        phi = np.full(len(X), np.nan)
        w = self.warmup
        for start in range(0, len(X), self.chunk):
            lo = max(0, start - w)
            part = self._phi(X.iloc[lo : start + self.chunk])
            phi[start : start + self.chunk] = part[start - lo :]
        phi[:w] = np.nan  # padded windows at the very start
        missing = df[self.variables_].isna().any(axis=1).to_numpy()
        out = pd.DataFrame({"phi": phi, "phi_lim": self.threshold_}, index=df.index)
        out.loc[missing, "phi"] = np.nan
        out["phi_lim_in_sample"] = self.threshold_in_sample_
        return finish_scores(out, ["phi"])

    # -- contributions ------------------------------------------------------------------------
    def contributions(self, df: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
        """Author's RCA (diag/off-diag/value channels vs. normal baseline) over the masked rows."""
        idx = np.flatnonzero(mask)
        lo = max(0, idx[0] - self.warmup)
        hi = idx[-1] + 1
        X, _ = _impute(df[self.variables_].iloc[lo:hi])
        res = self.model_._variable_contribution_scores(
            X, batch_size=self.batch_size, anomaly_windows=idx - lo
        )
        return contribution_frame(list(res["variable_names"]), res["contrib"])
