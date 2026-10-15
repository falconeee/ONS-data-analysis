"""Moving-window (recursive) PCA with a guard against learning from faulty data.

The model is refitted every ``update_every`` steps on the last ``window`` observations, excluding
points in alarm and the ``guard`` steps after them. This tracks slow structural change (load growth,
wind/solar expansion) at the cost of possibly absorbing slow faults (a drought that builds over
months) — the central trade-off discussed in the report.

Excluding alarmed points alone would freeze the model forever after a permanent step change (e.g.
the 2023-04 MMGD methodology break): every new point alarms, so none is ever learned. An alarm that
lasts longer than ``accept_after`` consecutive steps is therefore accepted as a new regime.
"""

import numpy as np
import pandas as pd

from ons_energy.anomaly.base import Detector, finish_scores
from ons_energy.anomaly.pca import PCAMonitor


class MovingWindowPCA(Detector):
    def __init__(
        self,
        window: int,
        update_every: int,
        guard: int = 0,
        accept_after: int | None = None,
        exclude_ratio: float = 1.0,
        min_fit: int | None = None,
        name: str = "RPCA",
        **pca_kw,
    ):
        self.window = window
        self.update_every = update_every
        self.guard = guard
        self.accept_after = accept_after
        # only points with stat/limit > exclude_ratio are kept out of the window: excluding every
        # marginal alarm shrinks the window variance and tightens the next limits (feedback spiral)
        self.exclude_ratio = exclude_ratio
        self.min_fit = min_fit or window // 2
        self.pca_kw = pca_kw
        self.name = name

    def fit(self, reference: list[pd.DataFrame]) -> "MovingWindowPCA":
        self.model_ = PCAMonitor(**self.pca_kw).fit(reference)
        self.initial_ = self.model_
        self.history_ = pd.concat(reference).dropna()
        return self

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        model = self.initial_
        hist = self.history_
        outs, self.models_ = [], [(df.index[0], model)]
        run = 0  # length of the current alarm run (carried across blocks)
        for start in range(0, len(df), self.update_every):
            block = df.iloc[start : start + self.update_every]
            sc = model.score(block)
            outs.append(sc)
            alarm = (sc["ratio"] > self.exclude_ratio).fillna(False).to_numpy()
            runs = np.empty(len(alarm), int)
            for i, a in enumerate(alarm):
                run = run + 1 if a else 0
                runs[i] = run
            # guard: drop alarmed rows and the `guard` rows after each alarm before updating
            bad = alarm.copy()
            if self.guard and bad.any():
                bad = np.convolve(bad.astype(int), np.ones(self.guard + 1, int))[: len(bad)] > 0
            if self.accept_after is not None:
                bad &= ~(runs > self.accept_after)  # persistent shift -> accept as new regime
            clean = block.loc[~bad].dropna()
            hist = pd.concat([hist, clean]).iloc[-self.window :]
            if len(hist) >= self.min_fit:
                model = PCAMonitor(**self.pca_kw).fit([hist])
                self.models_.append((block.index[-1], model))
        out = pd.concat(outs)
        return finish_scores(out[["T2", "SPE", "T2_lim", "SPE_lim"]].copy(), ["T2", "SPE"])

    def _model_at(self, ts: pd.Timestamp) -> PCAMonitor:
        chosen = self.initial_
        for t, m in self.models_:
            if t < ts:
                chosen = m
        return chosen

    def contributions(self, df: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
        first = df.index[np.asarray(mask)][0]
        return self._model_at(first).contributions(df, mask)
