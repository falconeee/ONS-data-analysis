"""Backtest configuration (built in the forecasting notebooks)."""

from datetime import date
from typing import Any, Literal

from pydantic import Field

from ons_energy.config import StrictModel

ModelKind = Literal["seasonal_naive", "lgbm", "nhits", "nbeats_i", "nbeatsx", "tft", "ons_programmed"]


class ModelSpec(StrictModel):
    name: str
    kind: ModelKind
    params: dict[str, Any] = Field(default_factory=dict)


class BacktestConfig(StrictModel):
    name: str
    subsystems: list[str] = ["SE"]
    train_start: date = date(2015, 1, 1)
    # Evaluation window. `validation` mode evaluates [val_start, test_start) instead (tuning only).
    mode: Literal["test", "validation"] = "test"
    val_start: date = date(2023, 5, 1)
    test_start: date = date(2024, 1, 1)
    test_end: date | None = None  # default: last full day in the data
    # Day-ahead setup: issued at D-1 `origin_hour`:00 with data up to the previous hour; the
    # evaluated steps are the 24 hours of D.
    origin_hour: int = 12
    horizon: int = 36
    eval_steps: tuple[int, int] = (13, 36)
    calib_days: int = 90  # extra windows before the evaluation start (rolling-conformal warm-up)
    refit_every_days: int = 30
    weather: Literal["none", "perfect", "honest"] = "none"
    quantiles: list[float] = [0.05, 0.1, 0.5, 0.9, 0.95]
    conformal_window_days: int = 90
    accelerator: Literal["mps", "cpu"] = "mps"
    seed: int = 42
    models: list[ModelSpec]

    @property
    def eval_start(self) -> date:
        return self.val_start if self.mode == "validation" else self.test_start

    def eval_end(self, last_day: date) -> date:
        if self.mode == "validation":
            return date.fromordinal(self.test_start.toordinal() - 1)
        return min(self.test_end or last_day, last_day)
