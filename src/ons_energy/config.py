"""Base class for the experiment configs (``BacktestConfig``, ``MSPCConfig``, data catalog).

Experiments are defined in the notebooks as validated Python objects, e.g.
``BacktestConfig(name="seco_tft", models=[ModelSpec(name="TFT", kind="tft")])``.
"""

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Base for config models: unknown keys are errors (typos fail loudly)."""

    model_config = ConfigDict(extra="forbid", frozen=True)
