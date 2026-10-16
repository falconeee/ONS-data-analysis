"""Shared utilities: project paths, logger, seeding and the base class of the experiment configs."""

import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict

# -- paths ------------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESOURCES_DIR = PROJECT_ROOT / "resources"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# -- logging ----------------------------------------------------------------------------------
logger = logging.getLogger("ons_energy")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Child logger of the package logger, e.g. ``get_logger(__name__)``."""
    return logger.getChild(name.removeprefix("ons_energy."))


# -- seeding ----------------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and — only if already imported — PyTorch / Lightning.

    torch is never imported here: LightGBM processes must stay torch-free (libomp clash on macOS,
    see ``ons_energy.__init__``). LightGBM takes the seed through its config.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    if "torch" not in sys.modules:
        return
    import torch

    torch.manual_seed(seed)  # also seeds CUDA and MPS generators
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if "lightning" in sys.modules:
        import lightning

        lightning.seed_everything(seed, workers=True, verbose=False)


# -- configs ----------------------------------------------------------------------------------
class StrictModel(BaseModel):
    """Base for experiment configs (built in the notebooks): unknown keys are errors."""

    model_config = ConfigDict(extra="forbid", frozen=True)
