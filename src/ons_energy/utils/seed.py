"""Global seeding for reproducible experiments."""

import os
import random
import sys

import numpy as np


def set_seed(seed: int = 42, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy and — only if already imported — PyTorch / Lightning.

    torch is never imported here: LightGBM processes must stay torch-free (libomp clash on macOS,
    see ``ons_energy.__init__``). LightGBM and scikit-learn take the seed through their configs.
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
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if "lightning" in sys.modules:
        import lightning

        lightning.seed_everything(seed, workers=True, verbose=False)
