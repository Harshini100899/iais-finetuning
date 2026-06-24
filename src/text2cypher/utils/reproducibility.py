"""Helpers to enforce deterministic run execution via seeding."""

import random

import numpy as np
import torch


def set_seeds(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Make CuDNN deterministic (no-op on CPU but harmless)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
