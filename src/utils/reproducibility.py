"""
Reproducibility Utilities
=========================
Functions to ensure reproducible experiments.
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Set random seeds for reproducibility.
    
    Parameters
    ----------
    seed : int
        Random seed.
    deterministic : bool
        Whether to use deterministic algorithms.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
