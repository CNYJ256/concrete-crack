"""Utility functions: seeding, directory creation."""
import os
import random

import numpy as np
import torch

from src.config import CKPT_DIR, RESULT_DIR, FIGURE_DIR, PERSONAL_VAL_DIR, LOG_DIR, SEED


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dirs():
    for d in [CKPT_DIR, RESULT_DIR, FIGURE_DIR, PERSONAL_VAL_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)
