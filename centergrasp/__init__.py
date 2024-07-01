import random
import torch
import numpy as np


def set_seeds(seed=0):
    """Sets all seeds."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    return
