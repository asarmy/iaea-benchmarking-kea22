"""This file contains constant input arguments that could be used in multiple models."""

import numpy as np


# Default constants
DEFAULT_LOCATIONS = np.arange(0, 1.01, 0.01)
DEFAULT_DISPLACEMENTS = np.logspace(
    start=np.log(0.001), stop=np.log(130), num=200, base=np.e
)
