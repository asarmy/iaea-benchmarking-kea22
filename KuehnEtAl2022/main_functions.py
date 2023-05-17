"""This file contains user functions to calcualte agggregated fault displacement using 
the Kuehn et al. (2022) model.
"""

# Import python libraries
from functools import partial
import numpy as np
from scipy import stats
from typing import Tuple, Union

# Import model helper functions
from model import helper_functions as helper

# Import warning statement
from model import warning_statement


# Define user functions
def calc_distrib_params(
    *,
    magnitude: float,
    location: float,
    style: str,
    mean_model: bool = True,
) -> Tuple[np.array, np.array, np.array, np.array, np.array]:
    """
    Calculate median and sigma values for KEA22 on magnitude, rupture location, and style.
    Note all returns are in natural log units.
    Note returned values are asymmetrical (i.e., not folded).

    Parameters
    ----------
    magnitude : float
        Earthquake moment magnitude.
    location : float
        Normalized location along rupture length, range [0, 1.0].
    style : str
        Style of faulting, case insensitive.
        Valid options are "strike-slip", "reverse", or "normal".
    mean_model : bool, optional
        If True, use mean coefficients and adjustments.
        If False, use full (n=2000)coefficients and adjustments.
        Default True.

    Returns
    -------
    Tuple[np.array, np.array, np.array, np.array, np.array]
        med : Median prediction in natural log units.
        sd_total : Total standard deviation in natural log units.
        sd_mode : Mode standard deviation in natural log units.
        sd_c : Event term standard deviation in natural log units.
        sd_u : Location standard deviation in natural log units.
    """

    f = helper._calc_distrib_params
    return f(magnitude=magnitude, location=location, style=style, mean_model=mean_model)



# ans = calc_distrib_params(magnitude=7, location=0.5, style="reverse", mean_model=False)
# print(ans)
