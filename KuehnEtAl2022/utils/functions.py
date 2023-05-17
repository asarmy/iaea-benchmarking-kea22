"""This file contains generic tools that could be used in multiple models."""

# Import python libraries
import numpy as np
from pathlib import Path
from typing import Union


# Define user functions
def calc_percentile_lognormal(ln_mu:  Union[float, np.ndarray], ln_sigma:  Union[float, np.ndarray], percentile: float) ->  Union[float, np.ndarray]:
    """
    Calculate the value of a log-normal distribution at a given percentile.

    Parameters
    ----------
    ln_mu : float
        The mean of the natural logarithm of the distribution (location parameter).
    ln_sigma : float
        The standard deviation of the natural logarithm of the distribution (scale parameter).
    percentile : float
        Percentile value. Use -1 for mean.

    Returns
    -------
    float
        The value of the random variable at which the cumulative distribution equals the
        given percentile.
    """
    
    if percentile == -1:
        return ln_mu + ln_sigma**2 / 2
    else:
        return stats.norm.ppf(percentile, loc=ln_mu, scale=ln_sigma)
        
        
def calc_probexceed_lognormal(ln_displ_array:  Union[float, np.ndarray], ln_mu:float, ln_sigma:float) -> np.ndarray:
    """
    Calculate the complementary cumulative distribution for a log-normal distribution.

    Parameters
    ----------
    ln displ_array : np.ndarray
        Array of the natural log of displacment test values in meters.
    ln_mu : float
        The mean of the natural logarithm of the distribution (location parameter).
    ln_sigma : float
        The standard deviation of the natural logarithm of the distribution (scale parameter).

    Returns
    -------
    np.ndarray
        The complementary cumulative distribution.
    """

    return 1 - stats.norm.cdf(x=ln_displ_array, loc=ln_mu, scale=ln_sigma)