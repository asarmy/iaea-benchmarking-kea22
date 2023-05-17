"""This file contains helper functions to calcualte agggregated fault displacement using 
the Kuehn et al. (2022) model.
"""

# Import python libraries
from functools import partial
import numpy as np
from typing import Tuple, Union

# Import model coefficients and look-up tables
from utils.import_data import POSTERIOR, MED_ADJ, SIG_ADJ

# Import package functions
from utils.functions import calc_percentile_lognormal, calc_probexceed_lognormal

# Import package modules
import model.model_functions as model


# Define data processing functions
def _calc_data_means(record_array: np.recarray) -> np.recarray:
    """
    Calculates the mean values of each column in the input record array.

    Parameters
    ----------
    record_array : np.recarray
        The input coefficient table or look-up table.

    Returns
    -------
    np.recarray
        The record array containing the mean values of each column.

    """

    return (
        np.array([np.mean(record_array[col]) for col in record_array.dtype.names])
        .view(dtype=[(col, float) for col in record_array.dtype.names])
        .view(np.recarray)
    )


# Define helper functions
def _calc_distrib_params(
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

    # Get appropriate coefficients and event term adjustments
    style = style.lower()
    func_get_data = _calc_data_means
    if mean_model:
        coefficients = func_get_data(POSTERIOR[style])
        median_adjust = (
            func_get_data(MED_ADJ[style]) if MED_ADJ[style] is not None else None
        )
        sigma_adjust = (
            func_get_data(SIG_ADJ[style]) if SIG_ADJ[style] is not None else None
        )
    else:
        coefficients = POSTERIOR[style]
        median_adjust = MED_ADJ[style]
        sigma_adjust = SIG_ADJ[style]

    # Get appropriate model
    if style == "reverse":
        func_model = model._func_rev
    elif style == "normal":
        func_model = model._func_nm
    elif style == "strike-slip":
        func_model = model._func_ss
    else:
        raise AssertionError(f"Invalid style of faulting.")

    return func_model(
        coefficients=coefficients,
        median_adjust=median_adjust,
        sigma_adjust=sigma_adjust,
        magnitude=magnitude,
        location=location,
    )


def _calc_ln_displ_no_epistemic(
    *,
    magnitude: float,
    location: float,
    style: str,
    percentile: float,
) -> float:
    """
    Calculate the natural log of displacement based on the mean model (i.e., does not
    include any within-model epistemic uncertainty).

    Parameters
    ----------
    magnitude : float
        Earthquake moment magnitude.
    location : float
        Normalized location along rupture length, range [0, 1.0].
    style : str
        Style of faulting, case insensitive.
        Valid options are "strike-slip", "reverse", or "normal".
    percentile : float, optional
        Percentile value. The default value is 0.5 (median). Use -1 for mean.

    Returns
    -------
    float
        Natural log of displacement, where displacement is in meters.
    """

    # Calculate distribution parameters
    params = _calc_distrib_params(
        magnitude=magnitude, location=location, style=style, mean_model=True
    )
    mu, sigma = params[0], params[1]

    # Calculate ln displacement
    ln_displ = calc_percentile_lognormal(mu, sigma, percentile)
    return np.squeeze(ln_displ)


def _calc_ln_displ_full_epistemic(
    *,
    magnitude: float,
    location: float,
    style: str,
    percentile: float,
) -> float:
    """
    Calculate the natural log of displacement for the full model (i.e., includes full
    within-model epistemic uncertainty), which produces 2000 predictions.

    Parameters
    ----------
    magnitude : float
        Earthquake moment magnitude.
    location : float
        Normalized location along rupture length, range [0, 1.0].
    style : str
        Style of faulting, case insensitive.
        Valid options are "strike-slip", "reverse", or "normal".
    percentile : float, optional
        Percentile value. The default value is 0.5 (median). Use -1 for mean.

    Returns
    -------
    float
        Natural log of displacement, where displacement is in meters.
    """

    # Calculate distribution parameters
    params = _calc_distrib_params(
        magnitude=magnitude, location=location, style=style, mean_model=False
    )
    mu, sigma = params[0], params[1]

    # Calculate ln displacement
    return calc_percentile_lognormal(mu, sigma, percentile)

