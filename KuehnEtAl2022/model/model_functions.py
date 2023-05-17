"""This file contains source functions to calcualte agggregated fault displacement using 
the Kuehn et al. (2022) model.

Reference: https://doi.org/10.34948/N3X59H
"""

# Import python libraries
from typing import Tuple
import numpy as np
from scipy.interpolate import interp1d

# Import package utilities
from utils.utilities import recarray_to_array

# Model constants
MAG_BREAK, DELTA = 7.0, 0.1

# Define source functions
def _func_mode(coefficients: np.recarray, *, magnitude: float) -> float:
    """
    Calculate magnitude scaling.

    Parameters
    ----------
    coefficients : np.recarray
        Numpy recarray containing model coefficients.
    magnitude : float
        Earthquake moment magnitude.

    Returns
    -------
    float
        Mode in natural log units.
    """
    fm = (
        coefficients["c1"]
        + coefficients["c2"] * (magnitude - MAG_BREAK)
        + (coefficients["c3"] - coefficients["c2"])
        * DELTA
        * np.log(1 + np.exp((magnitude - MAG_BREAK) / DELTA))
    )
    return fm


def _func_mu(coefficients: np.recarray, *, magnitude: float, location: float) -> float:
    """
    Calculate location scaling.

    Parameters
    ----------
    coefficients : np.recarray
        Numpy recarray containing model coefficients.
    magnitude : float
        Earthquake moment magnitude.
    location : float
        Normalized location along rupture length, range [0, 1.0].

    Returns
    -------
    float
        Mu in natural log units.
    """
    fm = _func_mode(coefficients, magnitude=magnitude)

    # Column name for shape coefficient "c" varies for style of faulting, fix that here
    c = (
        coefficients["mu_c"]
        if "mu_c" in coefficients.dtype.names
        else coefficients["c"]
    )

    alpha = coefficients["alpha"]
    beta = coefficients["beta"]
    a = fm - c * np.power(alpha / (alpha + beta), alpha) * np.power(
        beta / (alpha + beta), beta
    )
    mu = a + c * (location**alpha) * ((1 - location) ** beta)
    return mu


def _func_sd_mode_sigmoid(coefficients: np.recarray, *, magnitude: float) -> float:
    """
    Calculate standard deviation of the mode.

    Parameters
    ----------
    coefficients : np.recarray
        Numpy recarray containing model coefficients.
    magnitude : float
        Earthquake moment magnitude.

    Returns
    -------
    float
        Standard deviation in natural log units.
    """
    sd = coefficients["s_m_n1"] - coefficients["s_m_n2"] / (
        1 + np.exp(-1 * coefficients["s_m_n3"] * (magnitude - MAG_BREAK))
    )
    return sd


def _func_sd_mode_bilinear(coefficients: np.recarray, *, magnitude: float) -> float:
    """
    Calculate standard deviation of the mode.

    Parameters
    ----------
    coefficients : np.recarray
        Numpy recarray containing model coefficients.
    magnitude : float
        Earthquake moment magnitude.

    Returns
    -------
    float
        Standard deviation in natural log units.
    """
    sd = (
        coefficients["s_m_s1"]
        + coefficients["s_m_s2"] * (magnitude - coefficients["s_m_s3"])
        - coefficients["s_m_s2"]
        * DELTA
        * np.log(1 + np.exp((magnitude - coefficients["s_m_s3"]) / DELTA))
    )
    return sd


def _func_sd_u(coefficients: np.recarray, *, location: float) -> float:
    """
    Calculate standard deviation of the location.

    Parameters
    ----------
    coefficients : np.recarray
        Numpy recarray containing model coefficients.
    location : float
        Normalized location along rupture length, range [0, 1.0].

    Returns
    -------
    float
        Standard deviation in natural log units.
    """

    # Column name2 for stdv coefficients "s_" varies for style of faulting, fix that here
    s_s1 = (
        coefficients["s_s1"]
        if "s_s1" in coefficients.dtype.names
        else coefficients["s_r1"]
    )
    s_s2 = (
        coefficients["s_s2"]
        if "s_s2" in coefficients.dtype.names
        else coefficients["s_r2"]
    )

    alpha = coefficients["alpha"]
    beta = coefficients["beta"]

    sd = s_s1 + s_s2 * (location - alpha / (alpha + beta)) ** 2
    return sd


def _interpolate_from_location_table(
    data_table: np.recarray, *, location: float
) -> np.array:
    """
    Interpolate event term values from location table.

    Parameters
    ----------
    data_table : np.recarray
        Numpy recarray of model look-up table for each U_C location.
    location : float
        Normalized location along rupture length, range [0, 1.0].

    Returns
    -------
    np.array
        Numpy array of look-up table values for location.
    """

    # Convert recarray column names to location values
    columns = data_table.dtype.names
    vals = [float(col[2:].replace("pt", ".")) for col in columns]
    U_C = np.array(vals, dtype=float)

    # Linearly interpolate
    data_table = recarray_to_array(data_table)
    f1 = interp1d(U_C, data_table, kind="linear", axis=1)
    return f1(location).flatten()


def _func_rev(
    *,
    coefficients: np.recarray,
    median_adjust: np.recarray,
    sigma_adjust: np.recarray,
    magnitude: float,
    location: float,
) -> Tuple[np.array, np.array, np.array, np.array, np.array]:
    """
    Calculate median prediction (in natural log units) and standard deviations
    (in natural log units) for reverse style of faulting.

    Parameters
    ----------
    coefficients : np.recarray
        Numpy recarray containing model coefficients.
    median_adjust : np.recarray
        Numpy recarray containing event term median adjustment (delta_c) look-up.
    sigma_adjust : np.recarray
        Numpy recarray containing event term sigma adjustment (sigma_c) look-up.
    magnitude : float
        Earthquake moment magnitude.
    location : float
        Normalized location along rupture length, range [0, 1.0].

    Returns
    -------
    Tuple[np.array, np.array, np.array, np.array, np.array]
        med : Median prediction in natural log units.
        sd_total : Total standard deviation in natural log units.
        sd_mode : Mode standard deviation in natural log units.
        sd_c : Event term standard deviation in natural log units.
        sd_u : Location standard deviation in natural log units.
    Raises
    ------
    ValueError
        If the magnitude or location is not within the given bounds.
    """

    # Calculate median prediction
    mu = _func_mu(coefficients, magnitude=magnitude, location=location)
    delta_c = _interpolate_from_location_table(median_adjust, location=location)
    med = mu + delta_c

    # Calculate standard deviations
    sd_mode = coefficients["s_m_r"]
    sd_c = _interpolate_from_location_table(sigma_adjust, location=location)
    sd_u = _func_sd_u(coefficients, location=location)
    sd_total = np.sqrt(sd_mode**2 + sd_u**2 + sd_c**2)

    return med, sd_total, sd_mode, sd_c, sd_u


def _func_nm(
    *,
    coefficients: np.recarray,
    median_adjust: np.recarray,
    sigma_adjust: np.recarray,
    magnitude: float,
    location: float,
) -> Tuple[np.array, np.array, np.array, np.array, np.array]:
    """
    Calculate median prediction (in natural log units) and standard deviations
    (in natural log units) for normal style of faulting.

    Parameters
    ----------
    coefficients : np.recarray
        Numpy recarray containing model coefficients.
    median_adjust : np.recarray
        Numpy recarray containing event term median adjustment (delta_c) look-up.
    sigma_adjust : np.recarray
        Numpy recarray containing event term sigma adjustment (sigma_c) look-up.
    magnitude : float
        Earthquake moment magnitude.
    location : float
        Normalized location along rupture length, range [0, 1.0].

    Returns
    -------
    Tuple[np.array, np.array, np.array, np.array, np.array]
        med : Median prediction in natural log units.
        sd_total : Total standard deviation in natural log units.
        sd_mode : Mode standard deviation in natural log units.
        sd_c : Event term standard deviation in natural log units.
        sd_u : Location standard deviation in natural log units.
    Raises
    ------
    ValueError
        If the magnitude or location is not within the given bounds.
    """

    # Calculate median prediction
    mu = _func_mu(coefficients, magnitude=magnitude, location=location)
    delta_c = _interpolate_from_location_table(median_adjust, location=location)
    med = mu + delta_c

    # Calculate standard deviations
    sd_mode = _func_sd_mode_sigmoid(coefficients, magnitude=magnitude)
    sd_c = _interpolate_from_location_table(sigma_adjust, location=location)
    sd_u = coefficients["sigma"]
    sd_total = np.sqrt(sd_mode**2 + sd_u**2 + sd_c**2)

    return med, sd_total, sd_mode, sd_c, sd_u


def _func_ss(
    *,
    coefficients: np.recarray,
    median_adjust: np.recarray = None,
    sigma_adjust: np.recarray = None,
    magnitude: float,
    location: float,
) -> Tuple[np.array, np.array, np.array, np.array, np.array]:
    """
    Calculate median prediction (in natural log units) and standard deviations
    (in natural log units) for strike-slip style of faulting.

    Parameters
    ----------
    coefficients : np.recarray
        Numpy recarray containing model coefficients.
    median_adjust : np.recarray
        Numpy recarray containing event term median adjustment (delta_c) look-up.
        Not applicable for strike-slip.
    sigma_adjust : np.recarray
        Numpy recarray containing event term sigma adjustment (sigma_c) look-up.
        Not applicable for strike-slip.
    magnitude : float
        Earthquake moment magnitude.
    location : float
        Normalized location along rupture length, range [0, 1.0].

    Returns
    -------
    Tuple[np.array, np.array, np.array, np.array, np.array]
        med : Median prediction in natural log units.
        sd_total : Total standard deviation in natural log units.
        sd_mode : Mode standard deviation in natural log units.
        sd_c : Event term standard deviation in natural log units.
        sd_u : Location standard deviation in natural log units.
    Raises
    ------
    ValueError
        If the magnitude or location is not within the given bounds.
    """

    # Calculate median prediction
    med = _func_mu(coefficients, magnitude=magnitude, location=location)

    # Calculate standard deviations
    sd_mode = _func_sd_mode_bilinear(coefficients, magnitude=magnitude)
    sd_c = np.zeros_like(med, dtype=float)
    sd_u = _func_sd_u(coefficients, location=location)
    sd_total = np.sqrt(sd_mode**2 + sd_u**2 + sd_c**2)

    return med, sd_total, sd_mode, sd_c, sd_u
