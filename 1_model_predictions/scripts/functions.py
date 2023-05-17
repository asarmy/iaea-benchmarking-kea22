# Import python libraries
import numpy as np
import pandas as pd

# Import configurations
from model_config import *

# Import package functions
import main_functions as KEA22


def calc_params(row: pd.Series, style: str, mean_model_flag: bool) -> tuple:
    """
    Calculate lognormal distribution parameters based on row values.

    Parameters
    ----------
    row : pd.Series
        A pandas Series representing a row of a DataFrame.
    style : str
        Style of faulting.
    mean_model_flag : bool
        Flag indicating whether to use the mean model (True) or full model with
        2000 runs (False).

    Returns
    -------
    tuple
        A tuple of the calculated median and total sigma, both in natural log units.

    """

    # Define function variable and apply function to each row of dataframe
    f = KEA22.calc_distrib_params
    result = f(
        magnitude=row["magnitude"],
        location=row["u_star"],
        style=style,
        mean_model=mean_model_flag,
    )

    # Return the first two outputs of the function (which are median and total sigma)
    return tuple(np.squeeze(arr) for arr in result[:2])


def calc_model_predictions(
    dataframe: pd.DataFrame, style: str, mean_model_flag: bool
) -> pd.DataFrame:
    """
    Calculate model predictions for the input DataFrame. The column names are specific
    to this project and assumed to be correct.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input DataFrame containing the data.
    style : str
        Style of faulting.
    mean_model_flag : bool
        Flag indicating whether to use the mean model (True) or full model with
        2000 runs (False).

    Returns
    -------
    pd.DataFrame
        The DataFrame containing the input data and model predictions.

    """

    # Calculuate mu, sigma for each row and number of model runs
    dataframe = dataframe.copy()
    mu, sig = "ln_median", "sigma_total"
    dataframe[[mu, sig]] = dataframe.apply(
        lambda row: pd.Series(calc_params(row, style, mean_model_flag)), axis=1
    )

    # Additional processing based on number of model runs and weights
    if mean_model_flag:
        dataframe["MODEL_ID"] = 1
        dataframe["fdm_wt"] = 1
    else:
        # Enumerate MODEL_IDs
        dataframe["MODEL_ID"] = dataframe[mu].apply(
            lambda x: list(range(1, len(x) + 1))
        )
        # Calcualte weights for each MODEL_ID
        n_runs = dataframe[mu].apply(lambda x: len(x))
        dataframe["fdm_wt"] = pd.Series([[1 / i] * i for i in n_runs])
        dataframe = dataframe.explode(
            [mu, sig, "MODEL_ID", "fdm_wt"], ignore_index=True
        )

    return dataframe
