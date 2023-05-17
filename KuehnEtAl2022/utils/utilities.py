"""This file contains generic tools that could be used in multiple models."""

# Import python libraries
import numpy as np
from pathlib import Path
from typing import Union


# Define user functions
def read_csv_to_recarray(filepath: Path) -> np.recarray:
    """
    Read CSV data from file and return as a NumPy recarray.

    Parameters
    ----------
    filepath : pathlib.Path
        Filepath of the CSV file to be read.

    Returns
    -------
    np.recarray
        NumPy recarray of the CSV data.
    """
    data = np.genfromtxt(filepath, delimiter=",", names=True, encoding="UTF-8-sig")
    return data.view(np.recarray)


def recarray_to_array(record_array: np.ndarray) -> np.array:
    """
    Covert a NumPy recarray to a NumPy array.

    Parameters
    ----------
    record_array : np.recarray
        NumPy recarray.

    Returns
    -------
    np.array
        NumPy array.
    """
    return record_array.view((float, len(record_array.dtype.names)))


def check_numeric(variable: Union[float, int]) -> None:
    """
    Check if a variable is a float or integer.

    Parameters
    ----------
    variable : Union[float, int]
        The variable to check.

    Raises
    ------
    TypeError
        If the variable is not a float or integer.
    """
    if not isinstance(variable, (float, int)):
        raise TypeError("Expected float or int, got {0}.".format(type(variable).__name__))

