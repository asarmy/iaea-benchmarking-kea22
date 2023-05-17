"""This file contains setup functions to import the data (coefficients and event term 
look-up tables) used to calcualte agggregated fault displacement in the Kuehn et al. 
(2022) model.

Reference: https://doi.org/10.34948/N3X59H
"""

# Import python libraries
import numpy as np
from pathlib import Path

# Import package utilities
from utils.utilities import read_csv_to_recarray

# Set filepath for model data
DIR_DATA = Path(__file__).parents[1] / "data"

# Import model coefficients
POSTERIOR_SS = read_csv_to_recarray(
    DIR_DATA / "KEA22_coefficients_posterior_SS.csv"
)
POSTERIOR_RV = read_csv_to_recarray(
    DIR_DATA / "KEA22_coefficients_posterior_REV.csv"
)
POSTERIOR_NM = read_csv_to_recarray(
    DIR_DATA / "KEA22_coefficients_posterior_NM.csv"
)

# Import median adjustments
MEDIAN_ADJ_RV = read_csv_to_recarray(DIR_DATA / "KEA22_delta_med_c_reverse.csv")
MEDIAN_ADJ_NM = read_csv_to_recarray(DIR_DATA / "KEA22_delta_med_c_normal.csv")

# Import sigma adjustments
SIGMA_ADJ_RV = read_csv_to_recarray(DIR_DATA / "KEA22_sigma_c_reverse.csv")
SIGMA_ADJ_NM = read_csv_to_recarray(DIR_DATA / "KEA22_sigma_c_normal.csv")

# Import uncertainty standard deviations
# UNC_SD_SS = read_csv_to_recarray(DIR_DATA / "KEA22_unc_epistemic_strike-slip.csv")
# UNC_SD_RV = read_csv_to_recarray(DIR_DATA / "KEA22_unc_epistemic_reverse.csv")
# UNC_SD_NM = read_csv_to_recarray(DIR_DATA / "KEA22_unc_epistemic_normal.csv")

### Create style-data dictionaries
POSTERIOR = {
    "strike-slip": POSTERIOR_SS,
    "reverse": POSTERIOR_RV,
    "normal": POSTERIOR_NM,
}
MED_ADJ = {"strike-slip": None, "reverse": MEDIAN_ADJ_RV, "normal": MEDIAN_ADJ_NM}
SIG_ADJ = {"strike-slip": None, "reverse": SIGMA_ADJ_RV, "normal": SIGMA_ADJ_NM}
# UNC_SD = {"strike-slip": UNC_SD_SS, "reverse": UNC_SD_RV, "normal": UNC_SD_NM}
