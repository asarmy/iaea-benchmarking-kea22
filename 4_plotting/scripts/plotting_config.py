import pandas as pd
import sys
from pathlib import Path

# Set console output formatting
pd.set_option("display.max_columns", 800)
pd.set_option("display.width", 800)

# Define scripts working directory
PWD = Path(sys.argv[0]).absolute().parent

# Set root directory for fractile results
ROOT_RES = Path(__file__).parents[2] / "3_fractile_calcs" / "results"

# Set root directory for fractiles output
ROOT_OUT = PWD.parent / "figures"

# Define directory with case info blocks to put on plots
DIR_INFO = PWD / "info"

# Define plotting dictionaries
# FIXME: left is assumed to be U* and right is assumed to be 1-U*; fix wording
LS_DICT = {
    "contribs": "solid",
    "left": (0, (3, 3)),
    "right": (0, (6, 2)),
    "folded": "solid",
    "Mean": "solid",
    "0.5": "dashdot",
    "0.16": (0, (3, 3)),
    "0.84": (0, (3, 3)),
    "0.05": "dotted",
    "0.95": "dotted",
}
LC_DICT = {
    "left": "tab:blue",
    "right": "tab:orange",
    "folded": "black",
    "contribs": "gray",
}
LW_DICT = {
    "contribs": 0.2,
    "folded": 0.8,
    "left": 0.8,
    "right": 0.8,
    "Mean": 0.8,
    "0.5": 0.75,
    "0.16": 0.75,
    "0.84": 0.75,
    "0.05": 0.9,
    "0.95": 0.9,
}

# Define axis limits
LIMS_DICT = {
    "norcia_case1": {"x": [0.01, 10], "y": [1e-6, 1e-3]},
    "le_teil_case2": {"x": [0.01, 10], "y": [1e-7, 1e-4]},
    "le_teil_extra": {"x": [0.01, 10], "y": [1e-8, 1e-3]},
    "kumamoto_case3": {"x": [0.01, 10], "y": [1e-8, 1e-3]},
    "kumamoto_case2": {"x": [0.01, 10], "y": [1e-8, 1e-3]},
}

# Define Kumamoto Sensitivity #2 dictionaries for source contribution plots
KM_ID_DICT = {
    "Float": "Float (Unwtd. Branch)",
    "F2": "F2 (Unwtd. Branch)",
    "F1_F2": "F1+F2",
    "F2_F3": "F2+F3",
    "Full": "F1+F2+F3",
    "Combined_F2_and_Float": "Combined F2/Float Branch",
    "Total": "Total",
}

KM_LC_DICT = {
    "Float": "tab:purple",
    "F2": "tab:green",
    "F1_F2": "k",
    "F2_F3": "k",
    "Full": "k",
    "Combined_F2_and_Float": "gray",
    "Total": "tab:red",
}

KM_LS_DICT = {
    "Float": "solid",
    "F2": "solid",
    "F1_F2": "dotted",
    "F2_F3": (0, (9, 3)),
    "Full": "dashdot",
    "Combined_F2_and_Float": "solid",
    "Total": "solid",
}

KM_LW_DICT = {
    "Float": 0.5,
    "F2": 0.5,
    "F1_F2": 0.9,
    "F2_F3": 0.75,
    "Full": 0.75,
    "Combined_F2_and_Float": 0.75,
    "Total": 0.9,
}
