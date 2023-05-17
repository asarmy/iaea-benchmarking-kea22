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

# Set root directory for Excel files output
ROOT_OUT = PWD.parent / "results"

# Define directory with case info blocks to put as notes tab in Excel files
DIR_INFO = PWD / "info"

# Set configurations for writing to Excel files
OPTIONS = {}
OPTIONS["strings_to_formulas"] = False
OPTIONS["strings_to_urls"] = False
OPTIONS = {"options": OPTIONS}
