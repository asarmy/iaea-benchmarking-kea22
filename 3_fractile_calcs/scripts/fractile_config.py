import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Set console output formatting
pd.set_option("display.max_columns", 800)
pd.set_option("display.width", 800)

# Define scripts working directory
PWD = Path(sys.argv[0]).absolute().parent

# Set root directory for hazard curves
ROOT_HAZ = Path(__file__).parents[2] / "2_hazard_calcs" / "results"

# Set root directory for fractiles output
ROOT_OUT = PWD.parent / "results"

# Import fractile values
fin = "fractiles.csv"
FRAC = np.genfromtxt(PWD / fin)
del fin
