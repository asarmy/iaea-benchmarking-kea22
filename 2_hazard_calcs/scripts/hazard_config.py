import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Set console output formatting
pd.set_option("display.max_columns", 800)
pd.set_option("display.width", 800)

# Define scripts working directory
PWD = Path(sys.argv[0]).absolute().parent

# Set root directory for model predictions
ROOT_PRED = Path(__file__).parents[2] / "1_model_predictions" / "results"

# Set root directory for hazard output
ROOT_OUT = PWD.parent / "results"

# Import displacement test values
fin = "displ_array_meters.csv"
DISPL = np.genfromtxt(PWD / fin)
del fin
