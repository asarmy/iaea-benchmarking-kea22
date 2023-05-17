import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Set console output formatting
pd.set_option("display.max_columns", 800)
pd.set_option("display.width", 800)

# Define scripts working directory
PWD = Path(sys.argv[0]).absolute().parent

# Set root directory for scenario inputs
ROOT_INP = Path(__file__).parents[1] / "inputs"

# Set root directory for model prediction outputs
ROOT_OUT = PWD.parent / "results"

# Set directory for model calculations and add to path
MODEL_DIR = PWD.parents[1] / "KuehnEtAl2022"
sys.path.append(str(MODEL_DIR))
