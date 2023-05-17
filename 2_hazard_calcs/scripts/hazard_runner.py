# Import python libraries
import numpy as np
from pathlib import Path

# Import configurations
from hazard_config import *

# Import package functions
from functions import calc_hazard

# Set cases to loop over
CASES = [
    "norcia_case1",
    "le_teil_case2",
    "le_teil_extra",
    "kumamoto_case3",
    "kumamoto_case2",
]

# Set implementations of KEA22 model to loop over
MODELS = ["mean_model", "full_model"]

# Set filenames for model predictions
# FIXME: left is assumed to be U* and right is assumed to be 1-U*; fix wording
FILES = {"left": "site.csv", "right": "complement.csv"}

# Compute hazard curves for all logic tree branches
for m in MODELS:
    for c in CASES:

        # Directory set-up
        dir_predictions = ROOT_PRED / c / m
        dir_outputs = ROOT_OUT / c / m
        dir_outputs.mkdir(parents=True, exist_ok=True)

        # Import results into a dataframe
        df = pd.DataFrame()
        for key, filename in FILES.items():
            _df = pd.read_csv(dir_predictions / filename, low_memory=False)
            _df["side"] = key
            df = pd.concat([df, _df], ignore_index=True)

        # Run hazard
        calc_hazard(df, DISPL, dir_outputs)

        # Print status
        print(f"*** Hazard run complete for {c} with {m}.", flush=True)
