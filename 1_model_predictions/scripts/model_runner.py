# Import python libraries
import numpy as np
from pathlib import Path

# Import configurations
from model_config import *

# Import package functions
from functions import calc_model_predictions

# Set cases to read and their style of faulting
CASES = {
    "kumamoto_case2.csv": "Strike-Slip",
    "kumamoto_case3.csv": "Strike-Slip",
    "le_teil_case2.csv": "Reverse",
    "le_teil_extra.csv": "Reverse",
    "norcia_case1.csv": "Normal",
}

# Set implementations of KEA22 model to loop over
MODELS = {"mean_model": True, "full_model": False}

# Set sides to loop over
SIDES = ["site", "complement"]


## Loop over cases
for c, sof in CASES.items():

    # Import case information
    df = pd.read_csv(ROOT_INP / c, low_memory=False)

    # Loop over models
    for m, flag in MODELS.items():

        # Directory set-up
        dir_outputs = ROOT_OUT / Path(c).stem / m
        dir_outputs.mkdir(parents=True, exist_ok=True)

        # Loop over sides
        df2 = df.copy()

        for s in SIDES:
            if s == "complement":
                df2["u_star"] = 1 - df2["u_star"]

            # Use a helper function to calculate mu, sigma and clean up dataframe
            df_results = calc_model_predictions(df2, sof, flag)

            # Add a column for the final row weight
            df_results["total_wt"] = df_results["ssc_wt"] * df_results["fdm_wt"]

            # Save results
            fout = f"{s}.csv"
            df_results.to_csv(dir_outputs / fout, index=False)

            # Print status
            print(
                f"*** Model predictions calculated for for {c} with {m} at {s} side.",
                flush=True,
            )
