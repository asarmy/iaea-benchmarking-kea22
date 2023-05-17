# Import python libraries
import numpy as np
from pathlib import Path

# Import configurations
from fractile_config import *

# Import package functions
from functions import calc_weighted_statistics, aggregate_hazard_branches

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

# Set standard filename
FILE = "full_results.csv"


# Compute fractiles for all study cases
for m in MODELS:
    for c in CASES:

        # Directory set-up
        dir_haz = ROOT_HAZ / c / m
        dir_outputs = ROOT_OUT / c / m
        dir_outputs.mkdir(parents=True, exist_ok=True)

        # Import all hazard curves
        df = pd.read_csv(dir_haz / FILE, low_memory=False)

        # Calculate epistemic hazard curves
        # The FDM sides and SSC_ID branches are treated as epistemic uncertainty
        df_results = aggregate_hazard_branches(df)

        # Calculate fractiles and mean hazard for each side
        # FIXME: left is assumed to be U* and right is assumed to be 1-U*; fix wording
        results_sides = (
            df_results.groupby(["side", "displ_m"])
            .apply(
                calc_weighted_statistics,
                afe_column="afe",
                weights_column="total_wt2",
                fractiles=FRAC,
            )
            .reset_index()
        )

        # Calculate fractiles and mean hazard for both sides
        results_mean = (
            df_results.groupby(["displ_m"])
            .apply(
                calc_weighted_statistics,
                afe_column="afe",
                weights_column="total_wt2",
                fractiles=FRAC,
            )
            .reset_index()
        )
        results_mean["side"] = "folded"

        # Final fractiles output for each side and weighted mean
        results_final = pd.concat([results_sides, results_mean], axis=0)
        results_final.reset_index(drop=True)

        # Save results
        fout = "fractiles.csv"
        results_final.to_csv(dir_outputs / fout, index=False)
        fout = "epistemic_haz_curves.csv"
        df_results.to_csv(dir_outputs / fout, index=False)

        # Print status
        print(f"*** Fractile calculations complete for {c} with {m}.", flush=True)
