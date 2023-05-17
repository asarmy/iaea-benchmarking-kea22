# Import python libraries
import numpy as np
from pathlib import Path

# Import configurations
from fractile_config import *

# Import package functions
from functions import reshape_for_source_contributions

# Set the case name
CASE = "kumamoto_case2"

# Set implementations of KEA22 model to loop over
MODELS = ["mean_model", "full_model"]

# Set fault names for column order in output
COLS = ["Float", "F2", "F1_F2", "F2_F3", "Full"]

# Set output directory
DIR_OUT = ROOT_OUT / CASE

# Set standard filename
FILE = "full_results.csv"


# Loop over mean and full model results
for m in MODELS:
    DIR_RES = ROOT_HAZ / CASE / m

    # Import results
    df = pd.read_csv(DIR_RES / FILE, low_memory=False)

    # Subset
    df = df[
        ["FAULT_ID", "SCENARIO_ID", "MODEL_ID", "fdm_wt", "side", "displ_m", "afe"]
    ].copy()

    # Calculate afe weighted by model weight only; keep ssc weights unincluded
    df["afe_wtd2"] = df["afe"] * df["fdm_wt"]

    # Sum on FDM runs
    df = df.groupby(["FAULT_ID", "side", "displ_m"])["afe_wtd2"].sum().reset_index()

    # Reshape for source constributions
    df_wide = reshape_for_source_contributions(df, "afe_wtd2")
    cols = ["displ_m", "side"] + COLS
    df_wide = df_wide[cols].copy()

    # Calculate mean of the sides, append to dataframe
    # FIXME: left is assumed to be U* and right is assumed to be 1-U*; fix wording
    _means = df_wide.groupby(["displ_m"]).mean().reset_index()
    _means["side"] = "folded"
    df_wide = pd.concat([df_wide, _means], axis=0)

    # Calculate weighted branch (which is known to be 50/50)
    df_wide["Combined_F2_and_Float"] = df_wide[["Float", "F2"]].mean(axis=1)

    # Calculate total mean hazard
    df_wide["Total"] = df_wide[["Combined_F2_and_Float", "F1_F2", "F2_F3", "Full"]].sum(
        axis=1
    )

    # Save results
    fout = "mean_hazard_source_contributions.csv"
    df_wide.to_csv(DIR_OUT / m / fout, index=False)
