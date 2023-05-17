# Import python libraries
import matplotlib.pyplot as plt
import pandas as pd
import sys
from pathlib import Path

# Import configurations
from plotting_config import *
import plot_style

# Import package functions
from functions import *

# Set cases to loop over
CASES = [
    "norcia_case1",
    "le_teil_case2",
    "le_teil_extra",
    "kumamoto_case3",
    "kumamoto_case2",
]

# Set standard filename
FILE = "fractiles.csv"


# Create plots for all study cases
for c in CASES:

    # Directory set-up
    dir_data_mean_model = ROOT_RES / c / "mean_model"
    dir_data_full_model = ROOT_RES / c / "full_model"
    dir_outputs = ROOT_OUT / c
    dir_outputs.mkdir(parents=True, exist_ok=True)

    # Import case info block to add to the plots
    fin = DIR_INFO / f"{c}_both_models.txt"
    info = get_case_info_block(fin)

    # Get plotting axis limits
    xlimits, ylimits = LIMS_DICT[c]["x"], LIMS_DICT[c]["y"]

    # Import results
    df_mean = pd.read_csv(dir_data_mean_model / FILE, low_memory=False)
    df_full = pd.read_csv(dir_data_full_model / FILE, low_memory=False)

    # Convert wide-to-long
    df_full_long = pd.melt(df_full, id_vars=["side", "displ_m"], value_name="afe")

    # Loop over left, right, folded subsets; plot each separately
    # FIXME: left is assumed to be U* and right is assumed to be 1-U*; fix wording
    for s in ["left", "right", "folded"]:
        # Subset
        df_mean_subset = subset(df_mean, s)
        df_full_long_subset = subset(df_full_long, s)

        # Plotting
        fig, ax = plt.subplots(1, 1)
        plot_haz_curve_comparisons(
            ax, xlimits, ylimits, df_mean_subset, df_full_long_subset, s, info
        )

        fout = f"epistemic_haz_curves_compare_FDMs_{s}.png"
        plt.savefig(dir_outputs / fout, bbox_inches="tight")
        plt.close(fig)

    # Print status
    print(f"*** Hazard curve FDM comparisons plotted for {c}.", flush=True)
