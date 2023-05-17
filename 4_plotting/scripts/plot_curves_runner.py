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

# Set implementations of KEA22 model to loop over
MODELS = ["mean_model", "full_model"]

# Define mean model cases that don't have SSC epistemic uncertainty
NO_EPI = ["le_teil_case2", "kumamoto_case3", "norcia_case1"]

# Set standard filenames
FILE_FRACTILES = "fractiles.csv"
FILE_CURVES = "epistemic_haz_curves.csv"


# Create plots for all study cases
for m in MODELS:
    for c in CASES:

        # Directory set-up
        dir_data = ROOT_RES / c / m
        dir_outputs = ROOT_OUT / c / m
        dir_outputs.mkdir(parents=True, exist_ok=True)

        # Import case info block to add to the plots
        fin = DIR_INFO / f"{c}_{m}.txt"
        info = get_case_info_block(fin)

        # Get plotting axis limits
        xlimits, ylimits = LIMS_DICT[c]["x"], LIMS_DICT[c]["y"]

        # Import results
        df_frac = pd.read_csv(dir_data / FILE_FRACTILES, low_memory=False)
        df_curves = pd.read_csv(dir_data / FILE_CURVES, low_memory=False)

        # Convert wide-to-long
        df_fract_long = pd.melt(df_frac, id_vars=["side", "displ_m"], value_name="afe")

        # Loop over left, right, folded subsets; plot each separately
        # FIXME: left is assumed to be U* and right is assumed to be 1-U*; fix wording
        do_fracs = True if c in NO_EPI and m == "mean_model" else False
        for s in ["left", "right", "folded"]:
            # Subset
            df_curves_subset = subset(df_curves, s)
            df_frac_subset = subset(df_fract_long, s)

            # Plotting
            fig, ax = plt.subplots(1, 1)
            plot_haz_curves(
                ax,
                xlimits,
                ylimits,
                df_curves_subset,
                df_frac_subset,
                s,
                info,
                skip_fractiles=do_fracs,
            )

            # Save plot
            fout = f"epistemic_haz_curves_{s}.png"
            plt.savefig(dir_outputs / fout, bbox_inches="tight")
            plt.close(fig)

        # Print status
        print(f"*** Hazard curves plotted for {c} with {m}.", flush=True)
