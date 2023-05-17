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

# Set the case name
CASE = "kumamoto_case2"

# Set implementations of KEA22 model to loop over
MODELS = ["mean_model", "full_model"]

# Set standard filename
FILE = "mean_hazard_source_contributions.csv"


# Create plots
for m in MODELS:

    # Directory set-up
    dir_data = ROOT_RES / CASE / m
    dir_outputs = ROOT_OUT / CASE / m
    dir_outputs.mkdir(parents=True, exist_ok=True)

    # Import case info block to add to the plots
    fin = DIR_INFO / f"{CASE}_{m}.txt"
    info = get_case_info_block(fin)

    # Import results
    df = pd.read_csv(dir_data / FILE, low_memory=False)

    # Loop over left, right, folded subsets; plot each separately
    # FIXME: left is assumed to be U* and right is assumed to be 1-U*; fix wording
    for s in ["left", "right", "folded"]:
        # Subset
        df_subset = subset(df, s)

        # Plotting
        fig, ax = plt.subplots(1, 1)
        plot_kumamoto_source_contributions(ax, df_subset, s, info)

        fout = f"source_contributions_{s}.png"
        plt.savefig(dir_outputs / fout, bbox_inches="tight")
        plt.close(fig)

    # Print status
    print(f"*** Source contributions for Kumamoto Sensitivity 2 plotted for {CASE} with {m}.", flush=True)
