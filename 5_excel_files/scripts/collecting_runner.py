# Import python libraries
from datetime import datetime
import pandas as pd
import sys
from pathlib import Path

# Import configurations
from collecting_config import *

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

# Set today's date for file name
today = datetime.now().strftime("%Y-%b-%d")

# Set standard filenames
FILE = "fractiles.csv"
KUMOMOTO = "mean_hazard_source_contributions.csv"

# Collect results into Excel file
for c in CASES:

    # Directory set-up
    dir_data_mean_model = ROOT_RES / c / "mean_model"
    dir_data_full_model = ROOT_RES / c / "full_model"

    # Import case info block to add to the plots
    fin = DIR_INFO / f"{c}.txt"
    info = get_case_info_block(fin)
    info = pd.DataFrame(info.split("\n"), columns=["Notes"])

    # Import results
    df_mean = pd.read_csv(dir_data_mean_model / FILE, low_memory=False)
    df_full = pd.read_csv(dir_data_full_model / FILE, low_memory=False)

    # Subset for folded model results
    df_mean = subset(df_mean, "folded")
    df_full = subset(df_full, "folded")

    # Add column for centimeters
    df_mean["displ_cm"] = df_mean["displ_m"] * 100
    df_full["displ_cm"] = df_full["displ_m"] * 100

    # Only retain mean hazard curves for cases without SSC epistemic uncertainty
    # Note that in these cases, the fractiles are from the left/right model sides
    if c in NO_EPI:
        df_mean = df_mean[["side", "displ_m", "Mean", "displ_cm"]].copy()

    # Create Excel file and save data
    fout = f"{c}-UCLA_PGE-Results-{today}.xlsx"
    writer = pd.ExcelWriter(ROOT_OUT / fout, engine="xlsxwriter", engine_kwargs=OPTIONS)

    df_full.to_excel(writer, sheet_name="full_fdm", index=False)
    df_mean.to_excel(writer, sheet_name="mean_fdm", index=False)

    # Include source contribution curves if Kumamoto Sensitivity 2
    if c == "kumamoto_case2":
        # Import results
        df_mean_contribs = pd.read_csv(dir_data_mean_model / KUMOMOTO, low_memory=False)
        df_full_contribs = pd.read_csv(dir_data_full_model / KUMOMOTO, low_memory=False)
        # Subset
        df_mean_contribs = subset(df_mean_contribs, "folded")
        df_full_contribs = subset(df_full_contribs, "folded")
        # Centimeters
        df_mean_contribs["displ_cm"] = df_mean_contribs["displ_m"] * 100
        df_full_contribs["displ_cm"] = df_full_contribs["displ_m"] * 100
        # Excel
        df_mean_contribs.to_excel(writer, sheet_name="sources_mean_fdm", index=False)
        df_full_contribs.to_excel(writer, sheet_name="sources_full_fdm", index=False)

    # Include info notes
    info.to_excel(writer, sheet_name="Notes", index=False)

    # Save and close writer
    writer.save()
    writer.close()

    # Print status
    print(f"*** Excel files created for {c}.", flush=True)
