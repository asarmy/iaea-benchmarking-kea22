import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from statsmodels.stats.weightstats import DescrStatsW


def calc_weighted_statistics(
    dataframe: pd.DataFrame, afe_column: str, weights_column: str, fractiles: list
) -> pd.DataFrame:
    """
    Calculate weighted descriptive statistics for a given column.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input dataframe containing the data to analyze.
    afe_column : str
        The column name in the dataframe containing the values to calculate statistics for.
    weights_column : str
        The column name in the dataframe containing the weights to be used in the calculations.
    fractiles : list
        A list of fractiles (quantiles) to calculate for the given column.

    Returns
    -------
    info : pd.DataFrame
        A dataframe containing the calculated weighted statistics (mean and quantiles) for the given column.

    """

    stats = DescrStatsW(dataframe[afe_column], weights=dataframe[weights_column])
    info = stats.quantile(fractiles)
    info["Mean"] = stats.mean

    return info


def aggregate_hazard_branches(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate epistemic hazard curves. The FDM sides and SSC_ID branches are treated as epistemic uncertainty.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input dataframe containing the data to analyze.

    Returns
    -------
    df_results : pd.DataFrame
        A dataframe containing the epistemic hazard curves.

    """
    # Identify SSC branches for fractile calcs; if n > 1, then SSC model has epistemic
    # Also get weights for each SSC branch
    ssc_all_branches = (
        dataframe[["SSC_ID", "ssc_wt"]]
        .drop_duplicates()
        .set_index("SSC_ID")["ssc_wt"]
        .to_dict()
    )

    if len(ssc_all_branches) > 1:

        # It is assumed in the file formatting that SSC_ID=0 is aleatory and non-zero SSC_ID values are epistemic SSC branches
        ssc_epistemic_branches = ssc_all_branches.copy()
        ssc_epistemic_branches.pop(0, None)

        # Extract aleatory branch results
        df_branch_zero = dataframe[dataframe["SSC_ID"] == 0]

        # Subset analysis for each epistemic brach to combine epistemic & aleatory branches
        df_results = pd.DataFrame()

        for branch, wt in ssc_epistemic_branches.items():
            # Combine an epistemic branch with the aleatory branch
            _df = dataframe[dataframe["SSC_ID"] == branch]
            _df = pd.concat([df_branch_zero, _df], axis=0)
            _df["ssc_alt"] = branch

            # Sum for each FDM run; if 2000 runs, then each side-displ combo should have 2000 curves
            results = (
                _df.groupby(["ssc_alt", "MODEL_ID", "side", "displ_m"])["afe"]
                .sum()
                .reset_index()
            )

            # Retain SSC epistemic branch weighing for DescrStatsW
            results["total_wt2"] = wt
            df_results = pd.concat([df_results, results], axis=0)
    else:
        # There is no SSC epistemic uncertainty; keep the "ssc_alt" flag for consistency
        dataframe["ssc_alt"] = 1
        df_results = (
            dataframe.groupby(["ssc_alt", "MODEL_ID", "side", "displ_m", "total_wt"])[
                "afe"
            ]
            .sum()
            .reset_index()
        )
        df_results["total_wt2"] = df_results["total_wt"]

    return df_results.reset_index(drop=True)


def reshape_for_source_contributions(
    dataframe: pd.DataFrame, value_column: str
) -> pd.DataFrame:
    """
    Pivot the dataframe based on the 'FAULT_ID' column.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input dataframe with hazard curves in long format.
    value_column : str
        The column name containing the values for pivoting.

    Returns
    -------
    pd.DataFrame
        The pivoted dataframe.
    """

    df = dataframe.pivot(
        index=dataframe.columns.difference(["FAULT_ID", value_column]),
        columns="FAULT_ID",
        values=value_column,
    ).reset_index()

    df = df.rename_axis(None, axis=1)

    # Extract columns and sort
    faults = dataframe["FAULT_ID"].unique().tolist()
    cols = ["displ_m", "side"] + faults
    df = df[cols]
    sort_cols = ["side", "displ_m"]
    df = df.sort_values(by=sort_cols).reset_index(drop=True)

    return df
