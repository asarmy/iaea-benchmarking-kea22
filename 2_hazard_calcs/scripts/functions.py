import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats


def create_wide_output(
    dataframe: pd.DataFrame,
    pivot_column: str,
    value_column: str,
    drop_columns: list = None,
) -> pd.DataFrame:
    """
    This function pivots the input DataFrame by the `pivot_column` and creates a wide output where the values of the
    `value_column` are spread out into separate columns. It is similar to ".out3" in Haz45.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input DataFrame to pivot.
    pivot_column : str
        The name of the column to use as the pivot.
    value_column : str
        The name of the column to use as the values.
    drop_columns : list, optional
        A list of column names to drop from the input DataFrame before pivoting, by default None.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the values pivoted to columns.
    """

    # Drop columns if specified
    dataframe = dataframe.copy()
    if drop_columns is not None:
        dataframe = dataframe.drop(drop_columns, axis=1)

    # Pivot long-to-wide
    df = dataframe.pivot(
        index=dataframe.columns.difference([pivot_column, value_column]),
        columns=pivot_column,
        values=value_column,
    ).reset_index()
    df = df.rename_axis(None, axis=1)

    # Set columns to original order
    pivot_vals = dataframe[pivot_column].unique()
    cols = (
        dataframe.columns.drop(
            ["ln_median", "sigma_total", pivot_column, value_column]
        ).tolist()
        + pivot_vals.tolist()
    )
    df = df[cols]

    # Sort by columns
    sort_cols = ["side", "FAULT_ID", "SCENARIO_ID", "MODEL_ID"]
    df = df.sort_values(by=sort_cols).reset_index(drop=True)

    return df


def calc_hazard(
    dataframe: pd.DataFrame, displacement_array: np.ndarray, output_directory: Path
) -> None:
    """
    #TODO: define dataframe columns; they are very specific to this project.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input DataFrame to pivot.
    displacement_array : np.ndarray
        The array of displacment amplitude test values in meters.
    output_directory : Path
        The directory where outputs are saved.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the values pivoted to columns.
    """

    # Expand dataframe for displacement test values
    df = dataframe.copy()
    df = pd.merge(df, pd.Series(displacement_array, name="displ_m"), how="cross")

    # Calculate probability of exceedance
    f = lambda row: 1 - stats.norm.cdf(
        x=np.log(row["displ_m"]), loc=row["ln_median"], scale=row["sigma_total"]
    )
    df["prob_ex"] = df.apply(f, axis=1)

    # Save as wide-format .out1, where .out1 is for prob_ex
    df2 = create_wide_output(df, "displ_m", "prob_ex", None)
    fout = "hazard_matrix_probex.out1"
    df2.to_csv(output_directory / fout, index=False)
    del fout, df2

    # Calculate unweighted annual frequency of exceedance
    df["afe"] = df["prob_ex"] * df["scenario_rate"]

    # Save as wide-format .out2, where .out2 is for unweighted afe
    df2 = create_wide_output(df, "displ_m", "afe", ["prob_ex"])
    fout = "hazard_matrix_afe_unweighted.out2"
    df2.to_csv(output_directory / fout, index=False)
    del fout, df2

    # Calculate weighted annual frequency of exceedance
    df["afe_wtd"] = df["afe"] * df["total_wt"]

    # Save as wide-format .out3, where .out3 has the weighted afes with mean hazard
    df2 = create_wide_output(df, "displ_m", "afe_wtd", ["prob_ex", "afe"])
    # Sum over magnitude-frequency distribution
    mean_haz_sides = df2.groupby(["FAULT_ID", "MODEL_ID", "side"]).sum().reset_index()
    cols = ["side"] + displacement_array.tolist()
    mean_haz_sides = mean_haz_sides[cols].groupby("side").sum().reset_index()
    mean_haz_sides["FAULT_ID"] = "Wt_Total_Events/yr"
    mean_haz = mean_haz_sides.groupby("FAULT_ID").mean().reset_index()
    mean_haz["side"] = "mean"
    df2 = pd.concat([df2, mean_haz_sides, mean_haz])
    fout = "hazard_matrix_afe_weighted.out3"
    df2.to_csv(output_directory / fout, index=False)
    del fout, df2

    # Save all results in long-format
    fout = "full_results.csv"
    df.to_csv(output_directory / fout, index=False)
    del fout
