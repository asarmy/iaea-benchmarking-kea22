# Import python libraries
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# Import configurations
from plotting_config import *


def add_minor_gridlines(ax_obj, axis):
    plt.minorticks_on()
    ax_obj.grid(which="minor", axis=axis, color="#DDDDDD", lw=0.4, alpha=0.5)


def format_log_axes(ax_obj: plt.Axes, xlims: list, ylims: list) -> None:
    """
    Format the axes of a plot with logarithmic scales and nice tick labels.

    Parameters
    ----------
    ax_obj : plt.Axes
        The matplotlib Axes object to format.
    xlims : tuple
        The x-axis limits as a list [xmin, xmax].
    ylims : tuple
        The y-axis limits as a list [xmin, xmax].

    Returns
    -------
    None
    """

    xticks = [0.001, 0.01, 0.1, 1, 10, 100]
    ax_obj.set_xticks(xticks)
    ax_obj.set(xlim=xlims, ylim=ylims, xscale="log", yscale="log")
    ax_obj.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    add_minor_gridlines(ax_obj, axis="both")
    ax_obj.set(xlabel="Displacement (m)", ylabel="AFE ($yr^{-1}$)")


def subset(dataframe: pd.DataFrame, side: str) -> pd.DataFrame:
    """
    Subset a dataframe based on the specified side.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input dataframe containing the data.
    side : str
        The profile peak side to subset by, either "left", "right", or "folded".
        FIXME: left is assumed to be U* and right is assumed to be 1-U*; fix wording

    Returns
    -------
    pd.DataFrame
        The subsetted dataframe.
    """

    if side not in ["left", "right", "folded"]:
        raise ValueError(
            "Invalid value for 'side'. It must be either 'left', 'right', or 'folded'."
        )

    return dataframe[dataframe["side"] == side]


def plot_haz_curves(
    ax_obj: plt.Axes,
    x_lims: list,
    y_lims: list,
    dataframe_curves_long: pd.DataFrame,
    dataframe_fractiles_long: pd.DataFrame,
    side: str,
    info_block: str,
    skip_fractiles: bool = False,
) -> None:
    """
    Plot hazard curves and fractiles on the given axes object. Epistmic hazard curves
    (unweighted) are always plotted, and the mean hazard curve is always plotted. The
    plotting of fractile curves is optional.

    Parameters
    ----------
    ax_obj : matplotlib.axes.Axes
        Axes object for the plot.
    x_lims : list
        X-axis limits for the plot.
    y_lims : list
        Y-axis limits for the plot.
    dataframe_curves_long : pandas.DataFrame
        DataFrame in long format containing hazard curves.
    dataframe_fractiles_long : pandas.DataFrame
        DataFrame in long format containing fractile and mean hazard curves.
    side : str
        The profile peak side being plotted, either "left", "right", or "folded".
        FIXME: left is assumed to be U* and right is assumed to be 1-U*; fix wording
    info_block : str
        Additional information for the plot.
    skip_fractiles : bool, optional
        Whether to skip plotting fractiles. Defaults to False.

    Returns
    -------
    None
    """

    if side not in ["left", "right", "folded"]:
        raise ValueError(
            "Invalid value for 'side'. It must be either 'left', 'right', or 'folded'."
        )

    x_column, y_column = "displ_m", "afe"

    # Plot the unweighted branch hazard curves if applicable
    if side != "folded":
        for run, group in dataframe_curves_long.groupby(["MODEL_ID", "ssc_alt"]):
            ax_obj.plot(
                group[x_column],
                group[y_column],
                label=None,
                c=LC_DICT["contribs"],
                ls=LS_DICT["contribs"],
                lw=LW_DICT["contribs"],
            )

    # Plot fractiles
    if skip_fractiles:
        group = dataframe_fractiles_long[dataframe_fractiles_long["variable"] == "Mean"]
        ax_obj.plot(
            group[x_column],
            group[y_column],
            label="Mean",
            c=LC_DICT[side],
            ls=LS_DICT["Mean"],
            lw=LW_DICT["Mean"],
        )
        title="Hazard Curves"

    else:
        for level, group in dataframe_fractiles_long.groupby(["variable"]):
            label = "Mean" if level == "Mean" else f"Percentile: {level}"
            ax_obj.plot(
                group[x_column],
                group[y_column],
                label=label,
                c=LC_DICT[side],
                ls=LS_DICT[level],
                lw=LW_DICT[level],
            )
        title="Epistemic Hazard Curves"

    # Format with log-log axes
    format_log_axes(ax_obj, x_lims, y_lims)
    ax_obj.set(title=title)

    # Add unweighted branch hazard curves to legend if applicable
    handles, labels = ax_obj.get_legend_handles_labels()
    if side != "folded":
        n_curves = (
            dataframe_curves_long["MODEL_ID"].nunique()
            * dataframe_curves_long["ssc_alt"].nunique()
        )
        label = f"Unwtd. Branches, n={n_curves}"
        line = Line2D([0], [0], label=label, color="gray", lw=0.4)
        handles.append(line)
        labels.append(label)

    leg = ax_obj.legend(
        loc="upper left",
        bbox_to_anchor=(1.044, 0.6),
        handles=handles,
        labels=labels,
        frameon=False,
        title="Legend",
        title_fontsize=5,
    )
    leg._legend_box.align = "left"

    # Add info block to outside of plot
    label = {"left": "$U_*$", "right": "$1 - U_*$", "folded": "None (Equal Wt.)"}
    info2 = info_block + f"\nFDM Asymmetry: {label[side]}"
    plt.figtext(0.95, 0.6, info2, fontsize=5)


def plot_haz_curve_comparisons(
    ax_obj: plt.Axes,
    x_lims: list,
    y_lims: list,
    dataframe_mean_model: pd.DataFrame,
    dataframe_full_model_long: pd.DataFrame,
    side: str,
    info_block: str,
) -> None:
    """
    Plot mean hazard curves for both FDMs and fractiles from full model.

    Parameters
    ----------
    ax_obj : matplotlib.axes.Axes
        Axes object for the plot.
    x_lims : list
        X-axis limits for the plot.
    y_lims : list
        Y-axis limits for the plot.
    dataframe_mean_model : pandas.DataFrame
        DataFrame containing fractiles for mean model (only mean is used).
    dataframe_full_model_long : pandas.DataFrame
        DataFrame in long format containing fractile and mean hazard curves for full
        model.
    side : str
        The profile peak side being plotted, either "left", "right", or "folded".
    info_block : str
        Additional information for the plot.

    Returns
    -------
    None
    """

    if side not in ["left", "right", "folded"]:
        raise ValueError(
            "Invalid value for 'side'. It must be either 'left', 'right', or 'folded'."
        )

    x_column, y_column = "displ_m", "afe"

    # Plot results from full model
    for level, group in dataframe_full_model_long.groupby(["variable"]):
        label = (
            "Mean (from FDM w/ Epi)"
            if level == "Mean"
            else f"Percentile: {level} (from FDM w/ Epi)"
        )
        ax_obj.plot(
            group[x_column],
            group[y_column],
            label=label,
            c="tab:red",
            ls=LS_DICT[level],
            lw=LW_DICT[level],
        )

    # Plot results from mean model
    ax_obj.plot(
        dataframe_mean_model[x_column],
        dataframe_mean_model["Mean"],
        label="Mean (from FDM w/o Epi)",
        c="black",
        ls=LS_DICT["Mean"],
        lw=LW_DICT["Mean"],
    )

    # Format and add legend
    ax_obj.set(title="Hazard Curve Fractiles")
    format_log_axes(ax_obj, x_lims, y_lims)

    leg = ax_obj.legend(
        loc="upper left",
        bbox_to_anchor=(1.044, 0.6),
        frameon=False,
        title="Legend",
        title_fontsize=5,
    )
    leg._legend_box.align = "left"

    # Add info block to outside of plot
    label = {"left": "$U_*$", "right": "$1 - U_*$", "folded": "None (Equal Wt.)"}
    info2 = info_block + f"\nFDM Asymmetry: {label[side]}"
    plt.figtext(0.95, 0.6, info2, fontsize=5)


def plot_kumamoto_source_contributions(
    ax_obj: plt.Axes,
    dataframe: pd.DataFrame,
    side: str,
    info_block: str,
) -> None:
    """
    Plot source contributions for Kumamoto Sensitivity 2. Note that the faults, etc.
    from this case are hard-coded, so this function cannot be used with other cases.

    Parameters
    ----------
    ax_obj : matplotlib.axes.Axes
        Axes object for the plot.
    dataframe : pandas.DataFrame
        DataFrame containing hazard contributions by source.
    side : str
        The profile peak side being plotted, either "left", "right", or "folded".
    info_block : str
        Additional information for the plot.

    Returns
    -------
    None
    """

    if side not in ["left", "right", "folded"]:
        raise ValueError(
            "Invalid value for 'side'. It must be either 'left', 'right', or 'folded'."
        )

    # Get plotting axis limits
    xlimits = LIMS_DICT["kumamoto_case2"]["x"]
    ylimits = LIMS_DICT["kumamoto_case2"]["y"]

    # Note KM_X dictionaries are imported from plotting_config file
    for source, label in KM_ID_DICT.items():
        ax_obj.plot(
            dataframe["displ_m"],
            dataframe[source],
            label=label,
            c=KM_LC_DICT[source],
            ls=KM_LS_DICT[source],
            lw=KM_LW_DICT[source],
        )

    # Format and add legend
    ax_obj.set(title="Source Contributions")
    format_log_axes(ax_obj, xlimits, ylimits)

    leg = ax_obj.legend(
        loc="upper left",
        bbox_to_anchor=(1.044, 0.6),
        frameon=False,
        title="Legend",
        title_fontsize=5,
    )
    leg._legend_box.align = "left"

    # Add info block to outside of plot
    label = {"left": "$U_*$", "right": "$1 - U_*$", "folded": "None (Equal Wt.)"}
    info2 = info_block + f"\nFDM Asymmetry: {label[side]}"
    plt.figtext(0.95, 0.6, info2, fontsize=5)


def get_case_info_block(filepath: str) -> str:
    """
    Read the content of a text file.

    Parameters
    ----------
    filepath : str
        Path to the file.

    Returns
    -------
    str
        Content of the text file.
    """

    with open(filepath, "r") as i:
        content: str = i.read()
    return content
