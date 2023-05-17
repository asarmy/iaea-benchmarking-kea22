import pandas as pd


def subset(dataframe: pd.DataFrame, side: str) -> pd.DataFrame:
    """
    Subset a dataframe based on the specified side.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input dataframe containing the data.
    side : str
        The profile peak side to subset by, either "left", "right", or "folded".

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
