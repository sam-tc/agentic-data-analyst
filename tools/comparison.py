import pandas as pd


SUPPORTED_AGGREGATIONS = {
    "mean",
    "sum",
    "median",
    "min",
    "max",
    "count",
}


def compare_groups(
    df: pd.DataFrame,
    group_by: str,
    value_column: str,
    aggregation: str
) -> dict:
    """
    Compare groups by applying an aggregation to a numerical column.
    """

    if group_by not in df.columns:
        return {
            "error": f"Group column '{group_by}' does not exist in the dataset."
        }

    if value_column not in df.columns:
        return {
            "error": f"Value column '{value_column}' does not exist in the dataset."
        }

    if aggregation not in SUPPORTED_AGGREGATIONS:
        return {
            "error": f"Unsupported aggregation: {aggregation}"
        }

    if aggregation != "count" and not pd.api.types.is_numeric_dtype(
        df[value_column]
    ):
        return {
            "error": (
                f"Value column '{value_column}' must be numerical "
                f"for {aggregation}."
            )
        }

    grouped = df.groupby(group_by, dropna=False)[value_column]

    if aggregation == "mean":
        result = grouped.mean()
    elif aggregation == "sum":
        result = grouped.sum()
    elif aggregation == "median":
        result = grouped.median()
    elif aggregation == "min":
        result = grouped.min()
    elif aggregation == "max":
        result = grouped.max()
    else:
        result = grouped.count()

    return {
        "group_by": group_by,
        "value_column": value_column,
        "aggregation": aggregation,
        "results": result.to_dict(),
    }