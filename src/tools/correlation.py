
import pandas as pd


def calculate_correlation(
    df: pd.DataFrame,
    column_x: str,
    column_y: str
) -> dict:
    """Calculate Pearson correlation between two numeric columns."""

    if column_x == column_y:
        raise ValueError("Choose two different columns.")

    for column in (column_x, column_y):
        if column not in df.columns:
            raise ValueError(
                f"Column '{column}' does not exist."
            )

        series = df[column]

        if (
            not pd.api.types.is_numeric_dtype(series)
            or pd.api.types.is_bool_dtype(series)
        ):
            raise ValueError(
                f"Column '{column}' must be numeric."
            )

    # Keep only rows with valid, finite values in both columns.
    pairs = df[[column_x, column_y]].replace(
        [float("inf"), float("-inf")],
        float("nan")
    ).dropna()

    if len(pairs) < 2:
        raise ValueError(
            "At least two complete numeric rows are required."
        )

    # Correlation cannot be calculated if either column is constant.
    if (
        pairs[column_x].nunique() < 2
        or pairs[column_y].nunique() < 2
    ):
        raise ValueError(
            "Cannot calculate correlation for a constant column."
        )

    correlation = pairs[column_x].corr(
        pairs[column_y],
        method="pearson"
    )

    if pd.isna(correlation):
        raise ValueError(
            "Correlation could not be calculated."
        )

    return {
        "column_x": column_x,
        "column_y": column_y,
        "method": "pearson",
        "correlation": float(correlation),
        "rows_used": int(len(pairs)),
        "rows_excluded": int(len(df) - len(pairs)),
    }
