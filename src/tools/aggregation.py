
import pandas as pd


def aggregate_column(
    df: pd.DataFrame,
    column: str,
    operation: str
) -> dict:
    """Perform a statistical operation on a dataset column."""

    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist.")

    supported_operations = {
        "count", "sum", "mean", "median", "min", "max"
    }

    if operation not in supported_operations:
        raise ValueError(
            f"Unsupported aggregation operation: {operation}"
        )

    series = df[column]
    valid_values = series.dropna()

    if operation == "count":
        result = int(len(valid_values))

    else:
        if (
            not pd.api.types.is_numeric_dtype(series)
            or pd.api.types.is_bool_dtype(series)
        ):
            raise ValueError(
                f"Column '{column}' must contain numeric data."
            )

        if valid_values.empty:
            raise ValueError(
                f"Column '{column}' has no valid numeric values."
            )

        operations = {
            "sum": lambda s: s.sum(),
            "mean": lambda s: s.mean(),
            "median": lambda s: s.median(),
            "min": lambda s: s.min(),
            "max": lambda s: s.max(),
        }

        result = float(operations[operation](valid_values))

    return {
        "column": column,
        "operation": operation,
        "value": result,
        "non_missing_rows": int(len(valid_values)),
    }
