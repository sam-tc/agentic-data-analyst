import math

import numpy as np
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
    non_missing = series.dropna()
    valid_values = non_missing

    if operation == "count":
        result = int(len(non_missing))

    else:
        if (
            not pd.api.types.is_numeric_dtype(series)
            or pd.api.types.is_bool_dtype(series)
        ):
            raise ValueError(
                f"Column '{column}' must contain numeric data."
            )

        valid_values = series.replace(
            [float("inf"), float("-inf")], float("nan")
        ).dropna()

        if valid_values.empty:
            raise ValueError(
                f"Column '{column}' has no finite numeric values."
            )

        operations = {
            "sum": lambda s: s.sum(),
            "mean": lambda s: s.mean(),
            "median": lambda s: s.median(),
            "min": lambda s: s.min(),
            "max": lambda s: s.max(),
        }

        with np.errstate(over="ignore", invalid="ignore"):
            result = float(operations[operation](valid_values))

        if not math.isfinite(result):
            raise ValueError(
                "The result exceeds the supported finite numeric range."
            )

    return {
        "column": column,
        "operation": operation,
        "value": result,
        "non_missing_rows": int(len(non_missing)),
        "rows_used": int(len(valid_values)),
        "non_finite_excluded": int(len(non_missing) - len(valid_values)),
    }
