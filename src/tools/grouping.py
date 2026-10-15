import pandas as pd


def compare_groups(
    df: pd.DataFrame,
    group_column: str,
    operation: str,
    value_column: str | None = None
) -> dict:
    """Group rows and compare counts or numeric statistics."""

    if group_column not in df.columns:
        raise ValueError(
            f"Group column '{group_column}' does not exist."
        )

    allowed = {"count", "sum", "mean", "median", "min", "max"}

    if operation not in allowed:
        raise ValueError(f"Unsupported operation: {operation}")

    grouped = df.groupby(group_column, dropna=False)

    if operation == "count":
        result = grouped.size()

    else:
        if value_column is None:
            raise ValueError(
                "A value_column is required for this operation."
            )

        if value_column not in df.columns:
            raise ValueError(
                f"Value column '{value_column}' does not exist."
            )

        series = df[value_column]

        if (
            not pd.api.types.is_numeric_dtype(series)
            or pd.api.types.is_bool_dtype(series)
        ):
            raise ValueError(
                f"Value column '{value_column}' must be numeric."
            )

        clean_df = df.copy()
        clean_df[value_column] = clean_df[value_column].replace(
            [float("inf"), float("-inf")], float("nan")
        )

        values = clean_df.groupby(
            group_column,
            dropna=False
        )[value_column]

        if operation == "sum":
            result = values.sum(min_count=1)
        else:
            result = values.agg(operation)

    sorted_result = result.sort_values(
        ascending=False,
        na_position="last",
        kind="stable"
    )

    rows = []

    for group, number in sorted_result.items():
        rows.append({
            "group": None if pd.isna(group) else str(group),
            "value": (
                None
                if pd.isna(number) or number in (
                    float("inf"), float("-inf")
                )
                else int(number)
                if operation == "count"
                else float(number)
            ),
        })

    return {
        "group_column": group_column,
        "value_column": (
            value_column if operation != "count" else None
        ),
        "operation": operation,
        "results": rows,
    }
