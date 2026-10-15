import math

import pandas as pd


def profile_dataset(df: pd.DataFrame) -> dict:
    """Return structural and statistical information about a dataset."""

    columns = []

    for column_name in df.columns:
        series = df[column_name]

        info = {
            "name": str(column_name),
            "data_type": str(series.dtype),
            "missing_values": int(series.isna().sum()),
            "unique_values": int(series.nunique(dropna=True)),
        }

        if (
            pd.api.types.is_numeric_dtype(series)
            and not pd.api.types.is_bool_dtype(series)
        ):
            valid_values = series.replace(
                [float("inf"), float("-inf")], float("nan")
            ).dropna()

            if not valid_values.empty:
                mean = float(valid_values.mean())

                info["statistics"] = {
                    "minimum": float(valid_values.min()),
                    "maximum": float(valid_values.max()),
                    "mean": mean if math.isfinite(mean) else None,
                }

        columns.append(info)

    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": columns,
    }
