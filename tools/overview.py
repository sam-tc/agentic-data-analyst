import pandas as pd


def profile_dataset(df: pd.DataFrame) -> dict:
    """
    Analyze the structure and basic quality of a tabular dataset.
    """

    column_info = {}

    for column in df.columns:
        series = df[column]

        info = {
            "data_type": str(series.dtype),
            "missing_values": int(series.isna().sum()),
            "missing_percentage": round(
                float(series.isna().mean() * 100), 2
            ),
            "unique_values": int(series.nunique(dropna=True)),
        }

        if pd.api.types.is_numeric_dtype(series):
            info["kind"] = "numerical"
            info["minimum"] = float(series.min()) if not series.dropna().empty else None
            info["maximum"] = float(series.max()) if not series.dropna().empty else None
            info["mean"] = float(series.mean()) if not series.dropna().empty else None
            info["median"] = float(series.median()) if not series.dropna().empty else None

        elif pd.api.types.is_datetime64_any_dtype(series):
            info["kind"] = "datetime"

        else:
            info["kind"] = "categorical_or_text"

            values = series.dropna().unique()[:5]
            info["example_values"] = values.tolist()

        column_info[column] = info

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "column_info": column_info,
    }