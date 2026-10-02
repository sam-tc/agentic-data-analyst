import pandas as pd


def statistical_summary(df: pd.DataFrame) -> dict:
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        return {
            "message": "The dataset does not contain any numerical columns."
        }

    results = {}

    for column in numeric_df.columns:
        series = numeric_df[column].dropna()

        if series.empty:
            results[column] = {
                "message": "This column does not contain any valid numerical values."
            }
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        results[column] = {
            "count": int(series.count()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "minimum": float(series.min()),
            "maximum": float(series.max()),
            "standard_deviation": float(series.std()),
            "25th_percentile": float(q1),
            "75th_percentile": float(q3),
            "interquartile_range": float(q3 - q1),
            "skewness": float(series.skew()),
        }

    return results