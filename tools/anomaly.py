import pandas as pd


def detect_anomalies(
    df: pd.DataFrame,
    column: str
) -> dict:
    """
    Detect unusual values in a numerical column using the IQR method.
    """

    # Check whether the requested column exists.
    if column not in df.columns:
        return {
            "error": f"Column '{column}' does not exist in the dataset."
        }

    # Check whether the column is numerical.
    if not pd.api.types.is_numeric_dtype(df[column]):
        return {
            "error": f"Column '{column}' must be numerical."
        }

    # Remove rows where the selected column is missing.
    valid_data = df[df[column].notna()].copy()

    if valid_data.empty:
        return {
            "error": (
                f"Column '{column}' does not contain "
                "valid numerical values."
            )
        }

    # Calculate the first quartile and third quartile.
    q1 = valid_data[column].quantile(0.25)
    q3 = valid_data[column].quantile(0.75)

    # Calculate the interquartile range.
    iqr = q3 - q1

    # If there is no spread, meaningful anomalies cannot be detected.
    if iqr == 0:
        return {
            "column": column,
            "method": "IQR",
            "message": (
                f"Column '{column}' has no meaningful spread "
                "for IQR-based anomaly detection."
            ),
            "anomaly_count": 0,
            "anomalies": []
        }

    # Calculate the lower and upper boundaries.
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Find values outside the boundaries.
    anomaly_mask = (
        (valid_data[column] < lower_bound)
        | (valid_data[column] > upper_bound)
    )

    anomalies = valid_data[anomaly_mask]

    # Return structured results.
    return {
        "column": column,
        "method": "IQR",
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "anomaly_count": int(len(anomalies)),
        "anomalies": anomalies.to_dict(orient="records")
    }


def detect_all_anomalies(df: pd.DataFrame) -> dict:
    """
    Detect anomalies across all numerical columns.
    """

    # Find all numerical columns automatically.
    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    if len(numeric_columns) == 0:
        return {
            "error": "The dataset does not contain any numerical columns."
        }

    results = {}

    # Analyze each numerical column.
    for column in numeric_columns:
        results[column] = detect_anomalies(df, column)

    return {
        "results": results
    }