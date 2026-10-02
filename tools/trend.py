import pandas as pd


def analyze_trend(
    df: pd.DataFrame,
    time_column: str,
    value_column: str
) -> dict:
    """
    Analyze how a numerical value changes over time.
    """

    # Check that the time column exists.
    if time_column not in df.columns:
        return {
            "error": (
                f"Time column '{time_column}' "
                "does not exist in the dataset."
            )
        }

    # Check that the value column exists.
    if value_column not in df.columns:
        return {
            "error": (
                f"Value column '{value_column}' "
                "does not exist in the dataset."
            )
        }

    # Check that the value column is numerical.
    if not pd.api.types.is_numeric_dtype(df[value_column]):
        return {
            "error": (
                f"Value column '{value_column}' "
                "must be numerical."
            )
        }

    # Convert the time column into datetime values.
    time_values = pd.to_datetime(
        df[time_column],
        errors="coerce"
    )

    # Create a working DataFrame.
    trend_data = df.copy()
    trend_data["_parsed_time"] = time_values

    # Remove rows with invalid dates or missing values.
    trend_data = trend_data.dropna(
        subset=["_parsed_time", value_column]
    )

    if trend_data.empty:
        return {
            "error": (
                "No valid time and numerical value pairs "
                "were found."
            )
        }

    # Sort the data chronologically.
    trend_data = trend_data.sort_values(
        by="_parsed_time"
    )

    # Make sure there are at least two observations.
    if len(trend_data) < 2:
        return {
            "error": (
                "At least two valid observations "
                "are required for trend analysis."
            )
        }

    # Get the first and last values.
    start_value = trend_data[value_column].iloc[0]
    end_value = trend_data[value_column].iloc[-1]

    # Calculate the absolute change.
    absolute_change = end_value - start_value

    # Calculate percentage change.
    if start_value != 0:
        percentage_change = (
            absolute_change / abs(start_value)
        ) * 100
    else:
        percentage_change = None

    # Determine the overall direction.
    if absolute_change > 0:
        direction = "increasing"
    elif absolute_change < 0:
        direction = "decreasing"
    else:
        direction = "stable"

    # Find the highest and lowest periods.
    highest_row = trend_data.loc[
        trend_data[value_column].idxmax()
    ]

    lowest_row = trend_data.loc[
        trend_data[value_column].idxmin()
    ]

    return {
        "time_column": time_column,
        "value_column": value_column,
        "trend": direction,
        "start_time": str(
            trend_data["_parsed_time"].iloc[0]
        ),
        "end_time": str(
            trend_data["_parsed_time"].iloc[-1]
        ),
        "start_value": float(start_value),
        "end_value": float(end_value),
        "absolute_change": float(absolute_change),
        "percentage_change": (
            float(percentage_change)
            if percentage_change is not None
            else None
        ),
        "highest_period": str(
            highest_row["_parsed_time"]
        ),
        "highest_value": float(
            highest_row[value_column]
        ),
        "lowest_period": str(
            lowest_row["_parsed_time"]
        ),
        "lowest_value": float(
            lowest_row[value_column]
        ),
        "observations": int(len(trend_data))
    }