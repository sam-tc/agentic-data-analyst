import pandas as pd

def describe_correlation(correlation: float) -> dict:
    """
    Describe the direction and strength of a correlation.
    """

    if correlation > 0:
        direction = "positive"
    elif correlation < 0:
        direction = "negative"
    else:
        direction = "none"

    absolute_correlation = abs(correlation)

    if absolute_correlation < 0.3:
        strength = "weak"
    elif absolute_correlation < 0.7:
        strength = "moderate"
    else:
        strength = "strong"

    return {
        "direction": direction,
        "strength": strength,
    }


def analyze_relationship(
    df: pd.DataFrame,
    column_x: str,
    column_y: str
) -> dict:
    """
    Analyze the relationship between two numerical columns.
    """

    if column_x not in df.columns:
        return {
            "error": f"Column '{column_x}' does not exist in the dataset."
        }

    if column_y not in df.columns:
        return {
            "error": f"Column '{column_y}' does not exist in the dataset."
        }

    if not pd.api.types.is_numeric_dtype(df[column_x]):
        return {
            "error": f"Column '{column_x}' must be numerical."
        }

    if not pd.api.types.is_numeric_dtype(df[column_y]):
        return {
            "error": f"Column '{column_y}' must be numerical."
        }

    paired_data = df[[column_x, column_y]].dropna()

    if len(paired_data) < 2:
        return {
            "error": "At least two valid paired observations are required."
        }

    if paired_data[column_x].nunique() < 2:
        return {
            "error": (
                f"Column '{column_x}' has no variation, "
                "so correlation cannot be calculated."
            )
        }

    if paired_data[column_y].nunique() < 2:
        return {
            "error": (
                f"Column '{column_y}' has no variation, "
                "so correlation cannot be calculated."
            )
        }

    correlation = paired_data[column_x].corr(
        paired_data[column_y]
    )

    description = describe_correlation(correlation)

    return {
        "column_x": column_x,
        "column_y": column_y,
        "correlation": float(correlation),
        "direction": description["direction"],
        "strength": description["strength"],
        "observations": int(len(paired_data))
    }


def discover_relationships(df: pd.DataFrame) -> dict:
    """
    Discover relationships between all numerical columns.
    """

    numeric_df = df.select_dtypes(include="number")

    if len(numeric_df.columns) < 2:
        return {
            "error": (
                "At least two numerical columns are required "
                "to discover relationships."
            )
        }

    correlation_matrix = numeric_df.corr()

    relationships = []

    for i, column_x in enumerate(correlation_matrix.columns):
        for column_y in correlation_matrix.columns[i + 1:]:
            correlation = correlation_matrix.loc[
                column_x, column_y
            ]

            if pd.isna(correlation):
                continue

            description = describe_correlation(correlation)

            relationships.append({
                "column_x": column_x,
                "column_y": column_y,
                "correlation": float(correlation),
                "direction": description["direction"],
                "strength": description["strength"],
            })

    relationships.sort(
        key=lambda relationship: abs(
            relationship["correlation"]
        ),
        reverse=True
    )

    return {
        "relationships": relationships
    }