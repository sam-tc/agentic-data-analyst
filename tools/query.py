import pandas as pd


def query_dataset(
    df: pd.DataFrame,
    column: str | None = None,
    operator: str | None = None,
    value=None,
    conditions: list[dict] | None = None
) -> dict:

    if conditions is None:
        conditions = [
            {
                "column": column,
                "operator": operator,
                "value": value
            }
        ]

    if not conditions:
        return {
            "error": "At least one condition is required."
        }

    mask = pd.Series(True, index=df.index)

    for condition in conditions:
        condition_column = condition.get("column")
        condition_operator = condition.get("operator")
        condition_value = condition.get("value")

        if condition_column not in df.columns:
            return {
                "error": (
                    f"Column '{condition_column}' "
                    "does not exist in the dataset."
                )
            }

        if condition_operator not in {
            ">", "<", ">=", "<=", "==", "!="
        }:
            return {
                "error": (
                    f"Unsupported operator: "
                    f"{condition_operator}"
                )
            }

        series = df[condition_column]

        if condition_operator == ">":
            condition_mask = series > condition_value
        elif condition_operator == "<":
            condition_mask = series < condition_value
        elif condition_operator == ">=":
            condition_mask = series >= condition_value
        elif condition_operator == "<=":
            condition_mask = series <= condition_value
        elif condition_operator == "==":
            condition_mask = series == condition_value
        else:
            condition_mask = series != condition_value

        mask = mask & condition_mask

    result = df[mask]

    return {
        "rows": result.to_dict(orient="records"),
        "count": int(len(result))
    }


def top_n(
    df: pd.DataFrame,
    column: str,
    n: int,
    ascending: bool = False
) -> dict:

    if column not in df.columns:
        return {
            "error": f"Column '{column}' does not exist in the dataset."
        }

    if not pd.api.types.is_numeric_dtype(df[column]):
        return {
            "error": f"Column '{column}' must be numerical for ranking."
        }

    if n <= 0:
        return {
            "error": "n must be greater than 0."
        }

    result = df.sort_values(
        by=column,
        ascending=ascending
    ).head(n)

    return {
        "rows": result.to_dict(orient="records"),
        "count": int(len(result))
    }


def execute_query(df: pd.DataFrame, query: dict) -> dict:
    """
    Execute a structured query against the dataset.
    """

    operation = query.get("operation")

    if operation == "filter":
        return query_dataset(
            df,
            column=query.get("column"),
            operator=query.get("operator"),
            value=query.get("value"),
            conditions=query.get("conditions")
        )

    if operation == "top_n":
        return top_n(
            df,
            column=query.get("column"),
            n=query.get("n"),
            ascending=query.get("ascending", False)
        )

    return {
        "error": f"Unsupported query operation: {operation}"
    }