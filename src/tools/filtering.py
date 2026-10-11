import pandas as pd


def filter_rows(
    df: pd.DataFrame,
    column: str,
    operator: str,
    value
) -> pd.DataFrame:
    """Filter dataset rows using a specified condition."""

    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist.")

    operations = {
        "==": lambda s, v: s.eq(v),
        "!=": lambda s, v: s.ne(v),
        ">": lambda s, v: s.gt(v),
        ">=": lambda s, v: s.ge(v),
        "<": lambda s, v: s.lt(v),
        "<=": lambda s, v: s.le(v),
    }

    if operator not in operations and operator != "contains":
        raise ValueError(f"Unsupported operator: {operator}")

    series = df[column]

    try:
        if operator == "contains":
            mask = series.astype("string").str.contains(
                str(value),
                case=False,
                regex=False,
                na=False
            )
        else:
            mask = operations[operator](series, value)

    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Cannot compare column '{column}' with {value!r}"
        ) from error

    return df.loc[mask.fillna(False)].copy()
