
from pathlib import Path
import pandas as pd


def load_csv(file):
    """Load a CSV dataset into a Pandas DataFrame."""

    try:
        df = pd.read_csv(file)

    except (pd.errors.EmptyDataError,
            pd.errors.ParserError) as error:
        raise ValueError(f"Invalid CSV file: {error}") from error

    if df.empty:
        raise ValueError("The dataset contains no rows.")

    return df
