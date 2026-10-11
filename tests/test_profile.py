
import pandas as pd

from src.tools.profile import profile_dataset


def test_profile_mixed_dataset():
    df = pd.DataFrame({
        "department": ["CSE", "IT", "CSE"],
        "marks": [80, 90, None],
    })

    result = profile_dataset(df)

    assert result["row_count"] == 3
    assert result["column_count"] == 2

    columns = {
        column["name"]: column
        for column in result["columns"]
    }

    assert columns["department"]["unique_values"] == 2
    assert columns["marks"]["missing_values"] == 1
    assert columns["marks"]["statistics"]["mean"] == 85.0


def test_profile_text_only_dataset():
    df = pd.DataFrame({
        "city": ["Delhi", "Mumbai", "Delhi"],
    })

    result = profile_dataset(df)

    assert result["row_count"] == 3
    assert result["column_count"] == 1
    assert "statistics" not in result["columns"][0]


def test_profile_all_missing_numeric_values():
    df = pd.DataFrame({
        "score": pd.Series([float("nan"), float("nan")])
    })

    result = profile_dataset(df)

    assert result["columns"][0]["missing_values"] == 2
    assert "statistics" not in result["columns"][0]
