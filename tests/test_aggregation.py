
import pandas as pd
import pytest

from src.tools.aggregation import aggregate_column


@pytest.fixture
def dataset():
    return pd.DataFrame({
        "name": ["Asha", "Riya", "Aman", "Neha"],
        "marks": [80, 90, None, 100],
    })


@pytest.mark.parametrize(
    "operation, expected",
    [
        ("sum", 270.0),
        ("mean", 90.0),
        ("median", 90.0),
        ("min", 80.0),
        ("max", 100.0),
        ("count", 3),
    ]
)
def test_supported_operations(dataset, operation, expected):
    result = aggregate_column(
        dataset, "marks", operation
    )

    assert result["value"] == expected
    assert result["non_missing_rows"] == 3


def test_count_text_column(dataset):
    result = aggregate_column(dataset, "name", "count")

    assert result["value"] == 4


def test_invalid_column(dataset):
    with pytest.raises(ValueError):
        aggregate_column(dataset, "salary", "mean")


def test_invalid_operation(dataset):
    with pytest.raises(ValueError):
        aggregate_column(dataset, "marks", "multiply")


def test_non_numeric_column(dataset):
    with pytest.raises(ValueError):
        aggregate_column(dataset, "name", "mean")


def test_all_missing_numeric_column():
    df = pd.DataFrame({
        "marks": pd.Series([None, None], dtype="float64")
    })

    with pytest.raises(ValueError):
        aggregate_column(df, "marks", "mean")

    result = aggregate_column(df, "marks", "count")
    assert result["value"] == 0
