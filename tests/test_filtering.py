
import pandas as pd
import pytest

from src.tools.filtering import filter_rows


@pytest.fixture
def students():
    return pd.DataFrame({
        "name": ["Asha", "Riya", "Aman", "Neha"],
        "department": ["CSE", "IT", "CSE", "ECE"],
        "marks": [88, 72, 91, 65],
    })


def test_numeric_filter(students):
    result = filter_rows(students, "marks", ">", 80)

    assert result["name"].tolist() == ["Asha", "Aman"]


def test_text_equality(students):
    result = filter_rows(
        students, "department", "==", "CSE"
    )

    assert len(result) == 2


def test_text_contains(students):
    result = filter_rows(
        students, "name", "contains", "ASH"
    )

    assert result["name"].tolist() == ["Asha"]


def test_missing_values():
    df = pd.DataFrame({
        "marks": [90, None, 50]
    })

    result = filter_rows(df, "marks", ">", 60)

    assert len(result) == 1
    assert result["marks"].iloc[0] == 90


def test_invalid_column(students):
    with pytest.raises(ValueError):
        filter_rows(students, "salary", ">", 50000)


def test_invalid_operator(students):
    with pytest.raises(ValueError):
        filter_rows(students, "marks", "approximately", 80)


def test_incompatible_comparison(students):
    with pytest.raises(ValueError):
        filter_rows(students, "marks", ">", "high")
