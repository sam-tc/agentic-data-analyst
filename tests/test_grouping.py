
import pandas as pd
import pytest

from src.tools.grouping import compare_groups


@pytest.fixture
def students():
    return pd.DataFrame({
        "name": ["Asha", "Riya", "Aman", "Neha", "Kabir"],
        "department": ["CSE", "IT", "CSE", "IT", "ECE"],
        "marks": [80, 70, 100, 90, 85],
    })


def as_dictionary(result):
    return {
        row["group"]: row["value"]
        for row in result["results"]
    }


def test_grouped_mean(students):
    result = compare_groups(
        students, "department", "mean", "marks"
    )

    assert as_dictionary(result) == {
        "CSE": 90.0,
        "ECE": 85.0,
        "IT": 80.0,
    }
    assert result["results"][0]["group"] == "CSE"


def test_grouped_count(students):
    result = compare_groups(
        students, "department", "count"
    )

    assert as_dictionary(result) == {
        "CSE": 2,
        "IT": 2,
        "ECE": 1,
    }


def test_grouped_sum(students):
    result = compare_groups(
        students, "department", "sum", "marks"
    )

    assert as_dictionary(result) == {
        "CSE": 180.0,
        "IT": 160.0,
        "ECE": 85.0,
    }


def test_missing_numeric_values():
    df = pd.DataFrame({
        "department": ["CSE", "IT", "ECE"],
        "marks": [80, None, None],
    })

    result = compare_groups(
        df, "department", "sum", "marks"
    )

    assert as_dictionary(result) == {
        "CSE": 80.0,
        "IT": None,
        "ECE": None,
    }


def test_missing_group_labels():
    df = pd.DataFrame({
        "department": ["CSE", None, "CSE"],
        "marks": [80, 90, 100],
    })

    result = compare_groups(
        df, "department", "count"
    )

    assert as_dictionary(result) == {
        "CSE": 2,
        None: 1,
    }


def test_invalid_group_column(students):
    with pytest.raises(ValueError):
        compare_groups(students, "city", "count")


def test_missing_value_column(students):
    with pytest.raises(ValueError):
        compare_groups(students, "department", "mean")


def test_unknown_value_column(students):
    with pytest.raises(ValueError):
        compare_groups(
            students, "department", "mean", "salary"
        )


def test_non_numeric_value_column(students):
    with pytest.raises(ValueError):
        compare_groups(
            students, "department", "mean", "name"
        )


def test_invalid_operation(students):
    with pytest.raises(ValueError):
        compare_groups(
            students, "department", "multiply", "marks"
        )
