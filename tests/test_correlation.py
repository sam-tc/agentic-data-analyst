
import pandas as pd
import pytest

from src.tools.correlation import calculate_correlation


def test_positive_correlation():
    df = pd.DataFrame({
        "hours": [1, 2, 3, 4],
        "marks": [20, 40, 60, 80],
    })

    result = calculate_correlation(df, "hours", "marks")

    assert result["correlation"] == pytest.approx(1.0)
    assert result["rows_used"] == 4


def test_negative_correlation():
    df = pd.DataFrame({
        "x": [1, 2, 3, 4],
        "y": [4, 3, 2, 1],
    })

    result = calculate_correlation(df, "x", "y")

    assert result["correlation"] == pytest.approx(-1.0)


def test_no_linear_correlation():
    df = pd.DataFrame({
        "x": [1, 2, 3],
        "y": [1, 0, 1],
    })

    result = calculate_correlation(df, "x", "y")

    assert result["correlation"] == pytest.approx(
        0.0, abs=1e-12
    )


def test_missing_pairs():
    df = pd.DataFrame({
        "x": [1, 2, 3, 4],
        "y": [10, None, 30, 40],
    })

    result = calculate_correlation(df, "x", "y")

    assert result["rows_used"] == 3
    assert result["rows_excluded"] == 1
    assert result["correlation"] == pytest.approx(1.0)


def test_infinite_values():
    df = pd.DataFrame({
        "x": [1, 2, 3, 4],
        "y": [10, float("inf"), 30, 40],
    })

    result = calculate_correlation(df, "x", "y")

    assert result["rows_used"] == 3
    assert result["rows_excluded"] == 1


def test_invalid_column():
    df = pd.DataFrame({"x": [1, 2, 3]})

    with pytest.raises(ValueError):
        calculate_correlation(df, "x", "y")


def test_non_numeric_column():
    df = pd.DataFrame({
        "x": [1, 2, 3],
        "y": ["A", "B", "C"],
    })

    with pytest.raises(ValueError):
        calculate_correlation(df, "x", "y")


def test_boolean_column():
    df = pd.DataFrame({
        "x": [1, 2, 3],
        "y": [True, False, True],
    })

    with pytest.raises(ValueError):
        calculate_correlation(df, "x", "y")


def test_same_column():
    df = pd.DataFrame({"x": [1, 2, 3]})

    with pytest.raises(ValueError):
        calculate_correlation(df, "x", "x")


def test_insufficient_rows():
    df = pd.DataFrame({
        "x": [1, 2, 3],
        "y": [10, None, None],
    })

    with pytest.raises(ValueError):
        calculate_correlation(df, "x", "y")


def test_constant_column():
    df = pd.DataFrame({
        "x": [1, 1, 1],
        "y": [10, 20, 30],
    })

    with pytest.raises(ValueError):
        calculate_correlation(df, "x", "y")
