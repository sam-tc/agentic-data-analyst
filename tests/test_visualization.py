
import json

import pandas as pd
import pytest
import plotly.graph_objects as go

from src.tools.visualization import create_visualization


@pytest.fixture
def dataset():
    return pd.DataFrame({
        "department": ["CSE", "IT", "CSE"],
        "hours": [1, 2, 3],
        "marks": [40, 60, 80],
    })


def test_bar_chart(dataset):
    result = create_visualization(
        dataset, "bar", "department"
    )

    figure = go.Figure(result["figure"])

    assert figure.data[0].type == "bar"
    assert list(figure.data[0].x) == ["CSE", "IT"]
    assert list(figure.data[0].y) == [2, 1]
    assert result["rows_plotted"] == 3


def test_missing_category():
    df = pd.DataFrame({
        "department": ["CSE", None, "CSE"]
    })

    result = create_visualization(
        df, "bar", "department"
    )

    assert result["rows_plotted"] == 3


def test_histogram(dataset):
    result = create_visualization(
        dataset, "histogram", "marks"
    )

    figure = go.Figure(result["figure"])

    assert figure.data[0].type == "histogram"
    assert result["rows_plotted"] == 3


def test_scatter_with_missing_values():
    df = pd.DataFrame({
        "hours": [1, 2, 3],
        "marks": [40, None, 80],
    })

    result = create_visualization(
        df, "scatter", "hours", "marks"
    )

    assert result["rows_plotted"] == 2


def test_line_chart_sorted(dataset):
    result = create_visualization(
        dataset, "line", "marks", "hours"
    )

    figure = go.Figure(result["figure"])

    assert figure.data[0].mode == "lines+markers"
    assert list(figure.data[0].x) == [40, 60, 80]


def test_invalid_chart_type(dataset):
    with pytest.raises(ValueError):
        create_visualization(dataset, "pie", "department")


def test_invalid_column(dataset):
    with pytest.raises(ValueError):
        create_visualization(dataset, "bar", "salary")


def test_missing_y_column(dataset):
    with pytest.raises(ValueError):
        create_visualization(dataset, "scatter", "hours")


def test_non_numeric_scatter(dataset):
    with pytest.raises(ValueError):
        create_visualization(
            dataset, "scatter", "department", "marks"
        )


def test_empty_dataset():
    df = pd.DataFrame({"marks": []})

    with pytest.raises(ValueError):
        create_visualization(df, "histogram", "marks")


def test_chart_json_serializable(dataset):
    result = create_visualization(
        dataset, "scatter", "hours", "marks"
    )

    assert json.dumps(result)
