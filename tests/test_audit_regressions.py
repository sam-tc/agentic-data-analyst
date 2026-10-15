"""Regression tests for correctness and production side effects."""

from pathlib import Path

import pandas as pd
import pytest

from src.agent_tools import build_tools
from src.tools.aggregation import aggregate_column
from src.tools.grouping import compare_groups
from src.tools.profile import profile_dataset


def registry(df, output_dir=None):
    events = []
    tools = build_tools(df, events, output_dir=output_dir)
    return {tool.__name__: tool for tool in tools}, events


def test_large_integer_filter_is_exact():
    df = pd.DataFrame({
        "id": [9007199254740992, 9007199254740993]
    })
    tools, trace = registry(df)

    result = tools["row_filter"](
        "id", "==", "9007199254740993"
    )

    assert result["rows_matched"] == 1
    assert result["rows"][0]["id"] == 9007199254740993
    assert len(trace) == 1


def test_profile_ignores_infinite_values():
    df = pd.DataFrame({
        "value": [1.0, float("inf"), float("-inf"), 3.0]
    })

    stats = profile_dataset(df)["columns"][0]["statistics"]

    assert stats == {
        "minimum": 1.0,
        "maximum": 3.0,
        "mean": 2.0,
    }


def test_aggregation_excludes_infinite_values():
    df = pd.DataFrame({
        "value": [1.0, float("inf"), 3.0]
    })

    result = aggregate_column(df, "value", "mean")

    assert result["value"] == 2.0
    assert result["rows_used"] == 2
    assert result["non_finite_excluded"] == 1
    assert result["non_missing_rows"] == 3


def test_aggregation_rejects_overflow():
    df = pd.DataFrame({
        "value": [1e308, 1e308]
    })

    with pytest.raises(
        ValueError,
        match="finite numeric range"
    ):
        aggregate_column(df, "value", "sum")


def test_grouping_excludes_infinite_values():
    df = pd.DataFrame({
        "team": ["A", "A", "B"],
        "value": [4.0, float("inf"), 2.0],
    })

    results = compare_groups(
        df, "team", "mean", "value"
    )["results"]

    assert {
        item["group"]: item["value"]
        for item in results
    } == {"A": 4.0, "B": 2.0}


def test_grouping_returns_null_for_group_without_finite_values():
    df = pd.DataFrame({
        "team": ["A", "B"],
        "value": [float("inf"), 2.0],
    })

    results = compare_groups(
        df, "team", "mean", "value"
    )["results"]

    assert {
        item["group"]: item["value"]
        for item in results
    } == {"A": None, "B": 2.0}


def test_web_chart_does_not_write_server_files(
    tmp_path,
    monkeypatch
):
    monkeypatch.chdir(tmp_path)

    df = pd.DataFrame({
        "region": ["X", "Y"]
    })

    tools, events = registry(df, output_dir=None)
    result = tools["plot_chart"]("bar", "region")

    assert "error" not in result
    assert "chart_file" not in result
    assert not (tmp_path / "outputs").exists()
    assert events[0]["tool"] == "plot_chart"


def test_cli_chart_still_writes_html(tmp_path):
    df = pd.DataFrame({
        "region": ["X", "Y"]
    })

    tools, _ = registry(df, output_dir=tmp_path)
    result = tools["plot_chart"]("bar", "region")

    assert Path(result["chart_file"]).exists()
