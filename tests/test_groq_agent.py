"""Test Groq function calling without using the actual API."""

from types import SimpleNamespace

import pandas as pd
import pytest

from src import groq_agent


def response(content=None, calls=None):
    message = SimpleNamespace(
        content=content,
        tool_calls=calls,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)]
    )


def call(name, arguments, identifier="call_1"):
    return SimpleNamespace(
        id=identifier,
        function=SimpleNamespace(
            name=name,
            arguments=arguments,
        ),
    )


def mock_client(monkeypatch, responses):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")

    requests = []
    remaining = iter(responses)

    def create(**kwargs):
        requests.append(kwargs)
        return next(remaining)

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=create
            )
        )
    )

    monkeypatch.setattr(
        groq_agent,
        "Groq",
        lambda **kwargs: client,
    )

    return requests


def test_groq_executes_real_group_tool(monkeypatch):
    df = pd.DataFrame({
        "department": ["CSE", "CSE", "IT"],
        "marks": [80, 100, 70],
    })

    requests = mock_client(monkeypatch, [
        response(calls=[
            call(
                "group_comparison",
                '{"group_column":"department",'
                '"operation":"mean","value_column":"marks"}'
            )
        ]),
        response(
            content="CSE has the highest average at 90."
        ),
    ])

    answer, trace = groq_agent.ask_groq_agent(
        df,
        "Highest department average?",
        save_chart_files=False,
    )

    assert "90" in answer
    assert trace[0]["tool"] == "group_comparison"

    assert trace[0]["result"]["results"][0] == {
        "group": "CSE",
        "value": 90.0,
    }

    assert requests[0]["tool_choice"] == "required"
    assert requests[1]["tool_choice"] == "auto"
    assert len(requests[0]["tools"]) == 6

    assert (
        requests[1]["messages"][-1]["role"]
        == "tool"
    )


def test_groq_chart_receipt_does_not_write_html(
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)

    requests = mock_client(monkeypatch, [
        response(calls=[
            call(
                "plot_chart",
                '{"chart_type":"bar",'
                '"x_column":"region","y_column":""}'
            )
        ]),
        response(
            content="Bar chart generated."
        ),
    ])

    df = pd.DataFrame({
        "region": ["Delhi", "Delhi", "Mumbai"]
    })

    _, trace = groq_agent.ask_groq_agent(
        df,
        "Plot order counts by region",
        save_chart_files=False,
    )

    assert trace[0]["tool"] == "plot_chart"
    assert "error" not in trace[0]["result"]
    assert "chart_file" not in trace[0]["result"]
    assert not (tmp_path / "outputs").exists()

    assert (
        requests[1]["messages"][-1]["tool_call_id"]
        == "call_1"
    )


def test_groq_refuses_when_no_tool_executed(
    monkeypatch,
):
    mock_client(monkeypatch, [
        response(content="The answer is 100.")
    ])

    df = pd.DataFrame({
        "revenue": [1, 2]
    })

    with pytest.raises(
        RuntimeError,
        match="did not execute",
    ):
        groq_agent.ask_groq_agent(
            df,
            "What is the revenue?",
            save_chart_files=False,
        )


def test_groq_refuses_when_all_tools_fail(
    monkeypatch,
):
    mock_client(monkeypatch, [
        response(calls=[
            call(
                "column_aggregation",
                '{"column":"unknown",'
                '"operation":"mean"}'
            )
        ]),
        response(
            content="The mean is 42."
        ),
    ])

    df = pd.DataFrame({
        "revenue": [1, 2]
    })

    with pytest.raises(
        RuntimeError,
        match="All analytical tool calls failed",
    ):
        groq_agent.ask_groq_agent(
            df,
            "What is the average unknown?",
            save_chart_files=False,
        )


def test_bar_chart_tool_description_explains_category_counts():
    plot_spec = next(
        spec["function"]
        for spec in groq_agent.TOOL_SPECS
        if spec["function"]["name"] == "plot_chart"
    )

    description = plot_spec["description"]
    y_description = (
        plot_spec["parameters"]["properties"]["y_column"]["description"]
    )

    assert "automatically counts" in description
    assert "empty string" in y_description
    assert "bar" in y_description
