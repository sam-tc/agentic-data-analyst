"""Check that Gemini and Groq use identical Python calculations.

The AI responses are simulated, but the analytical tools are real.
No external API requests are made.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src import agent, groq_agent


SAMPLE_DIR = Path(__file__).resolve().parents[1] / "sample_data"


def run_fake_gemini(monkeypatch, df, tool_name, arguments):
    """Simulate Gemini calling a real analytical tool."""

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    registered = {}
    original_build_tools = agent.build_tools

    def capture_tools(dataframe, trace, **kwargs):
        functions = original_build_tools(
            dataframe,
            trace,
            output_dir=None,
        )
        registered.update({
            function.__name__: function
            for function in functions
        })
        return functions

    def fake_create(*, model, config):
        def fake_send_message(prompt):
            registered[tool_name](**arguments)
            return SimpleNamespace(text="Calculation completed.")

        return SimpleNamespace(
            send_message=fake_send_message
        )

    fake_client = SimpleNamespace(
        chats=SimpleNamespace(create=fake_create)
    )

    monkeypatch.setattr(
        agent,
        "build_tools",
        capture_tools,
    )
    monkeypatch.setattr(
        agent.genai,
        "Client",
        lambda **kwargs: fake_client,
    )

    _, trace = agent.ask_agent(
        df,
        "Analyze this dataset.",
        save_chart_files=False,
    )

    return trace


def run_fake_groq(monkeypatch, df, tool_name, arguments):
    """Simulate Groq calling the same real analytical tool."""

    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    tool_call = SimpleNamespace(
        id="consistency_call_1",
        function=SimpleNamespace(
            name=tool_name,
            arguments=json.dumps(arguments),
        ),
    )

    responses = iter([
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        tool_calls=[tool_call],
                    )
                )
            ]
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Calculation completed.",
                        tool_calls=[],
                    )
                )
            ]
        ),
    ])

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: next(responses)
            )
        )
    )

    monkeypatch.setattr(
        groq_agent,
        "Groq",
        lambda **kwargs: fake_client,
    )

    _, trace = groq_agent.ask_groq_agent(
        df,
        "Analyze this dataset.",
        save_chart_files=False,
    )

    return trace


@pytest.mark.parametrize(
    "filename,tool_name,arguments",
    [
        (
            "students.csv",
            "group_comparison",
            {
                "group_column": "department",
                "operation": "mean",
                "value_column": "marks",
            },
        ),
        (
            "sales.csv",
            "plot_chart",
            {
                "chart_type": "bar",
                "x_column": "region",
                "y_column": "",
            },
        ),
    ],
)
def test_gemini_and_groq_tool_results_match(
    monkeypatch,
    filename,
    tool_name,
    arguments,
):
    df = pd.read_csv(SAMPLE_DIR / filename)

    gemini_trace = run_fake_gemini(
        monkeypatch,
        df,
        tool_name,
        arguments,
    )

    groq_trace = run_fake_groq(
        monkeypatch,
        df,
        tool_name,
        arguments,
    )

    assert len(gemini_trace) == 1
    assert len(groq_trace) == 1

    assert gemini_trace[0]["tool"] == tool_name
    assert groq_trace[0]["tool"] == tool_name

    assert "error" not in gemini_trace[0]["result"]
    assert "error" not in groq_trace[0]["result"]

    assert (
        gemini_trace[0]["result"]
        == groq_trace[0]["result"]
    )

    if tool_name == "group_comparison":
        results = gemini_trace[0]["result"]["results"]

        assert results[0] == {
            "group": "CSE",
            "value": 90.0,
        }

    if tool_name == "plot_chart":
        result = groq_trace[0]["result"]

        assert result["chart_type"] == "bar"
        assert result["x_column"] == "region"
        assert result["rows_plotted"] == 5
