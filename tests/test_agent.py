
from types import SimpleNamespace

import pandas as pd
import pytest

from src import agent
from src.agent_tools import build_tools


@pytest.fixture
def sales_dataset():
    """A dataset different from our student example."""

    return pd.DataFrame({
        "region": ["Delhi", "Mumbai", "Delhi", "Jaipur", "Mumbai"],
        "revenue": [1200, 600, 1800, 900, 2000],
    })


def test_six_tools_registered(sales_dataset, tmp_path):
    trace = []

    functions = build_tools(
        sales_dataset, trace, output_dir=tmp_path
    )

    names = {function.__name__ for function in functions}

    assert names == {
        "dataset_profile",
        "row_filter",
        "column_aggregation",
        "group_comparison",
        "correlation_analysis",
        "plot_chart",
    }


def test_real_tool_execution_and_trace(sales_dataset, tmp_path):
    trace = []

    tools = build_tools(
        sales_dataset, trace, output_dir=tmp_path
    )

    registry = {func.__name__: func for func in tools}

    result = registry["group_comparison"](
        "region", "sum", "revenue"
    )

    assert result["results"][0] == {
        "group": "Delhi",
        "value": 3000.0,
    }

    assert len(trace) == 1
    assert trace[0]["tool"] == "group_comparison"
    assert trace[0]["result"] == result


def test_invalid_tool_arguments_are_recorded(sales_dataset, tmp_path):
    trace = []

    tools = build_tools(
        sales_dataset, trace, output_dir=tmp_path
    )

    registry = {func.__name__: func for func in tools}

    result = registry["column_aggregation"](
        "unknown_column", "mean"
    )

    assert "error" in result
    assert trace[0]["tool"] == "column_aggregation"
    assert trace[0]["result"] == result


def test_chart_tool_creates_file(sales_dataset, tmp_path):
    trace = []

    tools = build_tools(
        sales_dataset, trace, output_dir=tmp_path
    )

    registry = {func.__name__: func for func in tools}

    result = registry["plot_chart"]("bar", "region")

    assert "error" not in result
    assert result["chart_type"] == "bar"
    assert result["rows_plotted"] == 5

    chart_path = tmp_path / "chart_1.html"

    assert chart_path.exists()
    assert trace[0]["tool"] == "plot_chart"


def test_agent_executes_tool_using_fake_gemini(
    sales_dataset, monkeypatch
):
    """Simulate Gemini calling our actual group tool."""

    monkeypatch.setenv("GEMINI_API_KEY", "dummy-test-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")

    registered = {}

    original_builder = agent.build_tools

    def capture_tools(df, trace):
        functions = original_builder(df, trace)

        registered.update({
            func.__name__: func
            for func in functions
        })

        return functions

    monkeypatch.setattr(
        agent, "build_tools", capture_tools
    )

    def fake_create(*, model, config):
        assert model == "gemini-3.8-flash"
        assert len(config.tools) == 6

        def fake_send_message(prompt):
            assert "region" in prompt

            result = registered["group_comparison"](
                "region", "sum", "revenue"
            )

            highest = result["results"][0]

            return SimpleNamespace(
                text=(
                    f"{highest['group']} has the highest "
                    f"revenue: {highest['value']}."
                )
            )

        return SimpleNamespace(
            send_message=fake_send_message
        )

    fake_client = SimpleNamespace(
        chats=SimpleNamespace(create=fake_create)
    )

    monkeypatch.setattr(
        agent.genai,
        "Client",
        lambda api_key, **kwargs: fake_client
    )

    answer, trace = agent.ask_agent(
        sales_dataset,
        "Which region generated the most revenue?"
    )

    assert "Delhi" in answer
    assert "3000" in answer
    assert len(trace) == 1
    assert trace[0]["tool"] == "group_comparison"


def test_agent_rejects_answer_without_tool(
    sales_dataset, monkeypatch
):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-test-key")

    fake_chat = SimpleNamespace(
        send_message=lambda prompt: SimpleNamespace(
            text="Delhi has the highest revenue."
        )
    )

    fake_client = SimpleNamespace(
        chats=SimpleNamespace(
            create=lambda **kwargs: fake_chat
        )
    )

    monkeypatch.setattr(
        agent.genai,
        "Client",
        lambda api_key, **kwargs: fake_client
    )

    with pytest.raises(
        RuntimeError,
        match="did not execute"
    ):
        agent.ask_agent(
            sales_dataset,
            "Which region has the highest revenue?"
        )


def test_agent_validates_inputs(sales_dataset, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-test-key")

    with pytest.raises(ValueError, match="no rows"):
        agent.ask_agent(
            pd.DataFrame({"x": []}),
            "Analyze this dataset."
        )

    with pytest.raises(ValueError, match="question"):
        agent.ask_agent(sales_dataset, "   ")


def test_missing_api_key(sales_dataset, monkeypatch):
    monkeypatch.delenv(
        "GEMINI_API_KEY",
        raising=False
    )

    # Prevent loading the real key from .env.
    monkeypatch.setattr(
        agent,
        "load_dotenv",
        lambda: None
    )

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        agent.ask_agent(
            sales_dataset,
            "Analyze this dataset."
        )
