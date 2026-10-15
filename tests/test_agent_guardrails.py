from types import SimpleNamespace

import pandas as pd
import pytest

from src import agent


@pytest.fixture
def dataset():
    return pd.DataFrame({
        "region": ["Delhi", "Mumbai", "Delhi"],
        "revenue": [1200, 600, 1800],
    })


def test_agent_rejects_all_failed_tools(dataset, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    captured = {}

    def fake_build_tools(df, trace):
        captured["trace"] = trace
        return [lambda: {}]

    def fake_send_message(prompt):
        captured["trace"].append({
            "tool": "column_aggregation",
            "arguments": {
                "column": "missing",
                "operation": "mean",
            },
            "result": {
                "error": "Column does not exist."
            },
        })

        return SimpleNamespace(
            text="The average is 123."
        )

    fake_client = SimpleNamespace(
        chats=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(
                send_message=fake_send_message
            )
        )
    )

    monkeypatch.setattr(
        agent, "build_tools", fake_build_tools
    )

    monkeypatch.setattr(
        agent.genai,
        "Client",
        lambda api_key, **kwargs: fake_client
    )

    with pytest.raises(
        RuntimeError,
        match="All analytical tool calls failed"
    ):
        agent.ask_agent(
            dataset,
            "What is the average of missing?"
        )


def test_streamlit_mode_disables_chart_exports(
    dataset, monkeypatch
):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    captured = {}

    def fake_build_tools(df, trace, output_dir=None):
        captured["output_dir"] = output_dir
        captured["trace"] = trace
        return [lambda: {}]

    def fake_send_message(prompt):
        captured["trace"].append({
            "tool": "dataset_profile",
            "arguments": {},
            "result": {
                "row_count": 3,
                "column_count": 2,
            },
        })

        return SimpleNamespace(
            text="The dataset has 3 rows."
        )

    fake_client = SimpleNamespace(
        chats=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(
                send_message=fake_send_message
            )
        )
    )

    monkeypatch.setattr(
        agent, "build_tools", fake_build_tools
    )

    monkeypatch.setattr(
        agent.genai,
        "Client",
        lambda api_key, **kwargs: fake_client
    )

    answer, trace = agent.ask_agent(
        dataset,
        "Profile this dataset.",
        save_chart_files=False,
    )

    assert captured["output_dir"] is None
    assert answer == "The dataset has 3 rows."
    assert len(trace) == 1
    assert trace[0]["tool"] == "dataset_profile"
