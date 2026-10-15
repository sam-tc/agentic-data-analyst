"""Test provider selection without making external API requests."""

import httpx
import pandas as pd
import pytest
from groq import APIConnectionError

from src import provider


@pytest.fixture
def dataset():
    return pd.DataFrame({"marks": [80, 90, 100]})


def test_groq_is_primary(dataset, monkeypatch):
    monkeypatch.setenv("LUMO_PROVIDER", "groq")
    monkeypatch.setenv("LUMO_GEMINI_FALLBACK", "false")

    def fake_groq(df, question, *, save_chart_files):
        assert save_chart_files is False
        return "Groq answer", [{"tool": "dataset_profile"}]

    def unexpected_gemini(*args, **kwargs):
        pytest.fail("Gemini must not be called")

    monkeypatch.setattr(
        provider, "ask_groq_agent", fake_groq
    )
    monkeypatch.setattr(
        provider, "ask_agent", unexpected_gemini
    )

    answer, trace = provider.ask_lumo_agent(
        dataset, "Describe my data"
    )

    assert answer == "Groq answer"
    assert trace[0]["tool"] == "dataset_profile"


def test_gemini_can_be_selected(dataset, monkeypatch):
    monkeypatch.setenv("LUMO_PROVIDER", "gemini")

    def fake_gemini(df, question, *, save_chart_files):
        return "Gemini answer", []

    monkeypatch.setattr(
        provider, "ask_agent", fake_gemini
    )

    answer, trace = provider.ask_lumo_agent(
        dataset, "Describe my data"
    )

    assert answer == "Gemini answer"
    assert trace == []


@pytest.mark.parametrize(
    "fallback_enabled",
    [False, True],
)
def test_fallback_is_optional(
    dataset, monkeypatch, fallback_enabled
):
    monkeypatch.setenv("LUMO_PROVIDER", "groq")
    monkeypatch.setenv(
        "LUMO_GEMINI_FALLBACK",
        str(fallback_enabled).lower(),
    )

    called = []

    def failing_groq(df, question, *, save_chart_files):
        raise APIConnectionError(
            request=httpx.Request(
                "POST",
                "https://api.groq.com/openai/v1/chat/completions",
            )
        )

    def fake_gemini(df, question, *, save_chart_files):
        called.append("gemini")
        return "Fallback answer", []

    monkeypatch.setattr(
        provider, "ask_groq_agent", failing_groq
    )
    monkeypatch.setattr(
        provider, "ask_agent", fake_gemini
    )

    if fallback_enabled:
        answer, _ = provider.ask_lumo_agent(
            dataset, "Analyze"
        )
        assert answer == "Fallback answer"
        assert called == ["gemini"]
    else:
        with pytest.raises(APIConnectionError):
            provider.ask_lumo_agent(
                dataset, "Analyze"
            )
        assert called == []
