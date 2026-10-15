"""Test provider selection and Gemini quota-aware fallback."""

import time
from types import SimpleNamespace

import httpx
import pandas as pd
import pytest
from groq import APIConnectionError

from src import provider


class FakeGeminiAPIError(Exception):
    def __init__(self, code: int, message: str = ""):
        self.code = code
        super().__init__(message)


@pytest.fixture(autouse=True)
def isolate_provider(monkeypatch):
    provider._gemini_retry_after.clear()

    monkeypatch.setattr(
        provider,
        "genai_errors",
        SimpleNamespace(APIError=FakeGeminiAPIError),
    )

    monkeypatch.delenv(
        "LUMO_PROVIDER", raising=False
    )
    monkeypatch.delenv(
        "LUMO_GROQ_FALLBACK", raising=False
    )
    monkeypatch.delenv(
        "LUMO_GEMINI_FALLBACK", raising=False
    )

    monkeypatch.setenv(
        "GEMINI_MODEL",
        "gemini-3.6-flash",
    )


@pytest.fixture
def dataset():
    return pd.DataFrame({
        "marks": [80, 90, 100]
    })


def test_groq_is_primary_when_selected(
    dataset, monkeypatch
):
    monkeypatch.setenv("LUMO_PROVIDER", "groq")
    monkeypatch.setenv(
        "LUMO_GEMINI_FALLBACK", "false"
    )

    monkeypatch.setattr(
        provider,
        "ask_groq_agent",
        lambda *a, **kw: ("Groq answer", []),
    )

    monkeypatch.setattr(
        provider,
        "ask_agent",
        lambda *a, **kw: pytest.fail(
            "Unexpected Gemini call"
        ),
    )

    assert provider.ask_lumo_agent(
        dataset, "Analyze"
    ) == ("Groq answer", [])


def test_gemini_is_primary_when_healthy(
    dataset, monkeypatch
):
    monkeypatch.setenv(
        "LUMO_PROVIDER", "gemini"
    )
    monkeypatch.setenv(
        "LUMO_GROQ_FALLBACK", "true"
    )

    monkeypatch.setattr(
        provider,
        "ask_agent",
        lambda *a, **kw: ("Gemini answer", []),
    )

    monkeypatch.setattr(
        provider,
        "ask_groq_agent",
        lambda *a, **kw: pytest.fail(
            "Groq should not be used"
        ),
    )

    assert provider.ask_lumo_agent(
        dataset, "Analyze"
    ) == ("Gemini answer", [])


@pytest.mark.parametrize(
    "enabled", [False, True]
)
def test_original_groq_to_gemini_fallback(
    dataset, monkeypatch, enabled
):
    monkeypatch.setenv(
        "LUMO_PROVIDER", "groq"
    )
    monkeypatch.setenv(
        "LUMO_GEMINI_FALLBACK",
        str(enabled).lower(),
    )

    def failing_groq(*args, **kwargs):
        raise APIConnectionError(
            request=httpx.Request(
                "POST",
                "https://api.groq.com/openai/v1/chat/completions",
            )
        )

    monkeypatch.setattr(
        provider,
        "ask_groq_agent",
        failing_groq,
    )

    monkeypatch.setattr(
        provider,
        "ask_agent",
        lambda *a, **kw: ("Gemini", []),
    )

    if enabled:
        assert provider.ask_lumo_agent(
            dataset, "Analyze"
        ) == ("Gemini", [])

    else:
        with pytest.raises(APIConnectionError):
            provider.ask_lumo_agent(
                dataset, "Analyze"
            )


def test_429_switches_to_groq_and_skips_gemini(
    dataset, monkeypatch
):
    monkeypatch.setenv(
        "LUMO_PROVIDER", "gemini"
    )

    attempts = []

    def failing_gemini(*args, **kwargs):
        attempts.append("gemini")
        raise FakeGeminiAPIError(
            429,
            "requests_per_minute exhausted",
        )

    monkeypatch.setattr(
        provider,
        "ask_agent",
        failing_gemini,
    )

    monkeypatch.setattr(
        provider,
        "ask_groq_agent",
        lambda *a, **kw: ("Groq", []),
    )

    assert provider.ask_lumo_agent(
        dataset, "Analyze"
    ) == ("Groq", [])

    assert provider.ask_lumo_agent(
        dataset, "Analyze"
    ) == ("Groq", [])

    # Gemini is contacted only once during the cooldown.
    assert attempts == ["gemini"]


def test_503_switches_to_groq(
    dataset, monkeypatch
):
    def failing_gemini(*args, **kwargs):
        raise FakeGeminiAPIError(503)

    monkeypatch.setattr(
        provider,
        "ask_agent",
        failing_gemini,
    )

    monkeypatch.setattr(
        provider,
        "ask_groq_agent",
        lambda *a, **kw: ("Groq", []),
    )

    assert provider.ask_lumo_agent(
        dataset, "Analyze"
    ) == ("Groq", [])


def test_non_transient_error_does_not_fallback(
    dataset, monkeypatch
):
    def failing_gemini(*args, **kwargs):
        raise FakeGeminiAPIError(403)

    monkeypatch.setattr(
        provider,
        "ask_agent",
        failing_gemini,
    )

    monkeypatch.setattr(
        provider,
        "ask_groq_agent",
        lambda *a, **kw: pytest.fail(
            "Must not fall back on HTTP 403"
        ),
    )

    with pytest.raises(FakeGeminiAPIError) as error:
        provider.ask_lumo_agent(
            dataset, "Analyze"
        )

    assert error.value.code == 403


def test_gemini_fallback_can_be_disabled(
    dataset, monkeypatch
):
    monkeypatch.setenv(
        "LUMO_GROQ_FALLBACK", "false"
    )

    def failing_gemini(*args, **kwargs):
        raise FakeGeminiAPIError(429)

    monkeypatch.setattr(
        provider,
        "ask_agent",
        failing_gemini,
    )

    monkeypatch.setattr(
        provider,
        "ask_groq_agent",
        lambda *a, **kw: pytest.fail(
            "Fallback is disabled"
        ),
    )

    with pytest.raises(FakeGeminiAPIError):
        provider.ask_lumo_agent(
            dataset, "Analyze"
        )


def test_gemini_retried_after_cooldown_expires(
    dataset, monkeypatch
):
    provider._gemini_retry_after[
        "gemini-3.6-flash"
    ] = time.monotonic() - 1

    monkeypatch.setattr(
        provider,
        "ask_agent",
        lambda *a, **kw: ("Gemini", []),
    )

    monkeypatch.setattr(
        provider,
        "ask_groq_agent",
        lambda *a, **kw: pytest.fail(
            "Gemini should be retried"
        ),
    )

    assert provider.ask_lumo_agent(
        dataset, "Analyze"
    ) == ("Gemini", [])


def test_daily_quota_cooldown_uses_pacific_midnight():
    delay = provider._cooldown_seconds(
        FakeGeminiAPIError(
            429,
            "GenerateRequestsPerDayPerProjectPerModel",
        )
    )

    assert 0 < delay < 25 * 3600
