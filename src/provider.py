"""Route Lumo questions to Gemini or Groq, with quota-aware fallback."""

import logging
import os
import threading
import time
from datetime import datetime, time as midnight_time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv
from google.genai import errors as genai_errors
from groq import APIConnectionError, InternalServerError, RateLimitError

from .agent import ask_agent
from .groq_agent import ask_groq_agent


logger = logging.getLogger(__name__)

# Shared within this Python process, including Streamlit reruns.
_gemini_retry_after: dict[str, float] = {}
_cooldown_lock = threading.Lock()


def _enabled(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {
        "true", "1", "yes"
    }


def _gemini_status(error: Exception) -> int | None:
    if not isinstance(error, genai_errors.APIError):
        return None

    code = getattr(error, "code", None)

    if code is None:
        code = getattr(error, "status_code", None)

    try:
        return int(code)
    except (TypeError, ValueError):
        return None


def _cooldown_seconds(error: Exception) -> float:
    """Determine when Gemini should next be attempted."""

    code = _gemini_status(error)

    if code != 429:
        # Temporary Gemini server errors.
        return 90.0

    details = str(error).lower()

    daily_markers = (
        "perday",
        "per_day",
        "per day",
        "requests/day",
        "requests_per_day",
        "daily",
        "rpd",
    )

    if any(marker in details for marker in daily_markers):
        pacific = ZoneInfo("America/Los_Angeles")
        now = datetime.now(pacific)

        next_midnight = datetime.combine(
            now.date() + timedelta(days=1),
            midnight_time.min,
            tzinfo=pacific,
        )

        return max(
            60.0,
            (next_midnight - now).total_seconds(),
        )

    minute_markers = (
        "perminute",
        "per_minute",
        "per minute",
        "requests/minute",
        "rpm",
    )

    if any(marker in details for marker in minute_markers):
        return 75.0

    # Unknown quota type: avoid repeatedly hitting the same limit.
    return 15 * 60.0


def _gemini_in_cooldown(model: str) -> bool:
    with _cooldown_lock:
        return (
            time.monotonic()
            < _gemini_retry_after.get(model, 0.0)
        )


def _pause_gemini(model: str, error: Exception) -> None:
    delay = _cooldown_seconds(error)

    with _cooldown_lock:
        _gemini_retry_after[model] = max(
            _gemini_retry_after.get(model, 0.0),
            time.monotonic() + delay,
        )

    logger.warning(
        "Gemini returned HTTP %s; using Groq fallback "
        "for approximately %d seconds",
        _gemini_status(error),
        round(delay),
    )


def ask_lumo_agent(
    df: pd.DataFrame,
    question: str,
    *,
    save_chart_files: bool = False,
) -> tuple[str, list]:
    """Execute the primary agent with controlled provider fallback."""

    load_dotenv()

    provider = os.getenv(
        "LUMO_PROVIDER", "gemini"
    ).strip().lower()

    if provider == "gemini":
        allow_groq = _enabled(
            "LUMO_GROQ_FALLBACK", "true"
        )

        model = os.getenv(
            "GEMINI_MODEL", "gemini-3.8-flash"
        )

        # Skip known-unavailable Gemini models during cooldown.
        if allow_groq and _gemini_in_cooldown(model):
            return ask_groq_agent(
                df,
                question,
                save_chart_files=save_chart_files,
            )

        try:
            return ask_agent(
                df,
                question,
                save_chart_files=save_chart_files,
            )

        except genai_errors.APIError as error:
            retryable_codes = (
                429, 500, 502, 503, 504
            )

            if (
                not allow_groq
                or _gemini_status(error) not in retryable_codes
            ):
                raise

            _pause_gemini(model, error)

            return ask_groq_agent(
                df,
                question,
                save_chart_files=save_chart_files,
            )

    if provider == "groq":
        try:
            return ask_groq_agent(
                df,
                question,
                save_chart_files=save_chart_files,
            )

        except (
            RateLimitError,
            APIConnectionError,
            InternalServerError,
        ):
            if not _enabled("LUMO_GEMINI_FALLBACK"):
                raise

            return ask_agent(
                df,
                question,
                save_chart_files=save_chart_files,
            )

    raise ValueError(
        f"Unknown LUMO_PROVIDER '{provider}'. "
        "Choose 'gemini' or 'groq'."
    )
