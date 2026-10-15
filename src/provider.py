"""Select the AI provider for Lumo Studio."""

import logging
import os

import pandas as pd
from dotenv import load_dotenv
from groq import APIConnectionError, InternalServerError, RateLimitError

from .agent import ask_agent
from .groq_agent import ask_groq_agent


logger = logging.getLogger(__name__)


def ask_lumo_agent(
    df: pd.DataFrame,
    question: str,
    *,
    save_chart_files: bool = False,
) -> tuple[str, list]:
    """Run the selected provider and return (answer, tool trace)."""

    load_dotenv()

    provider = os.getenv(
        "LUMO_PROVIDER", "groq"
    ).strip().lower()

    if provider == "gemini":
        return ask_agent(
            df,
            question,
            save_chart_files=save_chart_files,
        )

    if provider != "groq":
        raise ValueError(
            f"Unknown LUMO_PROVIDER '{provider}'. "
            "Choose 'groq' or 'gemini'."
        )

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
    ) as error:
        fallback_enabled = os.getenv(
            "LUMO_GEMINI_FALLBACK", "false"
        ).strip().lower() in {"true", "1", "yes"}

        if not fallback_enabled:
            raise

        logger.warning(
            "Groq is unavailable (%s). Trying Gemini fallback.",
            type(error).__name__,
        )

        return ask_agent(
            df,
            question,
            save_chart_files=save_chart_files,
        )
