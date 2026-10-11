import os

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

from .agent_tools import build_tools


def ask_agent(df: pd.DataFrame, question: str) -> tuple[str, list]:
    """Answer dataset questions using real Python analytical tools."""

    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "Missing GEMINI_API_KEY. Configure it in your .env file."
        )

    if df.empty:
        raise ValueError("The dataset has no rows.")

    if not question.strip():
        raise ValueError("Please provide a question.")

    client = genai.Client(api_key=api_key)

    trace = []
    tools = build_tools(df, trace)

    schema = "\n".join(
        f"- {column}: {df[column].dtype}"
        for column in df.columns
    )

    prompt = (
        f"Dataset contains {len(df)} rows.\n"
        f"Available columns:\n{schema}\n\n"
        f"User question: {question}"
    )

    config = types.GenerateContentConfig(
        system_instruction=(
            "You are a data analysis agent. "
            "Use the provided Python analytical tools to answer "
            "questions about the current dataset. "
            "For every analytical question, execute at least one tool. "
            "Never invent calculated values or pretend tools ran. "
            "Base numerical answers on actual tool results. "
            "If a tool returns an error, explain the limitation "
            "or choose a suitable alternative tool. "
            "Clearly disclose if only part of a dataset was analyzed."
        ),
        tools=tools,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            maximum_remote_calls=8
        ),
    )

    chat = client.chats.create(
        model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
        config=config,
    )

    response = chat.send_message(prompt)

    if not trace:
        raise RuntimeError(
            "The model did not execute an analytical tool."
        )

    return response.text or "No final answer returned.", trace
