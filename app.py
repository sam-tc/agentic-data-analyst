from hashlib import sha256
from io import BytesIO
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from src.agent import ask_agent
from src.data_loader import load_csv
from src.tools.visualization import create_visualization

from google.genai import errors as genai_errors


st.set_page_config(
    page_title="DataLens | AI Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def charts_from_trace(df, trace):
    """Reconstruct charts selected by the agent for display."""

    charts = []

    for event in trace:
        if event["tool"] != "plot_chart":
            continue

        if "error" in event["result"]:
            continue

        arguments = event["arguments"]

        result = create_visualization(
            df,
            arguments["chart_type"],
            arguments["x_column"],
            arguments.get("y_column") or None,
        )

        charts.append(result["figure"])

    return charts


# Sidebar: dataset management
with st.sidebar:
    st.title("DataLens")
    st.caption("Agentic Data Intelligence")

    st.divider()

    st.subheader("Dataset")

    uploaded_file = st.file_uploader(
        "Upload a CSV dataset",
        type=["csv"],
        help="Choose a tabular CSV file to analyze.",
    )

    if uploaded_file is None:
        dataset_name = "students.csv"
        dataset_bytes = (
            Path(__file__).parent
            / "sample_data"
            / "students.csv"
        ).read_bytes()

        st.info("Using the sample student dataset.")
    else:
        dataset_name = uploaded_file.name
        dataset_bytes = uploaded_file.getvalue()

    if len(dataset_bytes) > 10 * 1024 * 1024:
        st.error("Please upload a CSV smaller than 10 MB.")
        st.stop()

    try:
        df = load_csv(BytesIO(dataset_bytes))
    except Exception as error:
        st.error(f"Could not load dataset: {error}")
        st.stop()

    # Reset conversation when the dataset changes.
    fingerprint = sha256(dataset_bytes).hexdigest()

    if st.session_state.get("dataset_id") != fingerprint:
        st.session_state.dataset_id = fingerprint
        st.session_state.history = []

    st.success(f"Loaded: {dataset_name}")

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    st.divider()
    st.caption("Powered by Gemini and six Python tools")


# Main dashboard
st.title("DataLens")
st.markdown(
    "Ask questions, uncover insights, and visualize your data "
    "using an AI agent that executes real analytical tools."
)

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Dataset rows", f"{len(df):,}")

with col2:
    st.metric("Columns", len(df.columns))

with col3:
    st.metric("Missing values", int(df.isna().sum().sum()))


chat_tab, data_tab = st.tabs([
    "AI Analysis",
    "Dataset Explorer",
])


# Agent chat
with chat_tab:
    st.subheader("Ask your dataset")

    st.caption(
        "Example: Which department has the highest average marks?"
    )

    for index, message in enumerate(st.session_state.history):
        with st.chat_message(message["role"]):
            st.markdown(message["text"])

            for chart_index, figure in enumerate(
                message.get("charts", [])
            ):
                st.plotly_chart(
                    go.Figure(figure),
                    key=f"chart_{index}_{chart_index}",
                    width="stretch",
                )

            if message.get("trace"):
                with st.expander("Inspect tool execution"):
                    for event in message["trace"]:
                        st.json(event)

    question = st.chat_input(
        "Ask a question about your dataset..."
    )


    if question:
        try:
            with st.spinner("Analyzing your dataset..."):
                answer, trace = ask_agent(df, question)
                charts = charts_from_trace(df, trace)

        except genai_errors.ServerError as error:
            if error.code == 503:
                st.warning(
                    "Gemini is temporarily overloaded. "
                    "Your dataset is safe. "
                    "Please try your question again in a few minutes."
                )
            else:
                st.error(
                    "Gemini encountered a server error. "
                    "Please try again later."
                )

        except Exception as error:
            st.error(
                f"Analysis could not be completed "
                f"({type(error).__name__}). "
                "Check the terminal for details."
            )

        else:
            st.session_state.history.extend([
                {
                    "role": "user",
                    "text": question,
                },
                {
                    "role": "assistant",
                    "text": answer,
                    "trace": trace,
                    "charts": charts,
                },
            ])

            st.rerun()


# Dataset explorer
with data_tab:
    st.subheader("Dataset Preview")
    st.caption("Showing the first 100 rows.")

    st.dataframe(
        df.head(100),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Column Information")

    column_info = df.dtypes.astype(str).reset_index()
    column_info.columns = ["Column", "Data Type"]

    st.dataframe(
        column_info,
        width="stretch",
        hide_index=True,
    )
