"""DataLens Studio: AI-powered data analysis workspace."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from google.genai import errors as genai_errors

from src.agent import ask_agent
from src.data_loader import load_csv
from src.tools.visualization import create_visualization


ROOT = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DataLens Studio | AI Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

css_file = ROOT / "assets" / "datalens.css"

if css_file.is_file():
    st.html(css_file)


# ============================================================
# DYNAMIC SUGGESTED QUESTIONS
# ============================================================

def suggested_questions(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Generate questions based on actual dataset columns."""

    numeric = [
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
        and not pd.api.types.is_bool_dtype(df[col])
    ]

    categorical = [
        col for col in df.columns
        if col not in numeric
    ]

    group_col = next(
        (
            col for col in categorical
            if 1 < df[col].nunique(dropna=True) < len(df)
        ),
        categorical[0] if categorical else None,
    )

    prompts = [
        (
            "✦ Explore dataset",
            "Give me an overview of this dataset.",
        )
    ]

    if group_col and numeric:
        prompts.append(
            (
                "▦ Compare groups",
                f"Compare average {numeric[-1]} by {group_col}.",
            )
        )

    elif len(numeric) >= 2:
        prompts.append(
            (
                "↗ Find correlation",
                f"Calculate the correlation between "
                f"{numeric[0]} and {numeric[1]}.",
            )
        )

    if group_col:
        prompts.append(
            (
                "▥ Create chart",
                f"Create a bar chart showing counts by {group_col}.",
            )
        )

    elif numeric:
        prompts.append(
            (
                "▥ Create chart",
                f"Create a histogram of {numeric[0]}.",
            )
        )

    return prompts


# ============================================================
# CHART HANDLING
# ============================================================

def charts_from_trace(
    df: pd.DataFrame,
    trace: list[dict]
) -> list[dict]:
    """Rebuild charts only when the agent executed plot_chart."""

    figures = []

    for event in trace:

        if event.get("tool") != "plot_chart":
            continue

        if "error" in event.get("result", {}):
            continue

        args = event.get("arguments", {})

        try:
            chart = create_visualization(
                df,
                args["chart_type"],
                args["x_column"],
                args.get("y_column") or None,
            )

        except (KeyError, ValueError, TypeError):
            continue

        figures.append(chart["figure"])

    return figures


def show_chart(figure: dict, key: str) -> None:
    """Display Plotly charts with DataLens styling."""

    fig = go.Figure(figure)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="#dbe5fb",
        ),
        margin=dict(
            l=12,
            r=12,
            t=60,
            b=20,
        ),
        hoverlabel=dict(
            bgcolor="#17223b",
            font_color="#ffffff",
        ),
    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
    )

    fig.update_yaxes(
        gridcolor="rgba(180,195,230,0.12)",
        zeroline=False,
    )

    for trace in fig.data:

        if trace.type in ("bar", "histogram"):
            trace.marker.color = "#8c86ff"

        elif trace.type == "scatter":
            trace.marker.color = "#63ddeb"
            trace.line.color = "#8c86ff"

    st.plotly_chart(
        fig,
        key=key,
        width="stretch",
    )


# ============================================================
# AGENT EXECUTION INSPECTOR
# ============================================================

def show_trace(trace: list[dict]) -> None:
    """Display real Python tool executions."""

    with st.expander(
        f"View execution details · {len(trace)} tool call(s)"
    ):

        st.caption(
            "Recorded calls to the Python analytical tools, "
            "not model-generated guesses."
        )

        for i, event in enumerate(trace, start=1):

            st.markdown(
                f"**{i}. `{event.get('tool', 'unknown')}`**"
            )

            arguments_col, results_col = st.columns(2)

            with arguments_col:

                st.caption("Arguments")

                st.json(
                    event.get("arguments", {})
                )

            with results_col:

                st.caption("Actual result")

                st.json(
                    event.get("result", {})
                )

            if i < len(trace):
                st.divider()


# ============================================================
# SIDEBAR — DATASET MANAGEMENT
# ============================================================

with st.sidebar:

    st.html("""
    <div class="dl-brand">
        <div class="dl-logo">✦</div>
        <div>
            <strong>DataLens</strong>
            <small>INTELLIGENCE STUDIO</small>
        </div>
    </div>
    """)

    st.markdown("### Workspace")

    st.caption(
        "Your data, ready to explore."
    )

    upload = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Maximum size: 10 MB.",
    )

    if upload is None:

        dataset_name = "students.csv"

        sample = (
            ROOT / "sample_data" / dataset_name
        )

        if not sample.is_file():

            st.error(
                "Missing sample_data/students.csv"
            )

            st.stop()

        dataset_bytes = sample.read_bytes()

        st.info(
            "Sample dataset is loaded. "
            "Upload your own CSV anytime."
        )

    else:

        dataset_name = upload.name

        dataset_bytes = upload.getvalue()

    if len(dataset_bytes) > MAX_UPLOAD_BYTES:

        st.error(
            "The CSV file must be 10 MB or smaller."
        )

        st.stop()

    # Load dataset using our existing backend.

    try:

        df = load_csv(
            BytesIO(dataset_bytes)
        )

    except (
        ValueError,
        UnicodeError,
        pd.errors.ParserError,
    ) as error:

        st.error(
            f"Could not read this CSV: {error}"
        )

        st.stop()

    # Reset chat whenever the dataset changes.

    fingerprint = (
        f"{dataset_name}:"
        f"{sha256(dataset_bytes).hexdigest()}"
    )

    if st.session_state.get("dataset_id") != fingerprint:

        st.session_state.dataset_id = fingerprint

        st.session_state.history = []

    st.success(
        f"Active dataset: {dataset_name}"
    )

    if st.button(
        "↺ Clear conversation",
        use_container_width=True,
    ):

        st.session_state.history = []

        st.rerun()

    st.html("""
    <div class="dl-sidebar-panel">
        <span class="dl-sidebar-label">
            AVAILABLE CAPABILITIES
        </span>

        <strong>
            06 <span>analytical tools</span>
        </strong>

        <p>
            Profile · Filter · Aggregate ·
            Group · Correlate · Visualize
        </p>
    </div>
    """)

    st.caption(
        "Gemini · Pandas · Plotly"
    )


# ============================================================
# HERO SECTION
# ============================================================

st.html("""
<section class="dl-hero">

    <div class="dl-overline">
        <span class="dl-status-dot"></span>
        YOUR INTELLIGENT ANALYSIS WORKSPACE
    </div>

    <h1>
        Make sense of data.<br>
        <span>Discover what matters.</span>
    </h1>

    <p>
        Drop in a dataset. Ask a question.
        Get reliable calculations, useful insights and
        interactive visualizations — all backed by
        real Python tools.
    </p>

    <div class="dl-chips">
        <span>✦ AI-assisted analysis</span>
        <span>◈ Six real tools</span>
        <span>↗ Interactive insights</span>
    </div>

    <div class="dl-orb dl-orb-one"></div>
    <div class="dl-orb dl-orb-two"></div>

</section>
""")


# ============================================================
# DATASET METRICS
# ============================================================

st.html("""
<div class="dl-section-label">
    01 <span>DATASET AT A GLANCE</span>
</div>
""")

row_card, col_card, null_card = st.columns(
    3,
    gap="medium",
)

with row_card:

    st.metric(
        "Records",
        f"{len(df):,}",
    )

with col_card:

    st.metric(
        "Features",
        f"{len(df.columns):,}",
    )

with null_card:

    st.metric(
        "Missing values",
        f"{int(df.isna().sum().sum()):,}",
    )


# ============================================================
# MAIN NAVIGATION
# ============================================================

analysis_tab, explore_tab = st.tabs(
    [
        "✦ AI Analyst",
        "▦ Dataset Explorer",
    ]
)


# ============================================================
# AI ANALYSIS
# ============================================================

with analysis_tab:

    st.markdown(
        "### Ask better questions. Get clearer answers."
    )

    st.caption(
        f"Currently exploring **{dataset_name}** · "
        f"{len(df):,} rows · "
        f"{len(df.columns)} columns"
    )

    st.html("""
    <div class="dl-prompt-intro">
        START WITH AN IDEA

        <span>
            Choose a quick prompt or ask anything below
        </span>
    </div>
    """)

    # Dynamic suggested prompts.

    prompts = suggested_questions(df)

    quick_question = None

    prompt_columns = st.columns(
        len(prompts),
        gap="small",
    )

    for i, (label, prompt) in enumerate(prompts):

        with prompt_columns[i]:

            if st.button(
                label,
                key=f"suggest_{i}",
                use_container_width=True,
            ):

                quick_question = prompt

    # Empty conversation state.

    if not st.session_state.history:

        st.html("""
        <div class="dl-empty-state">

            <div class="dl-empty-icon">✧</div>

            <strong>
                Your next insight starts here
            </strong>

            <p>
                Ask for patterns, comparisons, summaries
                or a chart. The agent will choose and run
                the appropriate analytical tools.
            </p>

        </div>
        """)

    # Display previous messages.

    for message_index, message in enumerate(
        st.session_state.history
    ):

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["text"]
            )

            for chart_index, chart in enumerate(
                message.get("charts", [])
            ):

                show_chart(
                    chart,
                    key=f"viz_{message_index}_{chart_index}",
                )

            if message.get("trace"):

                show_trace(
                    message["trace"]
                )

    # Receive user question.

    typed_question = st.chat_input(
        "Ask anything about your dataset…"
    )

    question = typed_question or quick_question

    # Execute the AI agent.

    if question:

        with st.chat_message("user"):

            st.markdown(question)

        with st.chat_message("assistant"):

            try:

                with st.spinner(
                    "Running analytical tools…"
                ):

                    answer, trace = ask_agent(
                        df,
                        question,
                    )

                    charts = charts_from_trace(
                        df,
                        trace,
                    )

            # Gemini API errors.

            except genai_errors.APIError as error:

                code = getattr(
                    error,
                    "code",
                    None,
                )

                if code == 503:

                    st.warning(
                        "Gemini is temporarily overloaded. "
                        "Please retry shortly."
                    )

                elif code == 429:

                    st.warning(
                        "Gemini's rate limit was reached. "
                        "Please wait and retry."
                    )

                elif code == 403:

                    st.error(
                        "Gemini access was denied for this API "
                        "project. Check Google AI Studio."
                    )

                else:

                    st.error(
                        "Gemini could not complete this request. "
                        "Try again."
                    )

            except Exception as error:

                st.error(
                    f"Analysis failed "
                    f"({type(error).__name__}). "
                    "Check the terminal for details."
                )

            # Save successful response.

            else:

                st.session_state.history.extend(
                    [
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
                    ]
                )

                st.rerun()


# ============================================================
# DATASET EXPLORER
# ============================================================

with explore_tab:

    st.markdown(
        "### Explore the underlying data"
    )

    st.caption(
        "Preview the first 100 rows, then review "
        "each column's type and missing values."
    )

    st.dataframe(
        df.head(100),
        width="stretch",
        hide_index=True,
    )

    st.markdown(
        "#### Column details"
    )

    column_details = pd.DataFrame(
        {
            "Column": [
                str(col)
                for col in df.columns
            ],
            "Type": [
                str(df[col].dtype)
                for col in df.columns
            ],
            "Missing": [
                int(df[col].isna().sum())
                for col in df.columns
            ],
            "Unique": [
                int(df[col].nunique(dropna=True))
                for col in df.columns
            ],
        }
    )

    st.dataframe(
        column_details,
        width="stretch",
        hide_index=True,
    )

    st.download_button(
        "↓ Download current CSV",
        data=dataset_bytes,
        file_name=dataset_name,
        mime="text/csv",
    )
