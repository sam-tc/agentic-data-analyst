"""DataLens Studio: a tool-using AI data analyst.

UI layer only. The backend contract is unchanged:
    ask_agent(df, question) -> (answer_text, trace)
    trace = [{"tool": str, "arguments": dict, "result": dict}, ...]
"""

from __future__ import annotations

import html
import re
import time
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from google.genai import errors as genai_errors

from src.agent import ask_agent
from src.data_loader import load_csv
from src.tools.visualization import create_visualization

ROOT = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

INK = "#0E1220"
COBALT = "#3B4BFF"

USER_AVATAR = ":material/person:"
BOT_AVATAR = ":material/auto_awesome:"

# One entry per backend tool. Order = order on the tool belt.
TOOLS = {
    "dataset_profile": {
        "label": "Profile",
        "icon": ":material/fact_check:",
        "glyph": "◧",
        "color": "#7C5CFF",
        "blurb": "Types, gaps and ranges",
    },
    "row_filter": {
        "label": "Filter",
        "icon": ":material/filter_alt:",
        "glyph": "▽",
        "color": "#0EA5A4",
        "blurb": "Rows that match a rule",
    },
    "column_aggregation": {
        "label": "Aggregate",
        "icon": ":material/functions:",
        "glyph": "Σ",
        "color": "#F59E0B",
        "blurb": "Mean, sum, min, max",
    },
    "group_comparison": {
        "label": "Group",
        "icon": ":material/stacked_bar_chart:",
        "glyph": "▤",
        "color": "#E8467C",
        "blurb": "Compare categories",
    },
    "correlation_analysis": {
        "label": "Correlate",
        "icon": ":material/scatter_plot:",
        "glyph": "↗",
        "color": "#2E7BFF",
        "blurb": "How two numbers move",
    },
    "plot_chart": {
        "label": "Plot",
        "icon": ":material/bar_chart:",
        "glyph": "▥",
        "color": "#12B76A",
        "blurb": "Bar, histogram, line",
    },
}
UNKNOWN_TOOL = {"label": "Tool", "glyph": "•", "color": "#64708A", "blurb": ""}


def tool_meta(name: str) -> dict:
    return TOOLS.get(name, {**UNKNOWN_TOOL, "label": name or "Tool"})


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="DataLens Studio",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="auto",
)

css_file = ROOT / "assets" / "datalens.css"
if css_file.is_file():
    st.html(css_file)


def tool_belt_css() -> str:
    """Per-tool colour and caption, generated from TOOLS (single source of truth)."""
    rules = []
    for name, tool in TOOLS.items():
        blurb = tool["blurb"].replace('"', "'")
        for state in ("on", "off"):
            key = f".st-key-tool-{state}-{name}"
            rules.append(f"{key} {{ --c: {tool['color']}; }}")
            rules.append(f'{key} button::after {{ content: "{blurb}"; }}')
    return "<style>" + "\n".join(rules) + "</style>"


st.html(tool_belt_css())


# ============================================================
# DATASET HELPERS
# ============================================================

def is_numeric(series: pd.Series) -> bool:
    return (
        pd.api.types.is_numeric_dtype(series)
        and not pd.api.types.is_bool_dtype(series)
    )


def id_like(df: pd.DataFrame, col: str) -> bool:
    name = str(col).lower().strip()
    if name == "id" or name.endswith(("_id", " id")):
        return True
    return (
        pd.api.types.is_integer_dtype(df[col])
        and len(df) > 10
        and df[col].nunique() == len(df)
    )


def split_columns(df: pd.DataFrame):
    """numeric, measures (numeric minus IDs), categories, best grouping column."""
    numeric = [c for c in df.columns if is_numeric(df[c])]
    measures = [c for c in numeric if not id_like(df, c)] or numeric
    categories = [c for c in df.columns if c not in numeric]
    group_col = next(
        (c for c in categories if 1 < df[c].nunique(dropna=True) < len(df)),
        None,
    )
    return numeric, measures, categories, group_col


def fmt(value: float) -> str:
    if value is None or pd.isna(value):
        return "–"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:,.2f}".rstrip("0").rstrip(".")


def completeness(df: pd.DataFrame) -> float:
    return 100 * (1 - df.isna().sum().sum() / max(df.size, 1))


def tool_prompts(df: pd.DataFrame) -> dict[str, str | None]:
    """A starter question per tool, built from the real columns and values."""
    _, measures, categories, group_col = split_columns(df)
    target = measures[-1] if measures else None
    driver = measures[0] if len(measures) > 1 else None

    prompts: dict[str, str | None] = {
        "dataset_profile": (
            "Profile this dataset: column types, missing values "
            "and numeric ranges."
        ),
        "row_filter": None,
        "column_aggregation": None,
        "group_comparison": None,
        "correlation_analysis": None,
        "plot_chart": None,
    }

    if target:
        median = df[target].median()
        prompts["row_filter"] = (
            f"Show rows where {target} is greater than {fmt(median)}."
        )
        prompts["column_aggregation"] = (
            f"What are the average and highest {target}?"
        )
    elif categories:
        top = df[categories[0]].mode(dropna=True)
        if len(top):
            prompts["row_filter"] = (
                f"Show rows where {categories[0]} is {top.iloc[0]}."
            )

    if group_col and target:
        prompts["group_comparison"] = (
            f"Compare average {target} by {group_col}."
        )
    if driver and target:
        prompts["correlation_analysis"] = (
            f"How strongly are {driver} and {target} correlated?"
        )
    if group_col:
        prompts["plot_chart"] = (
            f"Create a bar chart showing counts by {group_col}."
        )
    elif target:
        prompts["plot_chart"] = f"Create a histogram of {target}."

    return prompts


# ---------- small inline graphics ----------

KIND_STYLE = {
    "num": ("#", COBALT),
    "text": ("Aa", "#0EA5A4"),
    "bool": ("T/F", "#F59E0B"),
    "time": ("◷", "#7C5CFF"),
}


def column_kind(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "time"
    if pd.api.types.is_numeric_dtype(series):
        return "num"
    return "text"


def mini_histogram(series: pd.Series, color: str) -> str:
    values = (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if values.empty:
        return ""
    counts, _ = np.histogram(values, bins=min(14, max(4, values.nunique())))
    peak = max(counts.max(), 1)
    bars = "".join(
        f'<i style="height:{0 if c == 0 else max(8, 100 * c / peak):.0f}%;'
        f'opacity:{0.35 + 0.65 * c / peak:.2f}"></i>'
        for c in counts
    )
    return f'<div class="mini" style="--c:{color}">{bars}</div>'


def mini_categories(series: pd.Series, color: str) -> str:
    counts = series.astype("string").fillna("(missing)").value_counts().head(3)
    total = max(len(series), 1)
    rows = []
    for label, count in counts.items():
        pct = 100 * count / total
        rows.append(
            f'<div class="cat"><span>{html.escape(str(label))}</span>'
            f'<i style="--w:{pct:.0f}%;--c:{color}"></i>'
            f"<em>{pct:.0f}%</em></div>"
        )
    return f'<div class="cats">{"".join(rows)}</div>'


def column_card(df: pd.DataFrame, col: str) -> str:
    series = df[col]
    kind = column_kind(series)
    glyph, color = KIND_STYLE[kind]
    filled = 100 * (1 - series.isna().mean())

    if kind == "num":
        graphic = mini_histogram(series, color)
        valid = series.dropna()
        detail = (
            f"{fmt(valid.min())} to {fmt(valid.max())} · mean {fmt(valid.mean())}"
            if not valid.empty
            else "no values"
        )
    else:
        graphic = mini_categories(series, color)
        detail = f"{series.nunique(dropna=True):,} unique"

    return (
        f'<div class="col-card" style="--c:{color}">'
        f'<div class="col-head"><span class="col-glyph">{glyph}</span>'
        f'<b title="{html.escape(str(col))}">{html.escape(str(col))}</b>'
        f"<em>{html.escape(str(series.dtype))}</em></div>"
        f"{graphic}"
        f'<div class="col-foot"><span>{html.escape(detail)}</span>'
        f'<span class="meter" title="{filled:.0f}% filled">'
        f'<i style="width:{filled:.0f}%"></i></span></div></div>'
    )


def pct_text(pct: float) -> str:
    """Whole-number percent that never rounds a real gap up to 100%."""
    return f"{int(pct // 1)}%" if pct < 100 else "100%"


def completeness_ring(pct: float) -> str:
    return (
        f'<div class="ring" style="--p:{pct:.1f}">'
        f"<span>{pct_text(pct)}</span></div>"
    )


# ============================================================
# CHARTS
# ============================================================

def charts_from_trace(df: pd.DataFrame, trace: list[dict]) -> list[dict]:
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
    fig = go.Figure(figure)
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Instrument Sans, system-ui, sans-serif",
            color=INK,
            size=13,
        ),
        title=dict(x=0, xanchor="left", font=dict(size=16)),
        margin=dict(l=8, r=8, t=56, b=16),
        hoverlabel=dict(bgcolor=INK, font_color="#ffffff", bordercolor=INK),
        bargap=0.3,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#C9D1E3")
    fig.update_yaxes(gridcolor="#E3E8F3", zeroline=False)

    for trace in fig.data:
        if trace.type in ("bar", "histogram"):
            trace.marker.color = COBALT
        elif trace.type == "scatter":
            trace.marker.color = COBALT
            trace.marker.size = 11
            trace.marker.line = dict(color="#ffffff", width=1.5)
            trace.line.color = COBALT
    try:
        fig.update_traces(marker_cornerradius=6, selector=dict(type="bar"))
    except (ValueError, AttributeError):
        pass

    st.plotly_chart(
        fig,
        key=key,
        width="stretch",
        config={"displayModeBar": False},
    )


def correlation_heatmap(df: pd.DataFrame, numeric: list[str]) -> go.Figure:
    corr = df[numeric].corr()
    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=numeric,
            y=numeric,
            zmin=-1,
            zmax=1,
            colorscale=[
                [0.0, "#E8467C"],
                [0.5, "#F4F6FB"],
                [1.0, COBALT],
            ],
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            hovertemplate="%{x} × %{y}: %{z:.2f}<extra></extra>",
            xgap=4,
            ygap=4,
            colorbar=dict(thickness=10, len=0.8, outlinewidth=0),
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Instrument Sans, system-ui, sans-serif",
            color=INK,
            size=13,
        ),
        margin=dict(l=8, r=8, t=8, b=8),
        height=max(260, 70 * len(numeric)),
        yaxis=dict(autorange="reversed"),
    )
    return fig


# ============================================================
# AGENT OUTPUT: PIPELINE, RECEIPTS, ERRORS
# ============================================================

def result_failed(event: dict) -> bool:
    result = event.get("result")
    return isinstance(result, dict) and "error" in result


def pipeline_html(trace: list[dict]) -> str:
    nodes = []
    for event in trace:
        meta = tool_meta(event.get("tool", ""))
        failed = result_failed(event)
        nodes.append(
            f'<span class="node{" fail" if failed else ""}" '
            f'style="--c:{meta["color"]}">'
            f'<b>{meta["glyph"]}</b>{html.escape(meta["label"])}</span>'
        )

    count = len(trace)
    failures = sum(result_failed(e) for e in trace)
    plural = "" if count == 1 else "s"
    if failures:
        badge = (
            f'<span class="verified warn">{count} tool call{plural} · '
            f"{failures} returned an error</span>"
        )
    else:
        badge = (
            f'<span class="verified">Backed by {count} real tool '
            f"call{plural}</span>"
        )

    wire = '<i class="wire"></i>'
    return (
        '<div class="dl-pipe">'
        '<span class="end">Question</span>'
        f"{wire}{wire.join(nodes)}{wire}"
        '<span class="end">Answer</span>'
        f"{badge}</div>"
    )


def short(value, limit: int = 56) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def call_header_html(step: int, event: dict) -> str:
    meta = tool_meta(event.get("tool", ""))
    args = event.get("arguments") or {}
    if args:
        chips = "".join(
            f'<span class="arg"><i>{html.escape(str(k))}</i>'
            f"{html.escape(short(v))}</span>"
            for k, v in args.items()
        )
    else:
        chips = '<span class="arg none">no arguments</span>'
    failed = " fail" if result_failed(event) else ""
    return (
        f'<div class="call{failed}" style="--c:{meta["color"]}">'
        f'<span class="step">{step}</span>'
        f'<span class="node"><b>{meta["glyph"]}</b>'
        f'{html.escape(str(event.get("tool", "unknown")))}</span>'
        f'<div class="args">{chips}</div></div>'
    )


def clean_result(result):
    if isinstance(result, dict) and "figure" in result:
        trimmed = {k: v for k, v in result.items() if k != "figure"}
        trimmed["figure"] = "(chart rendered above)"
        return trimmed
    return result


def show_receipts(trace: list[dict]) -> None:
    count = len(trace)
    with st.expander(
        f"Receipts · {count} tool call{'s' if count != 1 else ''}",
        icon=":material/receipt_long:",
    ):
        st.caption(
            "Every call below actually ran in Python. "
            "These are the real inputs and outputs, not model guesses."
        )
        for step, event in enumerate(trace, start=1):
            st.html(call_header_html(step, event))
            st.json(clean_result(event.get("result", {})), expanded=1)


def friendly_error(error: Exception) -> tuple[str, str, str]:
    """-> (level, title, how to fix)"""
    if isinstance(error, genai_errors.APIError):
        code = getattr(error, "code", None)
        if code == 503:
            return ("warn", "Gemini is overloaded right now",
                    "This usually clears in a few seconds. Try again.")
        if code == 429:
            return ("warn", "Gemini rate limit reached",
                    "Wait a minute, then try again.")
        if code == 403:
            return ("error", "Gemini denied access for this API project",
                    "Check your key and model access in Google AI Studio.")
        return ("error", "Gemini couldn't complete this request",
                "Try again. If it keeps failing, check the terminal logs.")
    if isinstance(error, (RuntimeError, ValueError)):
        return ("error", "The agent couldn't answer", str(error))
    return ("error", f"Analysis failed ({type(error).__name__})",
            "Check the terminal for details.")


def typewriter(text: str):
    for token in re.findall(r"\S+\s*", text):
        yield token
        time.sleep(0.012)


THINKING_HTML = """
<div class="dl-thinking" role="status">
  <span class="pulse"></span>
  <div class="steps">
    <span>Reading your question</span>
    <span>Choosing a tool</span>
    <span>Running it in Python</span>
  </div>
</div>
"""


def render_message(message: dict, index: int, is_last: bool) -> None:
    if message["role"] == "user":
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(message["text"])
        return

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        if message.get("kind") == "error":
            st.html(
                f'<div class="dl-notice {message.get("level", "error")}">'
                f'<b>{html.escape(message.get("title", "Something went wrong"))}</b>'
                f'<span>{html.escape(message["text"])}</span></div>'
            )
            if is_last and st.button(
                "Try again",
                key=f"retry_{index}",
                icon=":material/refresh:",
            ):
                st.session_state.queued = st.session_state.history[index - 1]["text"]
                st.session_state.history = st.session_state.history[: index - 1]
                st.rerun()
            return

        trace = message.get("trace") or []
        if trace:
            st.html(pipeline_html(trace))

        if message.pop("fresh", False):
            st.write_stream(typewriter(message["text"]))
        else:
            st.markdown(message["text"])

        for chart_index, chart in enumerate(message.get("charts", [])):
            show_chart(chart, key=f"viz_{index}_{chart_index}")

        if trace:
            show_receipts(trace)


# ============================================================
# SIDEBAR: DATASET
# ============================================================

with st.sidebar:
    st.html(
        """
        <div class="dl-brand">
          <span class="logo"></span>
          <div><strong>DataLens</strong><small>Studio</small></div>
        </div>
        """
    )

    st.html('<p class="dl-side-title">Dataset</p>')
    upload = st.file_uploader(
        "Upload a CSV",
        type=["csv"],
        help="Up to 10 MB.",
        label_visibility="collapsed",
    )

    if upload is None:
        samples = sorted((ROOT / "sample_data").glob("*.csv"))
        if not samples:
            st.error("No CSV found in sample_data/.")
            st.stop()
        names = [p.name for p in samples]
        default_name = "students.csv" if "students.csv" in names else names[0]
        st.html('<p class="dl-side-hint">or start with a sample</p>')
        picked = st.segmented_control(
            "Sample dataset",
            options=names,
            default=default_name,
            key="sample_pick",
            label_visibility="collapsed",
        )
        dataset_name = picked or default_name
        dataset_bytes = (ROOT / "sample_data" / dataset_name).read_bytes()
    else:
        dataset_name = upload.name
        dataset_bytes = upload.getvalue()
        if len(dataset_bytes) > MAX_UPLOAD_BYTES:
            st.error("The CSV file must be 10 MB or smaller.")
            st.stop()

    try:
        df = load_csv(BytesIO(dataset_bytes))
    except (ValueError, UnicodeError, pd.errors.ParserError) as error:
        st.error(f"Could not read this CSV: {error}")
        st.stop()

    # Reset the chat whenever the dataset changes.
    fingerprint = f"{dataset_name}:{sha256(dataset_bytes).hexdigest()}"
    if st.session_state.get("dataset_id") != fingerprint:
        st.session_state.dataset_id = fingerprint
        st.session_state.history = []

    filled_pct = completeness(df)
    st.html(
        f"""
        <div class="dl-dataset">
          {completeness_ring(filled_pct)}
          <div>
            <strong title="{html.escape(dataset_name)}">{html.escape(dataset_name)}</strong>
            <span>{len(df):,} rows · {len(df.columns)} columns</span>
            <span>{pct_text(filled_pct)} of cells filled</span>
          </div>
        </div>
        """
    )

    st.html('<p class="dl-side-title">Columns</p>')
    rows = []
    for col in df.columns:
        glyph, color = KIND_STYLE[column_kind(df[col])]
        rows.append(
            f'<div class="dl-col" style="--c:{color}"><span class="g">{glyph}</span>'
            f"<b>{html.escape(str(col))}</b>"
            f"<em>{html.escape(str(df[col].dtype))}</em></div>"
        )
    st.html(f'<div class="dl-cols">{"".join(rows)}</div>')

    if st.button(
        "Clear conversation",
        key="clear",
        icon=":material/restart_alt:",
        width="stretch",
    ):
        st.session_state.history = []
        st.rerun()

    st.html(
        '<p class="dl-side-foot">Gemini picks the tool. '
        "Pandas does the math. Plotly draws it.</p>"
    )


# ============================================================
# MAIN: HEADER
# ============================================================

st.session_state.setdefault("history", [])
history: list[dict] = st.session_state.history

st.html(
    f"""
    <header class="dl-top">
      <div class="dl-title">
        <h1>What do you want to know about <code>{html.escape(dataset_name)}</code>?</h1>
        <p>Ask in plain English. The agent picks a tool, Python does the math,
        and every answer comes with receipts.</p>
      </div>
      <dl class="dl-stats">
        <div><dt>Rows</dt><dd>{len(df):,}</dd></div>
        <div><dt>Columns</dt><dd>{len(df.columns)}</dd></div>
        <div><dt>Filled</dt><dd>{pct_text(filled_pct)}</dd></div>
      </dl>
    </header>
    """
)

# ---------- tool belt ----------

last_trace = next(
    (
        m.get("trace")
        for m in reversed(history)
        if m["role"] == "assistant" and m.get("trace")
    ),
    [],
)
used_tools = {event.get("tool") for event in last_trace}

st.html(
    '<p class="dl-belt-caption">The agent has six tools. '
    "Pick one to run it on your data, or ask your own question below.</p>"
)

prompts = tool_prompts(df)
quick_question = st.session_state.pop("queued", None)

belt = st.columns(len(TOOLS), gap="small")
for column, (name, tool) in zip(belt, TOOLS.items()):
    prompt = prompts.get(name)
    state = "on" if name in used_tools else "off"
    with column:
        if st.button(
            tool["label"],
            key=f"tool-{state}-{name}",
            icon=tool["icon"],
            disabled=prompt is None,
            width="stretch",
            help=(
                f"Asks: {prompt}"
                if prompt
                else "This dataset has no columns this tool can use."
            ),
        ):
            quick_question = prompt
            st.session_state.view = "Analyst"

if used_tools:
    st.html('<p class="dl-belt-note">Lit tools ran in the last answer.</p>')

# ---------- view switch ----------

VIEWS = {
    "Analyst": ":material/forum: Analyst",
    "Dataset": ":material/table_chart: Dataset",
}
view = (
    st.segmented_control(
        "View",
        options=list(VIEWS),
        format_func=VIEWS.get,
        default="Analyst",
        key="view",
        label_visibility="collapsed",
    )
    or "Analyst"
)


# ============================================================
# VIEW: ANALYST
# ============================================================

if view == "Analyst":
    if not history and not quick_question:
        ideas = [
            prompts[n]
            for n in ("group_comparison", "correlation_analysis", "plot_chart")
            if prompts.get(n)
        ]
        st.html(
            """
            <div class="dl-flow">
              <div><b>1</b><span>You ask a question</span></div>
              <i></i>
              <div><b>2</b><span>The agent picks a tool</span></div>
              <i></i>
              <div><b>3</b><span>Python runs the numbers</span></div>
              <i></i>
              <div><b>4</b><span>You get the answer and the receipts</span></div>
            </div>
            """
        )
        if ideas:
            st.html(
                '<p class="dl-ideas-caption">'
                "Not sure where to start? Try one of these.</p>"
            )
            idea_cols = st.columns(len(ideas), gap="small")
            for i, (col, idea) in enumerate(zip(idea_cols, ideas)):
                with col:
                    if st.button(idea, key=f"idea_{i}", width="stretch"):
                        quick_question = idea

    for i, message in enumerate(history):
        render_message(message, i, is_last=(i == len(history) - 1))

    typed = st.chat_input("Ask anything about your data…")
    question = typed or quick_question

    if question:
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(question)

        with st.chat_message("assistant", avatar=BOT_AVATAR):
            thinking = st.empty()
            thinking.html(THINKING_HTML)
            try:
                answer, trace = ask_agent(df, question)
                charts = charts_from_trace(df, trace)
            except Exception as error:  # noqa: BLE001 - surfaced to the user
                thinking.empty()
                level, title, hint = friendly_error(error)
                history.extend(
                    [
                        {"role": "user", "text": question},
                        {
                            "role": "assistant",
                            "kind": "error",
                            "level": level,
                            "title": title,
                            "text": hint,
                        },
                    ]
                )
            else:
                thinking.empty()
                history.extend(
                    [
                        {"role": "user", "text": question},
                        {
                            "role": "assistant",
                            "kind": "answer",
                            "text": answer,
                            "trace": trace,
                            "charts": charts,
                            "fresh": True,
                        },
                    ]
                )
        st.rerun()


# ============================================================
# VIEW: DATASET
# ============================================================

else:
    _, measures, _, _ = split_columns(df)

    st.html('<h3 class="dl-h">Shape of each column</h3>')
    st.html(
        '<div class="col-strip">'
        + "".join(column_card(df, col) for col in df.columns)
        + "</div>"
    )

    if len(measures) >= 2:
        st.html('<h3 class="dl-h">How the numeric columns relate</h3>')
        st.caption(
            "Pearson correlation. Blue means they rise together, "
            "pink means one rises as the other falls."
        )
        st.plotly_chart(
            correlation_heatmap(df, measures),
            key="corr_heatmap",
            width="stretch",
            config={"displayModeBar": False},
        )

    st.html('<h3 class="dl-h">First 100 rows</h3>')
    st.dataframe(df.head(100), width="stretch", hide_index=True)

    st.download_button(
        "Download this CSV",
        data=dataset_bytes,
        file_name=dataset_name,
        mime="text/csv",
        icon=":material/download:",
    )