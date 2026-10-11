
import json

import pandas as pd
import plotly.graph_objects as go


MAX_POINTS = 1000
MAX_CATEGORIES = 20


def create_visualization(
    df: pd.DataFrame,
    chart_type: str,
    x_column: str,
    y_column: str | None = None,
) -> dict:
    """Create a Plotly chart using actual dataset values."""

    supported = {"bar", "histogram", "scatter", "line"}

    if chart_type not in supported:
        raise ValueError(f"Unsupported chart type: {chart_type}")

    if df.empty:
        raise ValueError("Cannot visualize an empty dataset.")

    if x_column not in df.columns:
        raise ValueError(f"Column '{x_column}' does not exist.")

    if chart_type in {"bar", "histogram"} and y_column is not None:
        raise ValueError(
            f"A {chart_type} chart does not require a y_column."
        )

    fig = go.Figure()
    note = ""

    if chart_type == "bar":
        counts = (
            df[x_column]
            .astype("string")
            .fillna("(missing)")
            .value_counts()
            .head(MAX_CATEGORIES)
        )

        fig.add_bar(
            x=counts.index.tolist(),
            y=counts.tolist()
        )

        fig.update_layout(yaxis_title="Row count")
        rows_plotted = int(counts.sum())

        if df[x_column].nunique(dropna=False) > MAX_CATEGORIES:
            note = (
                f"Showing the {MAX_CATEGORIES} "
                "most frequent categories."
            )

    elif chart_type == "histogram":
        series = df[x_column]

        if (
            not pd.api.types.is_numeric_dtype(series)
            or pd.api.types.is_bool_dtype(series)
        ):
            raise ValueError(
                "A histogram requires a numeric x_column."
            )

        values = series.replace(
            [float("inf"), float("-inf")],
            float("nan")
        ).dropna()

        if values.empty:
            raise ValueError("No usable numeric values to plot.")

        fig.add_histogram(
            x=values.tolist(),
            nbinsx=20
        )

        fig.update_layout(yaxis_title="Frequency")
        rows_plotted = len(values)

    else:
        if y_column is None:
            raise ValueError(
                f"A {chart_type} chart requires a y_column."
            )

        if y_column not in df.columns:
            raise ValueError(
                f"Column '{y_column}' does not exist."
            )

        for column in (x_column, y_column):
            series = df[column]

            if (
                not pd.api.types.is_numeric_dtype(series)
                or pd.api.types.is_bool_dtype(series)
            ):
                raise ValueError(
                    f"Column '{column}' must be numeric."
                )

        pairs = (
            df[[x_column, y_column]]
            .replace(
                [float("inf"), float("-inf")],
                float("nan")
            )
            .dropna()
        )

        if pairs.empty:
            raise ValueError(
                "No complete numeric pairs to plot."
            )

        if len(pairs) > MAX_POINTS:
            pairs = pairs.sample(
                n=MAX_POINTS,
                random_state=42
            )
            note = (
                f"Showing a reproducible sample "
                f"of {MAX_POINTS} data points."
            )

        if chart_type == "line":
            pairs = pairs.sort_values(x_column)

        fig.add_scatter(
            x=pairs[x_column].tolist(),
            y=pairs[y_column].tolist(),
            mode=(
                "lines+markers"
                if chart_type == "line"
                else "markers"
            )
        )

        fig.update_layout(yaxis_title=y_column)
        rows_plotted = len(pairs)

    fig.update_layout(
        title=f"{chart_type.title()} chart: {x_column}",
        xaxis_title=x_column,
        template="plotly_white"
    )

    return {
        "chart_type": chart_type,
        "x_column": x_column,
        "y_column": y_column,
        "rows_plotted": int(rows_plotted),
        "total_rows": int(len(df)),
        "note": note,
        "figure": json.loads(fig.to_json()),
    }
