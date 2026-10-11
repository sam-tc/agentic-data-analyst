import json
import math
from pathlib import Path

import pandas as pd
from plotly.graph_objects import Figure

from .tools.profile import profile_dataset as _profile_dataset
from .tools.filtering import filter_rows as _filter_rows
from .tools.aggregation import aggregate_column as _aggregate_column
from .tools.grouping import compare_groups as _compare_groups
from .tools.correlation import calculate_correlation as _calculate_correlation
from .tools.visualization import create_visualization as _create_visualization


def build_tools(df: pd.DataFrame, trace: list, output_dir=Path("outputs")):
    """Adapt our six data tools for LLM function calling."""

    output_dir = Path(output_dir)

    def execute(name, args, operation):
        try:
            result = operation()
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            result = {"error": str(error)}

        trace.append({
            "tool": name,
            "arguments": args,
            "result": result,
        })

        return result

    def dataset_profile() -> dict:
        """Describe dataset columns, data types, missing values and numeric ranges."""
        return execute(
            "dataset_profile",
            {},
            lambda: _profile_dataset(df)
        )

    def row_filter(column: str, operator: str, value: str) -> dict:
        """Find rows matching a condition. Operators: ==, !=, >, >=, <, <=, contains. Pass value as text."""

        def run():
            selected = value

            if (
                column in df.columns
                and operator != "contains"
                and pd.api.types.is_numeric_dtype(df[column])
                and not pd.api.types.is_bool_dtype(df[column])
            ):
                selected = float(value)

                if not math.isfinite(selected):
                    raise ValueError(
                        "Numeric comparisons require a finite value."
                    )

            matches = _filter_rows(df, column, operator, selected)

            return {
                "rows_matched": int(len(matches)),
                "rows_shown": min(len(matches), 20),
                "rows": json.loads(
                    matches.head(20).to_json(
                        orient="records",
                        date_format="iso"
                    )
                ),
            }

        return execute(
            "row_filter",
            {"column": column, "operator": operator, "value": value},
            run
        )

    def column_aggregation(column: str, operation: str) -> dict:
        """Calculate count, sum, mean, median, min, or max for one column."""
        return execute(
            "column_aggregation",
            {"column": column, "operation": operation},
            lambda: _aggregate_column(df, column, operation)
        )

    def group_comparison(
        group_column: str,
        operation: str,
        value_column: str = ""
    ) -> dict:
        """Compare categories using count, sum, mean, median, min or max. The value_column is required except for count."""
        return execute(
            "group_comparison",
            {
                "group_column": group_column,
                "operation": operation,
                "value_column": value_column,
            },
            lambda: _compare_groups(
                df, group_column, operation, value_column or None
            )
        )

    def correlation_analysis(column_x: str, column_y: str) -> dict:
        """Calculate Pearson correlation between two different numeric columns."""
        return execute(
            "correlation_analysis",
            {"column_x": column_x, "column_y": column_y},
            lambda: _calculate_correlation(df, column_x, column_y)
        )

    def plot_chart(
        chart_type: str,
        x_column: str,
        y_column: str = ""
    ) -> dict:
        """Create a bar, histogram, scatter or line chart. Scatter and line charts require y_column."""

        def run():
            result = _create_visualization(
                df, chart_type, x_column, y_column or None
            )

            output_dir.mkdir(parents=True, exist_ok=True)

            chart_path = output_dir / f"chart_{len(trace) + 1}.html"

            Figure(result["figure"]).write_html(
                str(chart_path),
                include_plotlyjs=True
            )

            return {
                **{
                    key: value
                    for key, value in result.items()
                    if key != "figure"
                },
                "chart_file": str(chart_path),
            }

        return execute(
            "plot_chart",
            {
                "chart_type": chart_type,
                "x_column": x_column,
                "y_column": y_column,
            },
            run
        )

    return [
        dataset_profile,
        row_filter,
        column_aggregation,
        group_comparison,
        correlation_analysis,
        plot_chart,
    ]
