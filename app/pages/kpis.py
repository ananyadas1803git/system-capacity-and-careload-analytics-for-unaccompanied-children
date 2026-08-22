"""KPI scorecard page for the HHS UAC capacity analytics dashboard.

Run directly with::

    streamlit run app/pages/kpis.py
"""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app_utils import (  # noqa: E402
    BACKLOG_STREAK_COLUMN,
    CBP_COLUMN,
    DISCHARGE_ANOMALY_COLUMN,
    DISCHARGE_COLUMN,
    GROWTH_RATE_COLUMN,
    HHS_COLUMN,
    NET_INTAKE_COLUMN,
    OFFSET_RATIO_COLUMN,
    ROLLING_14_COLUMN,
    ROLLING_7_COLUMN,
    TOTAL_LOAD_COLUMN,
    TRANSFER_ANOMALY_COLUMN,
    TRANSFER_COLUMN,
    DataValidationError,
    calculate_kpis,
    compute_capacity_metrics,
    generate_mock_data,
    validate_and_clean_data,
)


NAVY = "#163B65"
SLATE_950 = "#172033"
SLATE_700 = "#3F5063"
SLATE_500 = "#66788A"
SLATE_200 = "#D9E1EA"
WHITE = "#FFFFFF"
BLUE = "#2E75B6"
TEAL = "#167C80"
AMBER = "#D97706"
RED = "#B42318"
GREEN = "#238636"
PURPLE = "#7656A6"

ROLLING_VOLATILITY_COLUMN = "Rolling Care Load Volatility"


st.set_page_config(
    page_title="KPI Scorecard | HHS UAC Analytics",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_page_styles() -> None:
    """Apply the common government-analytics visual theme."""
    st.markdown(
        f"""
        <style>
            .stApp {{ background-color: #F5F7FA; color: {SLATE_950}; }}
            [data-testid="stSidebar"] {{
                background-color: #EDF2F7;
                border-right: 1px solid {SLATE_200};
            }}
            .page-header {{
                background: linear-gradient(110deg, {SLATE_950}, {NAVY});
                border-radius: 10px;
                color: {WHITE};
                margin-bottom: 1rem;
                padding: 1.25rem 1.5rem;
            }}
            .page-header h1 {{
                color: {WHITE};
                font-size: 1.8rem;
                margin: 0 0 .35rem 0;
            }}
            .page-header p {{ color: #DCE7F2; margin: 0; }}
            div[data-testid="stMetric"] {{
                background-color: {WHITE};
                border: 1px solid {SLATE_200};
                border-left: 4px solid {PURPLE};
                border-radius: 8px;
                min-height: 135px;
                padding: .9rem 1rem;
            }}
            div[data-testid="stMetricLabel"] {{ color: {SLATE_500}; }}
            div[data-testid="stMetricValue"] {{ color: {NAVY}; }}
            .status-strip {{
                background-color: {WHITE};
                border: 1px solid {SLATE_200};
                border-radius: 7px;
                color: {SLATE_700};
                margin: .5rem 0 1rem 0;
                padding: .8rem 1rem;
            }}
            .method-note {{
                background-color: #EFF6FF;
                border: 1px solid #B9D5F0;
                border-left: 5px solid {BLUE};
                border-radius: 7px;
                color: #173A5E;
                margin-bottom: 1rem;
                padding: .8rem 1rem;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_uploaded_csv(file_bytes: bytes) -> pd.DataFrame:
    """Read uploaded CSV bytes using string-first parsing."""
    try:
        return pd.read_csv(
            BytesIO(file_bytes),
            dtype=str,
            encoding="utf-8-sig",
        )
    except (
        UnicodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as exc:
        raise DataValidationError(f"The uploaded CSV could not be read: {exc}") from exc


@st.cache_data(show_spinner=False)
def load_mock_data() -> pd.DataFrame:
    """Return the cached deterministic mock dataset."""
    return generate_mock_data()


def add_rolling_volatility(
    metrics: pd.DataFrame,
    volatility_window: int,
) -> pd.DataFrame:
    """Add a rolling population standard deviation of daily growth rates."""
    frame = metrics.copy()
    growth = pd.to_numeric(frame[GROWTH_RATE_COLUMN], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    frame[ROLLING_VOLATILITY_COLUMN] = growth.rolling(
        volatility_window,
        min_periods=2,
    ).std(ddof=0)
    return frame


def aggregate_for_chart(metrics: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """Aggregate daily metrics while preserving KPI measurement semantics."""
    if granularity == "Daily":
        return metrics.copy()

    frequency = pd.offsets.Week(weekday=6) if granularity == "Weekly" else pd.offsets.MonthEnd()
    last_columns = [
        CBP_COLUMN,
        HHS_COLUMN,
        TOTAL_LOAD_COLUMN,
        GROWTH_RATE_COLUMN,
        ROLLING_7_COLUMN,
        ROLLING_14_COLUMN,
        BACKLOG_STREAK_COLUMN,
        ROLLING_VOLATILITY_COLUMN,
    ]
    sum_columns = [TRANSFER_COLUMN, DISCHARGE_COLUMN, NET_INTAKE_COLUMN]
    aggregations = {
        **{column: "last" for column in last_columns if column in metrics},
        **{column: "sum" for column in sum_columns if column in metrics},
    }
    aggregated = metrics.resample(frequency).agg(aggregations).dropna(how="all")
    aggregated[OFFSET_RATIO_COLUMN] = aggregated[DISCHARGE_COLUMN].div(
        aggregated[TRANSFER_COLUMN] + 1
    )
    return aggregated


def longest_positive_streak(values: pd.Series) -> int:
    """Calculate the longest consecutive run of positive numeric values."""
    positive = pd.to_numeric(values, errors="coerce").fillna(0).gt(0)
    groups = positive.ne(positive.shift(fill_value=False)).cumsum()
    streaks = positive.groupby(groups).cumsum()
    return int(streaks.max()) if not streaks.empty else 0


def comparison_values(
    metrics: pd.DataFrame,
    comparison_days: int,
) -> dict[str, float]:
    """Calculate current-versus-prior comparison-window KPI changes."""
    if metrics.empty:
        return {
            "load_delta": 0,
            "net_delta": 0,
            "volatility_delta": 0,
            "backlog_delta": 0,
            "offset_delta": 0,
        }

    effective_window = min(comparison_days, max(1, len(metrics) // 2))
    current = metrics.iloc[-effective_window:]
    previous = metrics.iloc[-2 * effective_window : -effective_window]
    if previous.empty:
        previous = metrics.iloc[:effective_window]

    latest_load = float(current[TOTAL_LOAD_COLUMN].iloc[-1])
    previous_load = float(previous[TOTAL_LOAD_COLUMN].iloc[-1])
    current_net = float(current[NET_INTAKE_COLUMN].mean())
    previous_net = float(previous[NET_INTAKE_COLUMN].mean())
    current_volatility = float(
        current[GROWTH_RATE_COLUMN].replace([np.inf, -np.inf], np.nan).std(ddof=0)
    )
    previous_volatility = float(
        previous[GROWTH_RATE_COLUMN].replace([np.inf, -np.inf], np.nan).std(ddof=0)
    )
    current_backlog = longest_positive_streak(current[NET_INTAKE_COLUMN])
    previous_backlog = longest_positive_streak(previous[NET_INTAKE_COLUMN])
    current_offset = float(current[DISCHARGE_COLUMN].sum() / (current[TRANSFER_COLUMN].sum() + 1))
    previous_offset = float(
        previous[DISCHARGE_COLUMN].sum() / (previous[TRANSFER_COLUMN].sum() + 1)
    )

    return {
        "load_delta": latest_load - previous_load,
        "net_delta": current_net - previous_net,
        "volatility_delta": (
            current_volatility - previous_volatility
            if np.isfinite(current_volatility) and np.isfinite(previous_volatility)
            else 0
        ),
        "backlog_delta": float(current_backlog - previous_backlog),
        "offset_delta": current_offset - previous_offset,
    }


def build_kpi_summary(metrics: pd.DataFrame, kpis: dict[str, float | int]) -> pd.DataFrame:
    """Build a compact selected-period summary for display and download."""
    rows = [
        {
            "KPI": "Total Children Under Care",
            "Latest": int(kpis["total_children_under_care"]),
            "Period Average": float(metrics[TOTAL_LOAD_COLUMN].mean()),
            "Period Minimum": float(metrics[TOTAL_LOAD_COLUMN].min()),
            "Period Maximum": float(metrics[TOTAL_LOAD_COLUMN].max()),
            "Unit": "children",
        },
        {
            "KPI": "Net Intake Pressure",
            "Latest": int(kpis["net_intake_pressure"]),
            "Period Average": float(metrics[NET_INTAKE_COLUMN].mean()),
            "Period Minimum": float(metrics[NET_INTAKE_COLUMN].min()),
            "Period Maximum": float(metrics[NET_INTAKE_COLUMN].max()),
            "Unit": "children/day",
        },
        {
            "KPI": "Care Load Volatility Index",
            "Latest": float(kpis["care_load_volatility_index"]),
            "Period Average": float(metrics[ROLLING_VOLATILITY_COLUMN].mean(skipna=True)),
            "Period Minimum": float(metrics[ROLLING_VOLATILITY_COLUMN].min(skipna=True)),
            "Period Maximum": float(metrics[ROLLING_VOLATILITY_COLUMN].max(skipna=True)),
            "Unit": "%",
        },
        {
            "KPI": "Backlog Accumulation Rate",
            "Latest": int(metrics[BACKLOG_STREAK_COLUMN].iloc[-1]),
            "Period Average": float(metrics[BACKLOG_STREAK_COLUMN].mean()),
            "Period Minimum": float(metrics[BACKLOG_STREAK_COLUMN].min()),
            "Period Maximum": int(kpis["backlog_accumulation_rate"]),
            "Unit": "consecutive days",
        },
        {
            "KPI": "Discharge Offset Ratio",
            "Latest": float(kpis["discharge_offset_ratio"]),
            "Period Average": float(metrics[OFFSET_RATIO_COLUMN].mean()),
            "Period Minimum": float(metrics[OFFSET_RATIO_COLUMN].min()),
            "Period Maximum": float(metrics[OFFSET_RATIO_COLUMN].max()),
            "Unit": "ratio",
        },
    ]
    return pd.DataFrame.from_records(rows).replace([np.inf, -np.inf], np.nan).fillna(0)


def chart_layout(title: str, y_title: str, *, height: int = 440) -> dict:
    """Return shared Plotly layout settings."""
    return {
        "title": {"text": title, "font": {"size": 18, "color": NAVY}},
        "height": height,
        "paper_bgcolor": WHITE,
        "plot_bgcolor": WHITE,
        "font": {"family": "Arial, sans-serif", "color": SLATE_700},
        "hovermode": "x unified",
        "margin": {"l": 60, "r": 25, "t": 70, "b": 50},
        "legend": {"orientation": "h", "x": 0, "y": 1.08},
        "xaxis": {"title": "Reporting date", "showgrid": False},
        "yaxis": {
            "title": y_title,
            "gridcolor": SLATE_200,
            "separatethousands": True,
        },
    }


def render_kpi_cards(
    kpis: dict[str, float | int],
    comparisons: dict[str, float],
    comparison_days: int,
) -> None:
    """Render five KPI cards with comparison-window deltas and tooltips."""
    columns = st.columns(5)
    columns[0].metric(
        "Total Children Under Care",
        f"{kpis['total_children_under_care']:,.0f}",
        delta=f"{comparisons['load_delta']:+,.0f} vs prior endpoint",
        delta_color="inverse",
        help="Latest CBP active custody plus latest HHS active care load.",
    )
    columns[1].metric(
        "Net Intake Pressure",
        f"{kpis['net_intake_pressure']:+,.0f}",
        delta=f"{comparisons['net_delta']:+.1f} vs prior mean",
        delta_color="inverse",
        help=f"Latest transfers minus discharges; comparison uses {comparison_days}-day means.",
    )
    columns[2].metric(
        "Care Load Volatility",
        f"{kpis['care_load_volatility_index']:.2f}%",
        delta=f"{comparisons['volatility_delta']:+.2f} pp",
        delta_color="inverse",
        help="Population standard deviation of daily Total System Load growth.",
    )
    columns[3].metric(
        "Backlog Accumulation",
        f"{kpis['backlog_accumulation_rate']:,.0f} day(s)",
        delta=f"{comparisons['backlog_delta']:+.0f} vs prior window",
        delta_color="inverse",
        help="Longest positive-net-intake streak in the selected reporting period.",
    )
    columns[4].metric(
        "Discharge Offset Ratio",
        f"{kpis['discharge_offset_ratio']:.1%}",
        delta=f"{comparisons['offset_delta']:+.1%} vs prior window",
        help="Latest discharges divided by latest transfers plus one.",
    )


def render_load_and_pressure(chart_data: pd.DataFrame, granularity: str) -> None:
    """Render load and net-intake trends in a two-row KPI chart."""
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("Total System Load", "Net Intake Pressure"),
    )
    figure.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data[TOTAL_LOAD_COLUMN],
            name="Total System Load",
            mode="lines",
            line={"color": NAVY, "width": 2.3},
            hovertemplate="%{y:,.0f} children<extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data[ROLLING_7_COLUMN],
            name="7-day average",
            mode="lines",
            line={"color": BLUE, "width": 1.8, "dash": "dot"},
            hovertemplate="%{y:,.1f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    net_intake = pd.to_numeric(chart_data[NET_INTAKE_COLUMN], errors="coerce").fillna(0)
    figure.add_trace(
        go.Bar(
            x=chart_data.index,
            y=net_intake,
            name="Net intake",
            marker_color=np.where(net_intake.gt(0), RED, GREEN),
            hovertemplate="%{y:+,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    figure.add_hline(y=0, line_color=SLATE_500, line_width=1, row=2, col=1)
    figure.update_layout(
        title={
            "text": f"Load & Intake KPI Trends ({granularity})",
            "font": {"size": 18, "color": NAVY},
        },
        height=650,
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        font={"family": "Arial, sans-serif", "color": SLATE_700},
        hovermode="x unified",
        margin={"l": 60, "r": 25, "t": 80, "b": 50},
        legend={"orientation": "h", "x": 0, "y": 1.06},
    )
    figure.update_xaxes(showgrid=False, title_text="Reporting date", row=2, col=1)
    figure.update_yaxes(gridcolor=SLATE_200, rangemode="tozero")
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})


def render_volatility_backlog_chart(
    daily_metrics: pd.DataFrame,
    volatility_window: int,
) -> None:
    """Render rolling volatility and backlog streak with separate axes."""
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scatter(
            x=daily_metrics.index,
            y=daily_metrics[ROLLING_VOLATILITY_COLUMN],
            name=f"{volatility_window}-day volatility",
            mode="lines",
            line={"color": PURPLE, "width": 2},
            hovertemplate="%{y:.2f}%<extra></extra>",
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=daily_metrics.index,
            y=daily_metrics[BACKLOG_STREAK_COLUMN],
            name="Backlog streak",
            mode="lines",
            fill="tozeroy",
            line={"color": AMBER, "width": 1.8},
            fillcolor="rgba(217, 119, 6, 0.14)",
            hovertemplate="%{y:,.0f} day(s)<extra></extra>",
        ),
        secondary_y=True,
    )
    figure.update_layout(**chart_layout("Volatility & Backlog Accumulation", "Volatility (%)"))
    figure.update_yaxes(title_text="Volatility (%)", ticksuffix="%", secondary_y=False)
    figure.update_yaxes(
        title_text="Consecutive backlog days",
        rangemode="tozero",
        secondary_y=True,
    )
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})


def render_offset_chart(chart_data: pd.DataFrame, granularity: str) -> None:
    """Render the discharge offset ratio against its 100% balance reference."""
    ratio = pd.to_numeric(chart_data[OFFSET_RATIO_COLUMN], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    figure = go.Figure(
        go.Scatter(
            x=chart_data.index,
            y=ratio.mul(100),
            name="Discharge Offset Ratio",
            mode="lines",
            line={"color": TEAL, "width": 2},
            fill="tozeroy",
            fillcolor="rgba(22, 124, 128, 0.12)",
            hovertemplate="%{y:.1f}%<extra></extra>",
        )
    )
    figure.add_hline(
        y=100,
        line_color=NAVY,
        line_dash="dash",
        annotation_text="Flow balance: 100%",
        annotation_position="top left",
    )
    figure.update_layout(
        **chart_layout(f"Discharge Offset Ratio ({granularity})", "Offset ratio (%)")
    )
    figure.update_yaxes(rangemode="tozero", ticksuffix="%")
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})


def methodology_table() -> pd.DataFrame:
    """Return KPI definitions, formulas, and interpretation guidance."""
    return pd.DataFrame(
        [
            {
                "KPI": "Total Children Under Care",
                "Formula": "CBP active custody + HHS active care",
                "Interpretation": "Current combined care load across both systems.",
            },
            {
                "KPI": "Net Intake Pressure",
                "Formula": "Transfers out of CBP − HHS discharges",
                "Interpretation": "Positive values indicate intake exceeding placements.",
            },
            {
                "KPI": "Care Load Volatility Index",
                "Formula": "Population SD of daily Total System Load growth (%)",
                "Interpretation": "Higher values indicate less predictable daily load changes.",
            },
            {
                "KPI": "Backlog Accumulation Rate",
                "Formula": "Longest consecutive run where Net Daily Intake > 0",
                "Interpretation": "Duration of the longest sustained pressure episode.",
            },
            {
                "KPI": "Discharge Offset Ratio",
                "Formula": "HHS discharges ÷ (transfers + 1)",
                "Interpretation": "100% or higher means discharges match or exceed transfers.",
            },
        ]
    )


def render_validation_log(selected_data: pd.DataFrame) -> None:
    """Display selected-period logical constraint violations."""
    with st.expander("KPI Data Quality & Validation Log", expanded=False):
        anomaly_columns = [
            TRANSFER_ANOMALY_COLUMN,
            DISCHARGE_ANOMALY_COLUMN,
        ]
        missing = [column for column in anomaly_columns if column not in selected_data]
        if missing:
            st.warning("Anomaly fields are unavailable: " + ", ".join(missing))
            return

        flagged = selected_data.loc[selected_data[anomaly_columns].fillna(False).any(axis=1)]
        if flagged.empty:
            st.success("No logical constraint violations were found in this period.")
            return

        st.error(f"{len(flagged):,} anomalous row(s) require review.")
        st.dataframe(
            flagged[
                [
                    CBP_COLUMN,
                    TRANSFER_COLUMN,
                    HHS_COLUMN,
                    DISCHARGE_COLUMN,
                    *anomaly_columns,
                ]
            ].reset_index(),
            width="stretch",
            hide_index=True,
        )


def main() -> None:
    """Build the complete KPI scorecard page."""
    apply_page_styles()
    st.markdown(
        """
        <div class="page-header">
            <h1>Capacity KPI Scorecard & Methodology</h1>
            <p>Monitor the five core HHS UAC capacity indicators, compare recent
            performance windows, and review calculation methodology.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("KPI Controls")
        uploaded_file = st.file_uploader(
            "Upload HHS capacity data",
            type=["csv"],
            help="Leave empty to use the synthetic 2023-2025 dataset.",
            key="kpi_csv_uploader",
        )

    try:
        if uploaded_file is None:
            raw_data = load_mock_data()
            source_name = "Synthetic 2023-2025 demonstration data"
            with st.sidebar:
                st.info("No CSV uploaded. Using the generated mock dataset.")
        else:
            raw_data = load_uploaded_csv(uploaded_file.getvalue())
            source_name = uploaded_file.name
        cleaned_data = validate_and_clean_data(raw_data)
    except (DataValidationError, TypeError, ValueError) as exc:
        st.error(f"Unable to prepare KPI data: {exc}")
        st.stop()
    except Exception as exc:
        st.exception(exc)
        st.stop()

    minimum_date = cleaned_data.index.min().date()
    maximum_date = cleaned_data.index.max().date()
    with st.sidebar:
        selected_dates = st.date_input(
            "Reporting period",
            value=(minimum_date, maximum_date),
            min_value=minimum_date,
            max_value=maximum_date,
            key="kpi_date_range",
        )
        granularity = st.selectbox(
            "Chart granularity",
            ["Daily", "Weekly", "Monthly"],
            key="kpi_granularity",
        )
        comparison_days = st.selectbox(
            "Comparison window",
            [7, 14, 30, 60, 90],
            index=2,
            format_func=lambda value: f"{value} days",
        )
        volatility_window = st.slider(
            "Rolling volatility window",
            min_value=7,
            max_value=60,
            value=30,
            help="Window used only for the volatility trend chart.",
        )
        st.divider()
        st.caption(f"Data source: {source_name}")

    if not isinstance(selected_dates, (tuple, list)) or len(selected_dates) != 2:
        st.info("Select both a start and end date to continue.")
        st.stop()

    start_date, end_date = selected_dates
    if start_date > end_date:
        st.error("The reporting-period start must be on or before the end.")
        st.stop()

    selected_data = cleaned_data.loc[pd.Timestamp(start_date) : pd.Timestamp(end_date)].copy()
    if selected_data.empty:
        st.warning("No data is available for the selected reporting period.")
        st.stop()

    try:
        daily_metrics = compute_capacity_metrics(selected_data)
        daily_metrics = add_rolling_volatility(daily_metrics, volatility_window)
        kpis = calculate_kpis(daily_metrics)
        comparisons = comparison_values(daily_metrics, comparison_days)
        chart_data = aggregate_for_chart(daily_metrics, granularity)
        summary = build_kpi_summary(daily_metrics, kpis)
    except (DataValidationError, TypeError, ValueError) as exc:
        st.error(f"Unable to calculate KPI analytics: {exc}")
        st.stop()

    st.caption(
        f"Reporting period: {start_date:%d %b %Y}–{end_date:%d %b %Y} "
        f"• {len(selected_data):,} daily observations • {granularity} charts"
    )
    render_kpi_cards(kpis, comparisons, comparison_days)

    latest_net = int(kpis["net_intake_pressure"])
    longest_backlog = int(kpis["backlog_accumulation_rate"])
    offset_ratio = float(kpis["discharge_offset_ratio"])
    status_items = [
        "Intake pressure is positive" if latest_net > 0 else "Latest day shows net relief",
        (
            f"Longest backlog episode: {longest_backlog} day(s)"
            if longest_backlog
            else "No positive-intake backlog episode"
        ),
        (
            "Latest discharge flow offsets transfers"
            if offset_ratio >= 1
            else "Latest discharges do not fully offset transfers"
        ),
    ]
    st.markdown(
        '<div class="status-strip"><strong>Current KPI signals:</strong> '
        + " &nbsp;•&nbsp; ".join(status_items)
        + "</div>",
        unsafe_allow_html=True,
    )

    scorecard_tab, trends_tab, methodology_tab = st.tabs(
        ["KPI Scorecard", "KPI Trends", "Definitions & Methodology"]
    )
    with scorecard_tab:
        render_load_and_pressure(chart_data, granularity)
        st.subheader("Selected-Period KPI Summary")
        st.dataframe(
            summary,
            width="stretch",
            hide_index=True,
            column_config={
                "Latest": st.column_config.NumberColumn(format="%.2f"),
                "Period Average": st.column_config.NumberColumn(format="%.2f"),
                "Period Minimum": st.column_config.NumberColumn(format="%.2f"),
                "Period Maximum": st.column_config.NumberColumn(format="%.2f"),
            },
        )
        st.download_button(
            "Download KPI summary",
            data=summary.to_csv(index=False).encode("utf-8"),
            file_name=f"uac_kpi_summary_{start_date}_{end_date}.csv",
            mime="text/csv",
        )
    with trends_tab:
        render_volatility_backlog_chart(daily_metrics, volatility_window)
        render_offset_chart(chart_data, granularity)
    with methodology_tab:
        st.markdown(
            """
            <div class="method-note">
                <strong>Calculation policy:</strong> KPI cards use daily data within
                the selected reporting period. Chart granularity changes visual
                aggregation only and never redefines consecutive-day or volatility KPIs.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(
            methodology_table(),
            width="stretch",
            hide_index=True,
        )
        st.markdown(
            """
            **Interpretation safeguards**

            - These indicators describe operational load and flow; they do not forecast demand.
            - The first daily growth observation is undefined because no prior selected day exists.
            - A denominator adjustment of `+1` prevents division by zero in the offset ratio.
            - Missing dates and logical constraint violations remain visible through validation flags.
            """
        )

    render_validation_log(selected_data)
    st.caption(
        "Decision-support scorecard. Validate operational conclusions against "
        "authoritative HHS and CBP source publications."
    )


if __name__ == "__main__":
    main()
