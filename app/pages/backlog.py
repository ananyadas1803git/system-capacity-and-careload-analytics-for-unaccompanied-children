"""Backlog pressure analysis page for the HHS UAC capacity dashboard.

This page can be launched as part of a Streamlit multipage application or
directly with::

    streamlit run app/pages/backlog.py
"""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# Allow the page to run directly while keeping the shared analytics module at
# the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app_utils import (  # noqa: E402
    BACKLOG_STREAK_COLUMN,
    CBP_COLUMN,
    DISCHARGE_ANOMALY_COLUMN,
    DISCHARGE_COLUMN,
    HHS_COLUMN,
    NET_INTAKE_COLUMN,
    TOTAL_LOAD_COLUMN,
    TRANSFER_ANOMALY_COLUMN,
    TRANSFER_COLUMN,
    DataValidationError,
    calculate_kpis,
    compute_capacity_metrics,
    generate_mock_data,
    validate_and_clean_data,
)


# Government-analytics palette shared conceptually with the main dashboard.
NAVY = "#163B65"
SLATE_950 = "#172033"
SLATE_700 = "#3F5063"
SLATE_500 = "#66788A"
SLATE_200 = "#D9E1EA"
WHITE = "#FFFFFF"
PRESSURE_RED = "#B42318"
PRESSURE_ORANGE = "#E16B26"
RELIEF_GREEN = "#238636"
TEAL = "#167C80"
BLUE = "#2E75B6"


st.set_page_config(
    page_title="Backlog Pressure | HHS UAC Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_page_styles() -> None:
    """Apply an accessible slate/navy theme to this page."""
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
                border-left: 4px solid {PRESSURE_ORANGE};
                border-radius: 8px;
                min-height: 126px;
                padding: .9rem 1rem;
            }}
            div[data-testid="stMetricLabel"] {{ color: {SLATE_500}; }}
            div[data-testid="stMetricValue"] {{ color: {NAVY}; }}
            .pressure-callout {{
                background-color: #FFF1F0;
                border: 1px solid #F2B8B5;
                border-left: 5px solid {PRESSURE_RED};
                border-radius: 7px;
                color: #7A271A;
                margin: .5rem 0 1rem 0;
                padding: .8rem 1rem;
            }}
            .relief-callout {{
                background-color: #ECFDF3;
                border: 1px solid #ABEFC6;
                border-left: 5px solid {RELIEF_GREEN};
                border-radius: 7px;
                color: #05603A;
                margin: .5rem 0 1rem 0;
                padding: .8rem 1rem;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_uploaded_csv(file_bytes: bytes) -> pd.DataFrame:
    """Read uploaded CSV bytes with predictable string-first parsing."""
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
    """Return the cached, deterministic 2023-2025 mock dataset."""
    return generate_mock_data()


def calculate_current_streak(metrics: pd.DataFrame) -> int:
    """Return the positive-net-intake streak active on the latest date."""
    streak = 0
    positive = pd.to_numeric(metrics[NET_INTAKE_COLUMN], errors="coerce").fillna(0).gt(0)
    for is_positive in reversed(positive.tolist()):
        if not is_positive:
            break
        streak += 1
    return streak


def build_backlog_episodes(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize every contiguous positive-net-intake episode.

    Returns one row per pressure episode, including duration, cumulative net
    intake, peak daily pressure, mean daily pressure, and whether the episode is
    active on the latest selected date.
    """
    if metrics.empty:
        return pd.DataFrame()

    working = metrics[[NET_INTAKE_COLUMN]].copy().sort_index()
    working["Is Positive"] = working[NET_INTAKE_COLUMN].fillna(0).gt(0)
    working["Episode Group"] = working["Is Positive"].ne(
        working["Is Positive"].shift(fill_value=False)
    ).cumsum()
    positive_rows = working.loc[working["Is Positive"]].copy()
    if positive_rows.empty:
        return pd.DataFrame(
            columns=[
                "Episode Start",
                "Episode End",
                "Duration (Days)",
                "Cumulative Net Intake",
                "Peak Daily Pressure",
                "Average Daily Pressure",
                "Status",
            ]
        )

    latest_date = working.index.max()
    records: list[dict[str, object]] = []
    for _, episode in positive_rows.groupby("Episode Group"):
        start = episode.index.min()
        end = episode.index.max()
        net_intake = pd.to_numeric(episode[NET_INTAKE_COLUMN], errors="coerce").fillna(0)
        records.append(
            {
                "Episode Start": start,
                "Episode End": end,
                "Duration (Days)": int(len(episode)),
                "Cumulative Net Intake": int(net_intake.sum()),
                "Peak Daily Pressure": int(net_intake.max()),
                "Average Daily Pressure": float(net_intake.mean()),
                "Status": "Active" if end == latest_date else "Closed",
            }
        )

    return pd.DataFrame.from_records(records).sort_values(
        ["Duration (Days)", "Cumulative Net Intake"],
        ascending=[False, False],
    )


def aggregate_for_chart(metrics: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """Aggregate daily metrics for display without changing daily KPIs."""
    if granularity == "Daily":
        return metrics.copy()

    frequency = (
        pd.offsets.Week(weekday=6)
        if granularity == "Weekly"
        else pd.offsets.MonthEnd()
    )
    aggregations = {
        TRANSFER_COLUMN: "sum",
        DISCHARGE_COLUMN: "sum",
        NET_INTAKE_COLUMN: "sum",
        TOTAL_LOAD_COLUMN: "last",
        CBP_COLUMN: "last",
        HHS_COLUMN: "last",
        BACKLOG_STREAK_COLUMN: "last",
    }
    available = {
        column: operation
        for column, operation in aggregations.items()
        if column in metrics.columns
    }
    return metrics.resample(frequency).agg(available).dropna(how="all")


def high_pressure_intervals(
    episodes: pd.DataFrame,
    threshold_days: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return episode intervals from the risk-threshold day onward."""
    if episodes.empty:
        return []

    qualifying = episodes.loc[episodes["Duration (Days)"] >= threshold_days]
    return [
        (
            pd.Timestamp(row["Episode Start"]) + pd.Timedelta(days=threshold_days - 1),
            pd.Timestamp(row["Episode End"]),
        )
        for _, row in qualifying.iterrows()
    ]


def chart_layout(title: str, y_title: str, *, height: int = 460) -> dict:
    """Return a consistent Plotly layout for the page."""
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


def render_pressure_timeline(
    chart_data: pd.DataFrame,
    episodes: pd.DataFrame,
    threshold_days: int,
    granularity: str,
) -> None:
    """Render net-intake pressure with prolonged-backlog shading."""
    net_intake = pd.to_numeric(chart_data[NET_INTAKE_COLUMN], errors="coerce").fillna(0)
    colors = np.where(net_intake.gt(0), PRESSURE_RED, RELIEF_GREEN)
    figure = go.Figure(
        go.Bar(
            x=chart_data.index,
            y=net_intake,
            name="Net intake",
            marker_color=colors,
            hovertemplate="%{y:+,.0f} children<extra></extra>",
        )
    )
    for position, (start, end) in enumerate(
        high_pressure_intervals(episodes, threshold_days)
    ):
        figure.add_vrect(
            x0=start - pd.Timedelta(hours=12),
            x1=end + pd.Timedelta(hours=12),
            fillcolor=PRESSURE_ORANGE,
            opacity=0.12,
            line_width=0,
            annotation_text="Elevated backlog" if position == 0 else None,
            annotation_position="top left",
        )
    figure.add_hline(y=0, line_color=SLATE_500, line_width=1)
    figure.update_layout(
        **chart_layout(
            f"Net Intake Pressure Timeline ({granularity})",
            "Transfers minus discharges",
        )
    )
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})
    st.caption(
        "Red bars indicate positive pressure; green bars indicate relief. "
        f"Shading starts on day {threshold_days} of a continuous pressure episode."
    )


def render_flow_balance(chart_data: pd.DataFrame, granularity: str) -> None:
    """Render transfer inflow against HHS discharge outflow."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data[TRANSFER_COLUMN],
            name="Transfers into HHS",
            mode="lines",
            line={"color": PRESSURE_ORANGE, "width": 2},
            hovertemplate="%{y:,.0f} children<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data[DISCHARGE_COLUMN],
            name="Discharges from HHS",
            mode="lines",
            line={"color": TEAL, "width": 2},
            hovertemplate="%{y:,.0f} children<extra></extra>",
        )
    )
    figure.update_layout(
        **chart_layout(
            f"Transfer–Discharge Flow Balance ({granularity})",
            "Children",
        )
    )
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})


def render_streak_chart(
    daily_metrics: pd.DataFrame,
    threshold_days: int,
) -> None:
    """Render the daily backlog-streak counter and escalation threshold."""
    streak = pd.to_numeric(
        daily_metrics[BACKLOG_STREAK_COLUMN], errors="coerce"
    ).fillna(0)
    figure = go.Figure(
        go.Scatter(
            x=daily_metrics.index,
            y=streak,
            name="Backlog streak",
            mode="lines",
            fill="tozeroy",
            line={"color": BLUE, "width": 2},
            fillcolor="rgba(46, 117, 182, 0.15)",
            hovertemplate="%{y:,.0f} consecutive day(s)<extra></extra>",
        )
    )
    figure.add_hline(
        y=threshold_days,
        line_color=PRESSURE_RED,
        line_dash="dash",
        annotation_text=f"Escalation threshold: {threshold_days} days",
        annotation_position="top left",
    )
    figure.update_layout(
        **chart_layout("Daily Backlog Accumulation Streak", "Consecutive days")
    )
    figure.update_yaxes(rangemode="tozero")
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})


def render_anomaly_log(selected_data: pd.DataFrame) -> None:
    """Display logical-constraint violations for the selected period."""
    with st.expander("Backlog Data Quality & Anomaly Log", expanded=False):
        anomaly_columns = [
            TRANSFER_ANOMALY_COLUMN,
            DISCHARGE_ANOMALY_COLUMN,
        ]
        missing = [column for column in anomaly_columns if column not in selected_data]
        if missing:
            st.warning("Anomaly fields are unavailable: " + ", ".join(missing))
            return

        mask = selected_data[anomaly_columns].fillna(False).any(axis=1)
        flagged = selected_data.loc[mask]
        if flagged.empty:
            st.success("No logical constraint violations were found in this period.")
            return

        st.error(f"{len(flagged):,} anomalous row(s) require review.")
        columns = [
            CBP_COLUMN,
            TRANSFER_COLUMN,
            HHS_COLUMN,
            DISCHARGE_COLUMN,
            *anomaly_columns,
        ]
        st.dataframe(
            flagged[columns].reset_index(),
            width="stretch",
            hide_index=True,
        )


def main() -> None:
    """Build the complete backlog analysis page."""
    apply_page_styles()
    st.markdown(
        """
        <div class="page-header">
            <h1>Backlog Pressure & Flow Balance</h1>
            <p>Monitor sustained intake pressure, transfer–discharge imbalance,
            and operational backlog episodes across the HHS care system.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Backlog Controls")
        uploaded_file = st.file_uploader(
            "Upload HHS capacity data",
            type=["csv"],
            help="Leave empty to use the synthetic 2023-2025 dataset.",
            key="backlog_csv_uploader",
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
        st.error(f"Unable to prepare backlog data: {exc}")
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
            key="backlog_date_range",
        )
        granularity = st.selectbox(
            "Chart granularity",
            ["Daily", "Weekly", "Monthly"],
            key="backlog_granularity",
        )
        threshold_days = st.slider(
            "Elevated backlog threshold",
            min_value=2,
            max_value=30,
            value=3,
            help="Minimum consecutive positive-intake days before escalation shading begins.",
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

    selected_data = cleaned_data.loc[
        pd.Timestamp(start_date):pd.Timestamp(end_date)
    ].copy()
    if selected_data.empty:
        st.warning("No data is available for the selected reporting period.")
        st.stop()

    try:
        daily_metrics = compute_capacity_metrics(selected_data)
        kpis = calculate_kpis(daily_metrics)
        episodes = build_backlog_episodes(daily_metrics)
        chart_data = aggregate_for_chart(daily_metrics, granularity)
    except (DataValidationError, TypeError, ValueError) as exc:
        st.error(f"Unable to calculate backlog analytics: {exc}")
        st.stop()

    current_streak = calculate_current_streak(daily_metrics)
    positive_days = int(daily_metrics[NET_INTAKE_COLUMN].gt(0).sum())
    cumulative_pressure = int(daily_metrics[NET_INTAKE_COLUMN].sum())
    longest_streak = int(kpis["backlog_accumulation_rate"])
    high_risk_episodes = (
        int(episodes["Duration (Days)"].ge(threshold_days).sum())
        if not episodes.empty
        else 0
    )

    st.caption(
        f"Reporting period: {start_date:%d %b %Y}–{end_date:%d %b %Y} "
        f"• {len(selected_data):,} daily observations • {granularity} charts"
    )

    kpi_columns = st.columns(5)
    kpi_columns[0].metric(
        "Active Backlog Streak",
        f"{current_streak:,} day(s)",
        help="Consecutive positive-net-intake days ending on the latest selected date.",
    )
    kpi_columns[1].metric(
        "Longest Pressure Episode",
        f"{longest_streak:,} day(s)",
        help="Longest positive-net-intake streak within the selected period.",
    )
    kpi_columns[2].metric(
        "Positive-Pressure Days",
        f"{positive_days:,}",
        help="Number of selected days when transfers exceeded discharges.",
    )
    kpi_columns[3].metric(
        "Cumulative Net Intake",
        f"{cumulative_pressure:+,}",
        help="Selected-period transfers minus discharges; negative values indicate net relief.",
    )
    kpi_columns[4].metric(
        "Elevated Episodes",
        f"{high_risk_episodes:,}",
        help=f"Pressure episodes lasting at least {threshold_days} consecutive days.",
    )

    if current_streak >= threshold_days:
        st.markdown(
            f"""
            <div class="pressure-callout">
                <strong>Escalation indicator:</strong> The active backlog streak is
                <strong>{current_streak} days</strong>, meeting the selected
                {threshold_days}-day elevated-pressure threshold.
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif current_streak > 0:
        st.markdown(
            f"""
            <div class="pressure-callout">
                <strong>Pressure is building:</strong> Positive net intake has continued
                for <strong>{current_streak} day(s)</strong>; escalation begins at
                {threshold_days} days.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="relief-callout">
                <strong>No active backlog streak.</strong> The latest selected day
                shows zero or negative net intake.
            </div>
            """,
            unsafe_allow_html=True,
        )

    timeline_tab, balance_tab, episodes_tab = st.tabs(
        ["Pressure Timeline", "Flow Balance & Streak", "Backlog Episodes"]
    )
    with timeline_tab:
        render_pressure_timeline(
            chart_data,
            episodes,
            threshold_days,
            granularity,
        )
    with balance_tab:
        render_flow_balance(chart_data, granularity)
        render_streak_chart(daily_metrics, threshold_days)
    with episodes_tab:
        if episodes.empty:
            st.success("No positive-net-intake episodes occurred in this period.")
        else:
            episode_display = episodes.copy()
            episode_display["Episode Start"] = episode_display["Episode Start"].dt.date
            episode_display["Episode End"] = episode_display["Episode End"].dt.date
            st.dataframe(
                episode_display,
                width="stretch",
                hide_index=True,
                column_config={
                    "Average Daily Pressure": st.column_config.NumberColumn(
                        format="%.1f"
                    )
                },
            )
            st.caption(
                "Episodes are ranked by duration, then cumulative net intake. "
                "An episode ends when Net Daily Intake becomes zero or negative."
            )

    render_anomaly_log(selected_data)
    st.caption(
        "Decision-support view. Synthetic data is for demonstration only; "
        "validate operational conclusions against authoritative HHS and CBP data."
    )


if __name__ == "__main__":
    main()
