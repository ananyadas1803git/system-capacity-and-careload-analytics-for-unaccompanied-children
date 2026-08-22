"""Operational overview page for the HHS UAC capacity analytics dashboard.

Run directly with::

    streamlit run app/pages/overview.py
"""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
    INTAKE_COLUMN,
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
LIGHT_BLUE = "#77A9D4"
TEAL = "#167C80"
AMBER = "#D97706"
RED = "#B42318"
GREEN = "#238636"


st.set_page_config(
    page_title="System Overview | HHS UAC Analytics",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_page_styles() -> None:
    """Apply the professional slate/navy visual theme."""
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
                padding: 1.3rem 1.55rem;
            }}
            .page-header h1 {{
                color: {WHITE};
                font-size: 1.9rem;
                margin: 0 0 .35rem 0;
            }}
            .page-header p {{ color: #DCE7F2; margin: 0; }}
            div[data-testid="stMetric"] {{
                background-color: {WHITE};
                border: 1px solid {SLATE_200};
                border-left: 4px solid {BLUE};
                border-radius: 8px;
                min-height: 130px;
                padding: .9rem 1rem;
            }}
            div[data-testid="stMetricLabel"] {{ color: {SLATE_500}; }}
            div[data-testid="stMetricValue"] {{ color: {NAVY}; }}
            .snapshot-card {{
                background-color: {WHITE};
                border: 1px solid {SLATE_200};
                border-left: 5px solid {BLUE};
                border-radius: 7px;
                color: {SLATE_700};
                min-height: 120px;
                padding: .85rem 1rem;
            }}
            .snapshot-card h4 {{ color: {NAVY}; margin: 0 0 .4rem 0; }}
            .snapshot-positive {{ border-left-color: {GREEN}; }}
            .snapshot-warning {{ border-left-color: {AMBER}; }}
            .snapshot-critical {{ border-left-color: {RED}; }}
            .section-heading {{
                border-bottom: 2px solid {NAVY};
                color: {NAVY};
                font-size: 1.12rem;
                font-weight: 700;
                margin: 1.15rem 0 .75rem 0;
                padding-bottom: .35rem;
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
    """Return the cached deterministic 2023-2025 mock dataset."""
    return generate_mock_data()


def aggregate_for_chart(metrics: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """Aggregate stock and flow variables correctly for chart display."""
    if granularity == "Daily":
        return metrics.copy()

    frequency = pd.offsets.Week(weekday=6) if granularity == "Weekly" else pd.offsets.MonthEnd()
    last_columns = [
        CBP_COLUMN,
        HHS_COLUMN,
        TOTAL_LOAD_COLUMN,
        ROLLING_7_COLUMN,
        ROLLING_14_COLUMN,
        GROWTH_RATE_COLUMN,
        BACKLOG_STREAK_COLUMN,
    ]
    sum_columns = [
        INTAKE_COLUMN,
        TRANSFER_COLUMN,
        DISCHARGE_COLUMN,
        NET_INTAKE_COLUMN,
    ]
    aggregations = {
        **{column: "last" for column in last_columns if column in metrics},
        **{column: "sum" for column in sum_columns if column in metrics},
    }
    aggregated = metrics.resample(frequency).agg(aggregations).dropna(how="all")
    aggregated[OFFSET_RATIO_COLUMN] = aggregated[DISCHARGE_COLUMN].div(
        aggregated[TRANSFER_COLUMN] + 1
    )
    return aggregated


def period_load_change(metrics: pd.DataFrame) -> float:
    """Return endpoint percentage change in Total System Load."""
    if len(metrics) < 2:
        return 0.0
    start = float(metrics[TOTAL_LOAD_COLUMN].iloc[0])
    end = float(metrics[TOTAL_LOAD_COLUMN].iloc[-1])
    if start == 0 or not np.isfinite(start) or not np.isfinite(end):
        return 0.0
    return float((end - start) / abs(start) * 100)


def chart_layout(title: str, y_title: str, *, height: int = 445) -> dict:
    """Return a consistent Plotly layout configuration."""
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


def render_kpi_row(
    metrics: pd.DataFrame,
    kpis: dict[str, float | int],
) -> None:
    """Render the five required system-level KPIs."""
    previous_load = (
        int(metrics[TOTAL_LOAD_COLUMN].iloc[-2])
        if len(metrics) > 1
        else int(metrics[TOTAL_LOAD_COLUMN].iloc[-1])
    )
    latest_load = int(kpis["total_children_under_care"])
    columns = st.columns(5)
    columns[0].metric(
        "Total Children Under Care",
        f"{latest_load:,}",
        delta=f"{latest_load - previous_load:+,} day over day",
        delta_color="inverse",
        help="Latest active CBP custody plus active HHS care.",
    )
    columns[1].metric(
        "Net Intake Pressure",
        f"{kpis['net_intake_pressure']:+,.0f}",
        help="Latest transfers out of CBP minus latest HHS discharges.",
    )
    columns[2].metric(
        "Care Load Volatility",
        f"{kpis['care_load_volatility_index']:.2f}%",
        help="Standard deviation of daily care-load growth in the selected period.",
    )
    columns[3].metric(
        "Backlog Accumulation",
        f"{kpis['backlog_accumulation_rate']:,.0f} day(s)",
        help="Longest positive-net-intake streak in the selected period.",
    )
    columns[4].metric(
        "Discharge Offset Ratio",
        f"{kpis['discharge_offset_ratio']:.1%}",
        help="Latest discharges divided by latest transfers plus one.",
    )


def render_snapshot_insights(
    metrics: pd.DataFrame,
    anomaly_count: int,
    imputed_count: int,
) -> None:
    """Render concise load, flow, and data-quality findings."""
    load_change = period_load_change(metrics)
    cumulative_net = int(metrics[NET_INTAKE_COLUMN].sum())
    total_transfers = int(metrics[TRANSFER_COLUMN].sum())
    total_discharges = int(metrics[DISCHARGE_COLUMN].sum())
    flow_ratio = total_discharges / (total_transfers + 1)

    load_class = (
        "snapshot-warning" if load_change > 1 else "snapshot-positive" if load_change < -1 else ""
    )
    load_direction = (
        "increased" if load_change > 0 else "decreased" if load_change < 0 else "held steady"
    )
    flow_class = "snapshot-warning" if cumulative_net > 0 else "snapshot-positive"
    quality_class = "snapshot-critical" if anomaly_count else "snapshot-positive"

    columns = st.columns(3)
    with columns[0]:
        st.markdown(
            f"""
            <div class="snapshot-card {load_class}">
                <h4>System load direction</h4>
                Total System Load {load_direction} by
                <strong>{abs(load_change):.1f}%</strong> between selected-period
                endpoints.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with columns[1]:
        st.markdown(
            f"""
            <div class="snapshot-card {flow_class}">
                <h4>Selected-period flow balance</h4>
                Cumulative net intake was <strong>{cumulative_net:+,}</strong> and
                aggregate discharge coverage was <strong>{flow_ratio:.1%}</strong>.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with columns[2]:
        st.markdown(
            f"""
            <div class="snapshot-card {quality_class}">
                <h4>Data-quality snapshot</h4>
                <strong>{anomaly_count:,}</strong> logical anomaly row(s) and
                <strong>{imputed_count:,}</strong> imputed calendar date(s) occur
                within this selection.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_system_load_chart(chart_data: pd.DataFrame, granularity: str) -> None:
    """Render Total System Load with 7-day and 14-day smoothing."""
    figure = go.Figure()
    for column, label, color, dash, width in (
        (TOTAL_LOAD_COLUMN, "Total System Load", NAVY, "solid", 2.4),
        (ROLLING_7_COLUMN, "7-day moving average", BLUE, "dot", 2),
        (ROLLING_14_COLUMN, "14-day moving average", TEAL, "dash", 2),
    ):
        figure.add_trace(
            go.Scatter(
                x=chart_data.index,
                y=chart_data[column],
                name=label,
                mode="lines",
                line={"color": color, "dash": dash, "width": width},
                hovertemplate="%{y:,.1f} children<extra></extra>",
            )
        )
    figure.update_layout(
        **chart_layout(f"System Load Overview ({granularity})", "Children under care")
    )
    figure.update_yaxes(rangemode="tozero")
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})


def render_load_composition_chart(chart_data: pd.DataFrame, granularity: str) -> None:
    """Render a stacked-area comparison of CBP and HHS active loads."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data[CBP_COLUMN],
            name="CBP active custody",
            mode="lines",
            line={"color": BLUE, "width": 1.6},
            stackgroup="care-load",
            hovertemplate="%{y:,.0f} children<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data[HHS_COLUMN],
            name="HHS active care",
            mode="lines",
            line={"color": TEAL, "width": 1.6},
            stackgroup="care-load",
            hovertemplate="%{y:,.0f} children<extra></extra>",
        )
    )
    figure.update_layout(**chart_layout(f"CBP vs HHS Care Load ({granularity})", "Children"))
    figure.update_yaxes(rangemode="tozero")
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})


def render_flow_balance_chart(chart_data: pd.DataFrame, granularity: str) -> None:
    """Render transfers, discharges, and net intake."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data[TRANSFER_COLUMN],
            name="Transfers",
            mode="lines",
            line={"color": AMBER, "width": 2},
            hovertemplate="%{y:,.0f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data[DISCHARGE_COLUMN],
            name="Discharges",
            mode="lines",
            line={"color": TEAL, "width": 2},
            hovertemplate="%{y:,.0f}<extra></extra>",
        )
    )
    net_intake = pd.to_numeric(chart_data[NET_INTAKE_COLUMN], errors="coerce").fillna(0)
    figure.add_trace(
        go.Bar(
            x=chart_data.index,
            y=net_intake,
            name="Net intake",
            marker_color=np.where(net_intake.gt(0), RED, GREEN),
            opacity=0.4,
            hovertemplate="%{y:+,.0f}<extra></extra>",
        )
    )
    figure.add_hline(y=0, line_color=SLATE_500, line_width=1)
    figure.update_layout(
        **chart_layout(f"Transfer–Discharge Flow Balance ({granularity})", "Children")
    )
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})


def render_latest_composition(metrics: pd.DataFrame) -> None:
    """Render the latest CBP/HHS proportional composition."""
    latest = metrics.iloc[-1]
    figure = go.Figure(
        go.Pie(
            labels=["CBP custody", "HHS care"],
            values=[latest[CBP_COLUMN], latest[HHS_COLUMN]],
            hole=0.62,
            marker={"colors": [BLUE, TEAL]},
            textinfo="label+percent",
            hovertemplate="%{label}: %{value:,.0f} children<extra></extra>",
        )
    )
    figure.update_layout(
        title={"text": "Latest Care-Load Composition", "font": {"color": NAVY}},
        height=445,
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        margin={"l": 25, "r": 25, "t": 70, "b": 30},
        showlegend=False,
        annotations=[
            {
                "text": f"{int(latest[TOTAL_LOAD_COLUMN]):,}<br>Total",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"size": 17, "color": NAVY},
            }
        ],
    )
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})


def render_data_quality_panel(
    cleaned_data: pd.DataFrame,
    selected_data: pd.DataFrame,
) -> None:
    """Display validation findings and selected logical anomaly rows."""
    with st.expander("Overview Data Quality & Validation Log", expanded=False):
        report = cleaned_data.attrs.get("validation_report")
        if report is not None and hasattr(report, "to_frame"):
            st.markdown("**Dataset validation findings**")
            st.dataframe(report.to_frame(), width="stretch", hide_index=True)

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
            st.success("No selected-period logical constraint violations were found.")
            return

        st.error(f"{len(flagged):,} selected row(s) require review.")
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
    """Build the complete system overview page."""
    apply_page_styles()
    st.markdown(
        """
        <div class="page-header">
            <h1>System Capacity & Care Load Overview</h1>
            <p>Executive snapshot of active CBP and HHS care loads, intake and
            discharge flows, backlog pressure, and source-data quality.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Overview Controls")
        uploaded_file = st.file_uploader(
            "Upload HHS capacity data",
            type=["csv"],
            help="Leave empty to use the synthetic 2023-2025 dataset.",
            key="overview_csv_uploader",
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
        st.error(f"Unable to prepare overview data: {exc}")
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
            key="overview_date_range",
        )
        granularity = st.selectbox(
            "Chart granularity",
            ["Daily", "Weekly", "Monthly"],
            key="overview_granularity",
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
        kpis = calculate_kpis(daily_metrics)
        chart_data = aggregate_for_chart(daily_metrics, granularity)
    except (DataValidationError, TypeError, ValueError) as exc:
        st.error(f"Unable to calculate overview analytics: {exc}")
        st.stop()

    anomaly_count = int(
        selected_data[[TRANSFER_ANOMALY_COLUMN, DISCHARGE_ANOMALY_COLUMN]]
        .fillna(False)
        .any(axis=1)
        .sum()
    )
    imputed_column = next(
        (name for name in ("Is_Imputed_Date", "Is Imputed Date") if name in selected_data.columns),
        None,
    )
    imputed_count = int(selected_data[imputed_column].fillna(False).sum()) if imputed_column else 0

    st.caption(
        f"Reporting period: {start_date:%d %b %Y}–{end_date:%d %b %Y} "
        f"• {len(selected_data):,} daily observations • {granularity} charts"
    )
    render_kpi_row(daily_metrics, kpis)

    st.markdown(
        '<div class="section-heading">Operational Snapshot</div>',
        unsafe_allow_html=True,
    )
    render_snapshot_insights(daily_metrics, anomaly_count, imputed_count)

    load_tab, composition_tab, flow_tab = st.tabs(
        ["System Load", "Care-Load Composition", "Flow Balance"]
    )
    with load_tab:
        render_system_load_chart(chart_data, granularity)
    with composition_tab:
        left_column, right_column = st.columns([2, 1])
        with left_column:
            render_load_composition_chart(chart_data, granularity)
        with right_column:
            render_latest_composition(daily_metrics)
    with flow_tab:
        render_flow_balance_chart(chart_data, granularity)

    with st.expander("Dashboard Module Guide", expanded=False):
        st.markdown(
            """
            - **KPI Scorecard:** detailed KPI comparisons, trends, and methodology.
            - **Capacity Planning:** configurable planning ceilings, utilization, and headroom.
            - **Backlog Pressure:** sustained positive-net-intake episodes and escalation signals.
            - **Executive Insights:** automated findings, correlations, and risk screening.
            - **Trends:** longer-term daily, weekly, and monthly movement analysis.
            """
        )

    render_data_quality_panel(cleaned_data, selected_data)
    st.download_button(
        "Download selected analytical data",
        data=daily_metrics.reset_index().to_csv(index=False).encode("utf-8"),
        file_name=f"uac_overview_{start_date}_{end_date}.csv",
        mime="text/csv",
    )
    st.caption(
        "Decision-support overview. Synthetic data is for demonstration only; "
        "validate findings against authoritative HHS and CBP source publications."
    )


if __name__ == "__main__":
    main()
