"""Capacity planning page for the HHS UAC analytics dashboard.

Run directly with::

    streamlit run app/pages/capacity.py
"""

from __future__ import annotations

import math
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
    CBP_COLUMN,
    DISCHARGE_ANOMALY_COLUMN,
    DISCHARGE_COLUMN,
    HHS_COLUMN,
    NET_INTAKE_COLUMN,
    ROLLING_14_COLUMN,
    ROLLING_7_COLUMN,
    TOTAL_LOAD_COLUMN,
    TRANSFER_ANOMALY_COLUMN,
    TRANSFER_COLUMN,
    DataValidationError,
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

CBP_UTILIZATION_COLUMN = "CBP Capacity Utilization"
HHS_UTILIZATION_COLUMN = "HHS Capacity Utilization"
SYSTEM_UTILIZATION_COLUMN = "Total Capacity Utilization"
CBP_HEADROOM_COLUMN = "CBP Capacity Headroom"
HHS_HEADROOM_COLUMN = "HHS Capacity Headroom"
SYSTEM_HEADROOM_COLUMN = "Total Capacity Headroom"


st.set_page_config(
    page_title="Capacity Planning | HHS UAC Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_page_styles() -> None:
    """Apply the slate/navy government-analytics visual theme."""
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
                border-left: 4px solid {BLUE};
                border-radius: 8px;
                min-height: 126px;
                padding: .9rem 1rem;
            }}
            div[data-testid="stMetricLabel"] {{ color: {SLATE_500}; }}
            div[data-testid="stMetricValue"] {{ color: {NAVY}; }}
            .assumption-note {{
                background-color: #EFF6FF;
                border: 1px solid #B9D5F0;
                border-left: 5px solid {BLUE};
                border-radius: 7px;
                color: #173A5E;
                margin: .5rem 0 1rem 0;
                padding: .8rem 1rem;
            }}
            .capacity-alert {{
                background-color: #FFF1F0;
                border: 1px solid #F2B8B5;
                border-left: 5px solid {RED};
                border-radius: 7px;
                color: #7A271A;
                margin: .5rem 0 1rem 0;
                padding: .8rem 1rem;
            }}
            .capacity-stable {{
                background-color: #ECFDF3;
                border: 1px solid #ABEFC6;
                border-left: 5px solid {GREEN};
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
    """Read an uploaded CSV using string-first parsing."""
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
    """Return the cached synthetic 2023-2025 capacity dataset."""
    return generate_mock_data()


def suggested_capacity(series: pd.Series) -> int:
    """Return a rounded planning ceiling above the observed peak load."""
    peak = float(pd.to_numeric(series, errors="coerce").fillna(0).max())
    if peak <= 0:
        return 100
    order = 10 ** max(1, int(math.floor(math.log10(peak))) - 1)
    return int(math.ceil((peak * 1.15) / order) * order)


def add_capacity_metrics(
    metrics: pd.DataFrame,
    cbp_capacity: int,
    hhs_capacity: int,
) -> pd.DataFrame:
    """Add utilization and remaining-headroom fields to daily analytics."""
    if cbp_capacity <= 0 or hhs_capacity <= 0:
        raise ValueError("CBP and HHS planning capacities must be positive.")

    frame = metrics.copy()
    total_capacity = cbp_capacity + hhs_capacity
    frame[CBP_UTILIZATION_COLUMN] = frame[CBP_COLUMN].div(cbp_capacity).mul(100)
    frame[HHS_UTILIZATION_COLUMN] = frame[HHS_COLUMN].div(hhs_capacity).mul(100)
    frame[SYSTEM_UTILIZATION_COLUMN] = frame[TOTAL_LOAD_COLUMN].div(total_capacity).mul(100)
    frame[CBP_HEADROOM_COLUMN] = cbp_capacity - frame[CBP_COLUMN]
    frame[HHS_HEADROOM_COLUMN] = hhs_capacity - frame[HHS_COLUMN]
    frame[SYSTEM_HEADROOM_COLUMN] = total_capacity - frame[TOTAL_LOAD_COLUMN]
    return frame.replace([np.inf, -np.inf], np.nan)


def aggregate_for_chart(metrics: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """Aggregate point-in-time capacity metrics for chart display."""
    if granularity == "Daily":
        return metrics.copy()

    frequency = pd.offsets.Week(weekday=6) if granularity == "Weekly" else pd.offsets.MonthEnd()
    stock_columns = [
        CBP_COLUMN,
        HHS_COLUMN,
        TOTAL_LOAD_COLUMN,
        ROLLING_7_COLUMN,
        ROLLING_14_COLUMN,
        CBP_UTILIZATION_COLUMN,
        HHS_UTILIZATION_COLUMN,
        SYSTEM_UTILIZATION_COLUMN,
        CBP_HEADROOM_COLUMN,
        HHS_HEADROOM_COLUMN,
        SYSTEM_HEADROOM_COLUMN,
    ]
    flow_columns = [TRANSFER_COLUMN, DISCHARGE_COLUMN, NET_INTAKE_COLUMN]
    aggregations = {
        **{column: "last" for column in stock_columns if column in metrics},
        **{column: "sum" for column in flow_columns if column in metrics},
    }
    return metrics.resample(frequency).agg(aggregations).dropna(how="all")


def build_stress_episodes(
    metrics: pd.DataFrame,
    warning_threshold: int,
) -> pd.DataFrame:
    """Summarize continuous periods above the utilization warning threshold."""
    if metrics.empty:
        return pd.DataFrame()

    working = metrics[[SYSTEM_UTILIZATION_COLUMN, SYSTEM_HEADROOM_COLUMN, TOTAL_LOAD_COLUMN]].copy()
    working["Above Threshold"] = working[SYSTEM_UTILIZATION_COLUMN].ge(warning_threshold)
    working["Episode Group"] = (
        working["Above Threshold"].ne(working["Above Threshold"].shift(fill_value=False)).cumsum()
    )
    stressed = working.loc[working["Above Threshold"]]
    if stressed.empty:
        return pd.DataFrame(
            columns=[
                "Episode Start",
                "Episode End",
                "Duration (Days)",
                "Peak Utilization (%)",
                "Minimum Headroom",
                "Peak System Load",
                "Status",
            ]
        )

    latest_date = working.index.max()
    records: list[dict[str, object]] = []
    for _, episode in stressed.groupby("Episode Group"):
        start = episode.index.min()
        end = episode.index.max()
        records.append(
            {
                "Episode Start": start,
                "Episode End": end,
                "Duration (Days)": int(len(episode)),
                "Peak Utilization (%)": float(episode[SYSTEM_UTILIZATION_COLUMN].max()),
                "Minimum Headroom": int(episode[SYSTEM_HEADROOM_COLUMN].min()),
                "Peak System Load": int(episode[TOTAL_LOAD_COLUMN].max()),
                "Status": "Active" if end == latest_date else "Closed",
            }
        )
    return pd.DataFrame.from_records(records).sort_values(
        ["Peak Utilization (%)", "Duration (Days)"],
        ascending=[False, False],
    )


def chart_layout(title: str, y_title: str, *, height: int = 460) -> dict:
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


def render_load_capacity_chart(
    chart_data: pd.DataFrame,
    cbp_capacity: int,
    hhs_capacity: int,
    granularity: str,
) -> None:
    """Render stacked active load against the combined planning ceiling."""
    total_capacity = cbp_capacity + hhs_capacity
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data[CBP_COLUMN],
            name="CBP active load",
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
            name="HHS active load",
            mode="lines",
            line={"color": TEAL, "width": 1.6},
            stackgroup="care-load",
            hovertemplate="%{y:,.0f} children<extra></extra>",
        )
    )
    figure.add_hline(
        y=total_capacity,
        line_color=RED,
        line_dash="dash",
        annotation_text=f"Combined planning capacity: {total_capacity:,}",
        annotation_position="top left",
    )
    figure.update_layout(
        **chart_layout(
            f"Active Load vs Planning Capacity ({granularity})",
            "Children",
        )
    )
    figure.update_yaxes(rangemode="tozero")
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})


def render_utilization_chart(
    chart_data: pd.DataFrame,
    warning_threshold: int,
    critical_threshold: int,
    granularity: str,
) -> None:
    """Render CBP, HHS, and combined utilization percentages."""
    figure = go.Figure()
    for column, label, color in (
        (CBP_UTILIZATION_COLUMN, "CBP utilization", BLUE),
        (HHS_UTILIZATION_COLUMN, "HHS utilization", TEAL),
        (SYSTEM_UTILIZATION_COLUMN, "Combined utilization", NAVY),
    ):
        figure.add_trace(
            go.Scatter(
                x=chart_data.index,
                y=chart_data[column],
                name=label,
                mode="lines",
                line={"color": color, "width": 2},
                hovertemplate="%{y:.1f}%<extra></extra>",
            )
        )
    figure.add_hline(
        y=warning_threshold,
        line_color=AMBER,
        line_dash="dot",
        annotation_text=f"Warning: {warning_threshold}%",
        annotation_position="bottom left",
    )
    figure.add_hline(
        y=critical_threshold,
        line_color=RED,
        line_dash="dash",
        annotation_text=f"Critical: {critical_threshold}%",
        annotation_position="top left",
    )
    figure.update_layout(**chart_layout(f"Capacity Utilization ({granularity})", "Utilization (%)"))
    figure.update_yaxes(rangemode="tozero", ticksuffix="%")
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})


def render_headroom_chart(chart_data: pd.DataFrame, granularity: str) -> None:
    """Render combined remaining capacity, including over-capacity periods."""
    headroom = pd.to_numeric(chart_data[SYSTEM_HEADROOM_COLUMN], errors="coerce").fillna(0)
    colors = np.where(headroom.lt(0), RED, GREEN)
    figure = go.Figure(
        go.Bar(
            x=chart_data.index,
            y=headroom,
            name="Combined headroom",
            marker_color=colors,
            hovertemplate="%{y:+,.0f} places<extra></extra>",
        )
    )
    figure.add_hline(y=0, line_color=SLATE_500, line_width=1)
    figure.update_layout(
        **chart_layout(f"Remaining System Headroom ({granularity})", "Available places")
    )
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})


def render_anomaly_log(selected_data: pd.DataFrame) -> None:
    """Display logical source-data violations within the selected period."""
    with st.expander("Capacity Data Quality & Anomaly Log", expanded=False):
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
    """Build the complete capacity planning page."""
    apply_page_styles()
    st.markdown(
        """
        <div class="page-header">
            <h1>System Capacity Planning & Utilization</h1>
            <p>Compare active CBP and HHS care loads with configurable planning
            ceilings, available headroom, and sustained capacity stress.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Capacity Controls")
        uploaded_file = st.file_uploader(
            "Upload HHS capacity data",
            type=["csv"],
            help="Leave empty to use the synthetic 2023-2025 dataset.",
            key="capacity_csv_uploader",
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
        st.error(f"Unable to prepare capacity data: {exc}")
        st.stop()
    except Exception as exc:
        st.exception(exc)
        st.stop()

    minimum_date = cleaned_data.index.min().date()
    maximum_date = cleaned_data.index.max().date()
    default_cbp_capacity = suggested_capacity(cleaned_data[CBP_COLUMN])
    default_hhs_capacity = suggested_capacity(cleaned_data[HHS_COLUMN])

    with st.sidebar:
        selected_dates = st.date_input(
            "Reporting period",
            value=(minimum_date, maximum_date),
            min_value=minimum_date,
            max_value=maximum_date,
            key="capacity_date_range",
        )
        granularity = st.selectbox(
            "Chart granularity",
            ["Daily", "Weekly", "Monthly"],
            key="capacity_granularity",
        )
        st.subheader("Planning ceilings")
        cbp_capacity = int(
            st.number_input(
                "CBP planning capacity",
                min_value=1,
                value=default_cbp_capacity,
                step=max(10, default_cbp_capacity // 20),
                help="User-defined planning limit; not an official capacity figure.",
            )
        )
        hhs_capacity = int(
            st.number_input(
                "HHS planning capacity",
                min_value=1,
                value=default_hhs_capacity,
                step=max(10, default_hhs_capacity // 20),
                help="User-defined planning limit; not an official capacity figure.",
            )
        )
        warning_threshold = st.slider(
            "Warning threshold (%)",
            min_value=50,
            max_value=95,
            value=80,
        )
        critical_threshold = st.slider(
            "Critical threshold (%)",
            min_value=warning_threshold + 1,
            max_value=120,
            value=max(95, warning_threshold + 1),
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
        capacity_metrics = add_capacity_metrics(
            daily_metrics,
            cbp_capacity,
            hhs_capacity,
        )
        chart_data = aggregate_for_chart(capacity_metrics, granularity)
        stress_episodes = build_stress_episodes(
            capacity_metrics,
            warning_threshold,
        )
    except (DataValidationError, TypeError, ValueError) as exc:
        st.error(f"Unable to calculate capacity analytics: {exc}")
        st.stop()

    latest = capacity_metrics.iloc[-1]
    latest_utilization = float(latest[SYSTEM_UTILIZATION_COLUMN])
    peak_utilization = float(capacity_metrics[SYSTEM_UTILIZATION_COLUMN].max())
    warning_days = int(capacity_metrics[SYSTEM_UTILIZATION_COLUMN].ge(warning_threshold).sum())
    critical_days = int(capacity_metrics[SYSTEM_UTILIZATION_COLUMN].ge(critical_threshold).sum())

    st.caption(
        f"Reporting period: {start_date:%d %b %Y}–{end_date:%d %b %Y} "
        f"• {len(selected_data):,} daily observations • {granularity} charts"
    )
    st.markdown(
        """
        <div class="assumption-note">
            <strong>Planning assumption:</strong> Capacity ceilings on this page
            are user-configurable scenario inputs. They are not official HHS or
            CBP capacity figures and do not alter the underlying source data.
        </div>
        """,
        unsafe_allow_html=True,
    )

    kpi_columns = st.columns(5)
    kpi_columns[0].metric(
        "Latest System Load",
        f"{int(latest[TOTAL_LOAD_COLUMN]):,}",
        help="Latest CBP active custody plus HHS active care load.",
    )
    kpi_columns[1].metric(
        "Current Utilization",
        f"{latest_utilization:.1f}%",
        help="Latest system load divided by combined planning capacity.",
    )
    kpi_columns[2].metric(
        "Available Headroom",
        f"{int(latest[SYSTEM_HEADROOM_COLUMN]):+,}",
        help="Combined planning capacity minus latest Total System Load.",
    )
    kpi_columns[3].metric(
        "Peak Utilization",
        f"{peak_utilization:.1f}%",
        help="Highest combined utilization during the selected period.",
    )
    kpi_columns[4].metric(
        "Threshold Breach Days",
        f"{warning_days:,}",
        delta=f"{critical_days} critical",
        delta_color="inverse",
        help="Days at or above the warning threshold; delta shows critical days.",
    )

    if latest_utilization >= critical_threshold:
        st.markdown(
            f"""
            <div class="capacity-alert">
                <strong>Critical capacity pressure:</strong> Current utilization is
                {latest_utilization:.1f}%, above the {critical_threshold}% critical threshold.
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif latest_utilization >= warning_threshold:
        st.markdown(
            f"""
            <div class="capacity-alert">
                <strong>Capacity warning:</strong> Current utilization is
                {latest_utilization:.1f}%, above the {warning_threshold}% warning threshold.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="capacity-stable">
                <strong>Capacity within planning range:</strong> Current utilization is
                {latest_utilization:.1f}%, below the {warning_threshold}% warning threshold.
            </div>
            """,
            unsafe_allow_html=True,
        )

    load_tab, utilization_tab, stress_tab = st.tabs(
        ["Load & Capacity", "Utilization & Headroom", "Capacity Stress Episodes"]
    )
    with load_tab:
        render_load_capacity_chart(
            chart_data,
            cbp_capacity,
            hhs_capacity,
            granularity,
        )
    with utilization_tab:
        render_utilization_chart(
            chart_data,
            warning_threshold,
            critical_threshold,
            granularity,
        )
        render_headroom_chart(chart_data, granularity)
    with stress_tab:
        if stress_episodes.empty:
            st.success("No sustained capacity-stress episodes reached the warning threshold.")
        else:
            episode_display = stress_episodes.copy()
            episode_display["Episode Start"] = episode_display["Episode Start"].dt.date
            episode_display["Episode End"] = episode_display["Episode End"].dt.date
            st.dataframe(
                episode_display,
                width="stretch",
                hide_index=True,
                column_config={
                    "Peak Utilization (%)": st.column_config.NumberColumn(format="%.1f%%")
                },
            )
            st.caption(
                "Episodes are continuous periods at or above the selected warning threshold."
            )

    render_anomaly_log(selected_data)
    st.caption(
        "Scenario-planning view. Capacity values are user assumptions; validate "
        "operational decisions against authoritative HHS and CBP capacity information."
    )


if __name__ == "__main__":
    main()
