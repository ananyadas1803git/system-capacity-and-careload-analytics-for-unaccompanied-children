"""Streamlit interface for HHS UAC System Capacity & Care Load Analytics.

Run the application with: ``streamlit run app.py``.
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_utils import *  # noqa: F403


# Slate/navy government-analytics palette.
SLATE_950 = "#172033"
SLATE_800 = "#27364A"
SLATE_600 = "#526477"
SLATE_200 = "#D9E1EA"
SLATE_100 = "#EDF2F7"
NAVY = "#163B65"
BLUE = "#2E75B6"
TEAL = "#167C80"
PRESSURE_RED = "#B42318"
RELIEF_GREEN = "#238636"
WHITE = "#FFFFFF"


st.set_page_config(
    page_title="HHS UAC Capacity Analytics",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_theme() -> None:
    """Apply a restrained slate/navy visual theme to the Streamlit page."""
    st.markdown(
        f"""
        <style>
            .stApp {{ background-color: #F5F7FA; color: {SLATE_950}; }}
            [data-testid="stSidebar"] {{
                background-color: {SLATE_100};
                border-right: 1px solid {SLATE_200};
            }}
            .government-header {{
                background: linear-gradient(110deg, {SLATE_950}, {NAVY});
                border-radius: 10px;
                color: {WHITE};
                margin-bottom: 1rem;
                padding: 1.35rem 1.6rem;
            }}
            .government-header h1 {{
                color: {WHITE}; font-size: 1.9rem; line-height: 1.2;
                margin: 0 0 .35rem 0;
            }}
            .government-header p {{ color: #DCE7F2; margin: 0; }}
            div[data-testid="stMetric"] {{
                background-color: {WHITE};
                border: 1px solid {SLATE_200};
                border-left: 4px solid {BLUE};
                border-radius: 8px;
                min-height: 130px;
                padding: .9rem 1rem;
            }}
            div[data-testid="stMetricLabel"] {{ color: {SLATE_600}; }}
            div[data-testid="stMetricValue"] {{ color: {NAVY}; }}
            .section-heading {{
                border-bottom: 2px solid {NAVY}; color: {NAVY};
                font-size: 1.12rem; font-weight: 700;
                margin: 1.2rem 0 .75rem 0; padding-bottom: .35rem;
            }}
            .backlog-pressure {{
                background-color: #FFF1F0; border: 1px solid #F2B8B5;
                border-left: 5px solid {PRESSURE_RED}; border-radius: 7px;
                color: #7A271A; margin-top: .65rem; padding: .85rem 1rem;
            }}
            .backlog-relief {{
                background-color: #ECFDF3; border: 1px solid #ABEFC6;
                border-left: 5px solid {RELIEF_GREEN}; border-radius: 7px;
                color: #05603A; margin-top: .65rem; padding: .85rem 1rem;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def read_uploaded_csv(file_bytes: bytes) -> pd.DataFrame:
    """Parse uploaded CSV bytes without mutating the upload object."""
    try:
        return pd.read_csv(BytesIO(file_bytes), dtype=str, encoding="utf-8-sig")
    except (
        UnicodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as exc:
        raise DataValidationError(  # noqa: F405
            f"The uploaded CSV could not be read: {exc}"
        ) from exc


@st.cache_data(show_spinner=False)
def get_mock_data() -> pd.DataFrame:
    """Cache the deterministic 2023-2025 mock dataset across reruns."""
    return generate_mock_data()  # noqa: F405


def aggregate_chart_data(daily_metrics: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """Resample charts while preserving stock-versus-flow semantics."""
    if granularity == "Daily":
        return daily_metrics.copy()

    frequency = pd.offsets.Week(weekday=6) if granularity == "Weekly" else pd.offsets.MonthEnd()
    aggregations = {
        INTAKE_COLUMN: "sum",  # noqa: F405
        TRANSFER_COLUMN: "sum",  # noqa: F405
        DISCHARGE_COLUMN: "sum",  # noqa: F405
        CBP_COLUMN: "last",  # noqa: F405
        HHS_COLUMN: "last",  # noqa: F405
        TOTAL_LOAD_COLUMN: "last",  # noqa: F405
        NET_INTAKE_COLUMN: "sum",  # noqa: F405
        ROLLING_7_COLUMN: "last",  # noqa: F405
        ROLLING_14_COLUMN: "last",  # noqa: F405
    }
    available = {
        column: operation
        for column, operation in aggregations.items()
        if column in daily_metrics.columns
    }
    return daily_metrics.sort_index().resample(frequency).agg(available).dropna(how="all")


def calculate_active_backlog_streak(daily_metrics: pd.DataFrame) -> int:
    """Count the current consecutive run of positive net-intake days."""
    if daily_metrics.empty or NET_INTAKE_COLUMN not in daily_metrics:  # noqa: F405
        return 0

    positive_days = (
        pd.to_numeric(
            daily_metrics[NET_INTAKE_COLUMN],  # noqa: F405
            errors="coerce",
        )
        .fillna(0)
        .gt(0)
        .tolist()
    )
    streak = 0
    for has_positive_pressure in reversed(positive_days):
        if not has_positive_pressure:
            break
        streak += 1
    return streak


def base_chart_layout(title: str, y_axis_title: str) -> dict:
    """Return consistent Plotly layout settings."""
    return {
        "title": {"text": title, "font": {"size": 18, "color": NAVY}},
        "height": 470,
        "paper_bgcolor": WHITE,
        "plot_bgcolor": WHITE,
        "font": {"family": "Arial, sans-serif", "color": SLATE_800},
        "hovermode": "x unified",
        "margin": {"l": 60, "r": 25, "t": 70, "b": 50},
        "legend": {"orientation": "h", "x": 0, "y": 1.09},
        "xaxis": {
            "title": "Reporting date",
            "showgrid": False,
            "rangeslider": {"visible": False},
        },
        "yaxis": {
            "title": y_axis_title,
            "gridcolor": SLATE_200,
            "rangemode": "tozero",
            "separatethousands": True,
        },
    }


def render_kpi_summary(kpis: dict[str, int | float]) -> None:
    """Render the requested five-card KPI summary row."""
    columns = st.columns(5)
    columns[0].metric(
        "Total Children Under Care",
        f"{kpis['total_children_under_care']:,.0f}",
        help="Latest active CBP custody plus latest active HHS care.",
    )
    columns[1].metric(
        "Net Intake Pressure",
        f"{kpis['net_intake_pressure']:+,.0f}",
        help="Latest transfers out of CBP minus latest HHS discharges.",
    )
    columns[2].metric(
        "Care Load Volatility",
        f"{kpis['care_load_volatility_index']:.2f}%",
        help="Standard deviation of daily Total System Load growth.",
    )
    columns[3].metric(
        "Backlog Accumulation",
        f"{kpis['backlog_accumulation_rate']:,.0f} day(s)",
        help="Longest positive Net Daily Intake streak in the selected period.",
    )
    columns[4].metric(
        "Discharge Offset Ratio",
        f"{kpis['discharge_offset_ratio']:.1%}",
        help="Latest HHS discharges divided by latest transfers plus one.",
    )


def render_system_load_chart(chart_data: pd.DataFrame) -> None:
    """Render Total System Load and both moving averages."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data[TOTAL_LOAD_COLUMN],  # noqa: F405
            name="Total System Load",
            mode="lines",
            line={"color": NAVY, "width": 2.4},
            hovertemplate="%{y:,.0f} children<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data[ROLLING_7_COLUMN],  # noqa: F405
            name="7-day moving average",
            mode="lines",
            line={"color": BLUE, "width": 2, "dash": "dot"},
            hovertemplate="%{y:,.1f} children<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data[ROLLING_14_COLUMN],  # noqa: F405
            name="14-day moving average",
            mode="lines",
            line={"color": TEAL, "width": 2, "dash": "dash"},
            hovertemplate="%{y:,.1f} children<extra></extra>",
        )
    )
    figure.update_layout(**base_chart_layout("System Load Overview", "Children under care"))
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})


def render_cbp_hhs_chart(chart_data: pd.DataFrame) -> None:
    """Render a stacked-area comparison of active CBP and HHS loads."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data[CBP_COLUMN],  # noqa: F405
            name="Children in CBP custody",
            mode="lines",
            line={"color": BLUE, "width": 1.8},
            stackgroup="active-care",
            hovertemplate="%{y:,.0f} children<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data[HHS_COLUMN],  # noqa: F405
            name="Children in HHS Care",
            mode="lines",
            line={"color": TEAL, "width": 1.8},
            stackgroup="active-care",
            hovertemplate="%{y:,.0f} children<extra></extra>",
        )
    )
    figure.update_layout(**base_chart_layout("CBP vs HHS Active Care Load", "Children"))
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})


def render_net_intake_chart(
    chart_data: pd.DataFrame,
    active_backlog_streak: int,
    granularity: str,
) -> None:
    """Render net-intake bars and active-backlog status."""
    net_intake = pd.to_numeric(
        chart_data[NET_INTAKE_COLUMN],  # noqa: F405
        errors="coerce",
    ).fillna(0)
    colors = np.where(net_intake.gt(0), PRESSURE_RED, RELIEF_GREEN)
    figure = go.Figure(
        go.Bar(
            x=chart_data.index,
            y=net_intake,
            name="Net Daily Intake",
            marker_color=colors,
            hovertemplate="%{y:+,.0f} children<extra></extra>",
        )
    )
    figure.add_hline(y=0, line_color=SLATE_600, line_width=1)
    figure.update_layout(
        **base_chart_layout(
            f"Net Intake & Backlog Pressure ({granularity})",
            "Transfers minus discharges",
        )
    )
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})

    if active_backlog_streak > 0:
        st.markdown(
            f"""
            <div class="backlog-pressure">
                <strong>Active backlog pressure:</strong> Net intake has
                remained positive for <strong>{active_backlog_streak:,}
                consecutive day(s)</strong> through the latest selected date.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="backlog-relief">
                <strong>No active backlog accumulation streak.</strong>
                The latest selected day has zero or negative net intake.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_validation_logs(cleaned_data: pd.DataFrame, selected_data: pd.DataFrame) -> None:
    """Display validation messages and logical-anomaly rows."""
    with st.expander("Data Quality & Validation Logs", expanded=False):
        messages = cleaned_data.attrs.get("validation_messages", [])
        if messages:
            st.markdown("**Cleaning and validation actions**")
            for message in messages:
                st.write(f"• {message}")

        anomaly_columns = [
            TRANSFER_ANOMALY_COLUMN,  # noqa: F405
            DISCHARGE_ANOMALY_COLUMN,  # noqa: F405
        ]
        available = [column for column in anomaly_columns if column in selected_data.columns]
        if not available:
            st.warning("Logical anomaly columns are unavailable.")
            return

        anomaly_mask = selected_data[available].fillna(False).any(axis=1)
        flagged_rows = selected_data.loc[anomaly_mask].copy()
        if flagged_rows.empty:
            st.success(
                "No transfer or discharge constraint violations were found "
                "within the selected reporting period."
            )
            return

        st.error(
            f"{len(flagged_rows):,} flagged row(s) were found within the selected reporting period."
        )
        display_columns = [
            CBP_COLUMN,  # noqa: F405
            TRANSFER_COLUMN,  # noqa: F405
            HHS_COLUMN,  # noqa: F405
            DISCHARGE_COLUMN,  # noqa: F405
            *available,
        ]
        for imputed_column in ("Is_Imputed_Date", "Is Imputed Date"):
            if imputed_column in flagged_rows.columns:
                display_columns.append(imputed_column)
                break
        st.dataframe(
            flagged_rows[display_columns].reset_index(),
            width="stretch",
            hide_index=True,
        )


def main() -> None:
    """Load data, apply controls, calculate metrics, and render the UI."""
    apply_theme()
    st.markdown(
        """
        <div class="government-header">
            <h1>HHS UAC System Capacity &amp; Care Load Analytics</h1>
            <p>Operational monitoring of care loads, intake pressure,
            discharge capacity, backlog trends, and data-quality exceptions</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Dashboard Controls")
        uploaded_file = st.file_uploader(
            "Upload HHS capacity data",
            type=["csv"],
            help="Upload a CSV containing the six required HHS UAC fields.",
        )

    try:
        if uploaded_file is None:
            raw_data = get_mock_data()
            source_name = "Synthetic 2023-2025 demonstration data"
            with st.sidebar:
                st.info(
                    "No CSV was uploaded. The dashboard is using the "
                    "automatically generated 2023-2025 mock dataset."
                )
        else:
            raw_data = read_uploaded_csv(uploaded_file.getvalue())
            source_name = uploaded_file.name

        cleaned_data = validate_and_clean_data(raw_data)  # noqa: F405
    except (DataValidationError, TypeError, ValueError) as exc:  # noqa: F405
        st.error(f"Unable to prepare the dataset: {exc}")
        st.info(
            "Confirm that the CSV contains the required Date, CBP, transfer, "
            "HHS care, and discharge columns."
        )
        st.stop()
    except Exception as exc:
        st.exception(exc)
        st.stop()

    minimum_date = cleaned_data.index.min().date()
    maximum_date = cleaned_data.index.max().date()
    with st.sidebar:
        selected_dates = st.date_input(
            "Date range",
            value=(minimum_date, maximum_date),
            min_value=minimum_date,
            max_value=maximum_date,
            help="Limit KPI and chart calculations to this period.",
        )
        granularity = st.selectbox(
            "Time granularity",
            options=["Daily", "Weekly", "Monthly"],
            help="Stocks use period-end values; daily flows are summed.",
        )
        st.divider()
        st.caption(f"Data source: {source_name}")

    if not isinstance(selected_dates, (tuple, list)) or len(selected_dates) != 2:
        st.info("Select both a start date and an end date to continue.")
        st.stop()

    start_date, end_date = selected_dates
    if start_date > end_date:
        st.error("The start date must be on or before the end date.")
        st.stop()

    selected_data = cleaned_data.loc[pd.Timestamp(start_date) : pd.Timestamp(end_date)].copy()
    if selected_data.empty:
        st.warning("No observations are available for the selected range.")
        st.stop()

    try:
        daily_metrics = compute_capacity_metrics(selected_data)  # noqa: F405
        kpis = calculate_kpis(daily_metrics)  # noqa: F405
        chart_data = aggregate_chart_data(daily_metrics, granularity)
        active_streak = calculate_active_backlog_streak(daily_metrics)
    except (DataValidationError, TypeError, ValueError) as exc:  # noqa: F405
        st.error(f"Unable to calculate dashboard metrics: {exc}")
        st.stop()

    st.caption(
        f"Reporting period: {start_date:%d %b %Y}–{end_date:%d %b %Y} "
        f"• {len(selected_data):,} daily observation(s) "
        f"• Chart granularity: {granularity}"
    )
    st.markdown(
        '<div class="section-heading">Key Performance Indicators</div>',
        unsafe_allow_html=True,
    )
    render_kpi_summary(kpis)

    st.markdown(
        '<div class="section-heading">Capacity Analytics</div>',
        unsafe_allow_html=True,
    )
    system_tab, comparison_tab, backlog_tab = st.tabs(
        [
            "System Load Overview",
            "CBP vs HHS Comparison",
            "Net Intake & Backlog",
        ]
    )
    with system_tab:
        render_system_load_chart(chart_data)
    with comparison_tab:
        render_cbp_hhs_chart(chart_data)
    with backlog_tab:
        render_net_intake_chart(chart_data, active_streak, granularity)

    render_validation_logs(cleaned_data, selected_data)
    st.caption(
        "Decision-support dashboard. Synthetic data is for demonstration only; "
        "validate findings against authoritative HHS and CBP sources."
    )


if __name__ == "__main__":
    main()
