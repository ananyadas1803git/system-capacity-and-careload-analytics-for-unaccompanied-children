"""Longitudinal trend analysis page for the HHS UAC capacity dashboard.

Run directly with::

    streamlit run app/pages/trends.py
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
    CBP_COLUMN,
    DISCHARGE_ANOMALY_COLUMN,
    DISCHARGE_COLUMN,
    GROWTH_RATE_COLUMN,
    HHS_COLUMN,
    INTAKE_COLUMN,
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
LIGHT_BLUE = "#77A9D4"
TEAL = "#167C80"
AMBER = "#D97706"
RED = "#B42318"
GREEN = "#238636"
PURPLE = "#7656A6"

CUSTOM_TREND_COLUMN = "Custom Load Trend"


st.set_page_config(
    page_title="Longitudinal Trends | HHS UAC Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_page_styles() -> None:
    """Apply a consistent slate/navy government-analytics theme."""
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
                border-left: 4px solid {TEAL};
                border-radius: 8px;
                min-height: 130px;
                padding: .9rem 1rem;
            }}
            div[data-testid="stMetricLabel"] {{ color: {SLATE_500}; }}
            div[data-testid="stMetricValue"] {{ color: {NAVY}; }}
            .trend-note {{
                background-color: #EFF6FF;
                border: 1px solid #B9D5F0;
                border-left: 5px solid {BLUE};
                border-radius: 7px;
                color: #173A5E;
                margin: .5rem 0 1rem 0;
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
    """Return the cached synthetic 2023-2025 dataset."""
    return generate_mock_data()


def add_custom_trend(metrics: pd.DataFrame, smoothing_days: int) -> pd.DataFrame:
    """Add a user-controlled daily moving average of Total System Load."""
    frame = metrics.copy()
    frame[CUSTOM_TREND_COLUMN] = frame[TOTAL_LOAD_COLUMN].rolling(
        smoothing_days,
        min_periods=1,
    ).mean()
    return frame


def aggregate_for_chart(metrics: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """Aggregate daily values while preserving stock-versus-flow semantics."""
    if granularity == "Daily":
        return metrics.copy()

    frequency = (
        pd.offsets.Week(weekday=6)
        if granularity == "Weekly"
        else pd.offsets.MonthEnd()
    )
    last_columns = [
        CBP_COLUMN,
        HHS_COLUMN,
        TOTAL_LOAD_COLUMN,
        ROLLING_7_COLUMN,
        ROLLING_14_COLUMN,
        CUSTOM_TREND_COLUMN,
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
    aggregated[GROWTH_RATE_COLUMN] = (
        aggregated[TOTAL_LOAD_COLUMN]
        .pct_change(fill_method=None)
        .replace([np.inf, -np.inf], np.nan)
        .mul(100)
    )
    return aggregated


def calculate_trend_slope(metrics: pd.DataFrame) -> float:
    """Calculate the least-squares Total System Load slope in children/day."""
    load = pd.to_numeric(metrics[TOTAL_LOAD_COLUMN], errors="coerce")
    valid = load.notna()
    if valid.sum() < 2:
        return 0.0
    x_values = np.arange(len(load), dtype=float)[valid.to_numpy()]
    y_values = load.loc[valid].to_numpy(dtype=float)
    return float(np.polyfit(x_values, y_values, 1)[0])


def calculate_period_change(metrics: pd.DataFrame) -> float:
    """Calculate Total System Load change between selected endpoints."""
    if len(metrics) < 2:
        return 0.0
    start = float(metrics[TOTAL_LOAD_COLUMN].iloc[0])
    end = float(metrics[TOTAL_LOAD_COLUMN].iloc[-1])
    if start == 0 or not np.isfinite(start) or not np.isfinite(end):
        return 0.0
    return float((end - start) / abs(start) * 100)


def build_change_table(metrics: pd.DataFrame, number_of_rows: int = 10) -> pd.DataFrame:
    """Return the largest absolute daily care-load movements."""
    working = metrics[[TOTAL_LOAD_COLUMN, NET_INTAKE_COLUMN, GROWTH_RATE_COLUMN]].copy()
    working["Absolute Growth"] = pd.to_numeric(
        working[GROWTH_RATE_COLUMN], errors="coerce"
    ).abs()
    working = working.dropna(subset=["Absolute Growth"])
    if working.empty:
        return pd.DataFrame(
            columns=[
                "Date",
                "Total System Load",
                "Net Daily Intake",
                "Growth Rate (%)",
                "Direction",
            ]
        )
    top = working.nlargest(number_of_rows, "Absolute Growth").copy()
    top["Direction"] = np.where(
        top[GROWTH_RATE_COLUMN].gt(0),
        "Increase",
        "Decrease",
    )
    top = top.rename(columns={GROWTH_RATE_COLUMN: "Growth Rate (%)"})
    result = top.reset_index().drop(columns="Absolute Growth")
    # ValidationReport is useful on the analytical frame but is not Arrow/JSON
    # serializable when a presentation table is sent to Streamlit.
    result.attrs.clear()
    return result


def weekday_profile(metrics: pd.DataFrame) -> pd.DataFrame:
    """Calculate average flows for each day of the week."""
    working = metrics[
        [INTAKE_COLUMN, TRANSFER_COLUMN, DISCHARGE_COLUMN, NET_INTAKE_COLUMN]
    ].copy()
    working["Weekday Number"] = working.index.dayofweek
    grouped = working.groupby("Weekday Number").mean(numeric_only=True).reindex(range(7))
    grouped.index = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    return grouped.fillna(0)


def monthly_heatmap_data(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return mean system load by year and calendar month."""
    working = metrics[[TOTAL_LOAD_COLUMN]].copy()
    working["Year"] = working.index.year
    working["Month"] = working.index.month
    pivot = working.pivot_table(
        index="Year",
        columns="Month",
        values=TOTAL_LOAD_COLUMN,
        aggfunc="mean",
    )
    return pivot.reindex(columns=range(1, 13))


def chart_layout(title: str, y_title: str, *, height: int = 455) -> dict:
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


def render_load_trend_chart(
    chart_data: pd.DataFrame,
    smoothing_days: int,
    show_linear_trend: bool,
    granularity: str,
) -> None:
    """Render system load, custom smoothing, and optional regression trend."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data[TOTAL_LOAD_COLUMN],
            name="Total System Load",
            mode="lines",
            line={"color": NAVY, "width": 1.8},
            hovertemplate="%{y:,.0f} children<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data[CUSTOM_TREND_COLUMN],
            name=f"{smoothing_days}-day moving average",
            mode="lines",
            line={"color": TEAL, "width": 2.5},
            hovertemplate="%{y:,.1f} children<extra></extra>",
        )
    )
    if show_linear_trend and len(chart_data) > 1:
        y_values = pd.to_numeric(
            chart_data[TOTAL_LOAD_COLUMN], errors="coerce"
        ).interpolate(limit_direction="both")
        x_values = np.arange(len(chart_data), dtype=float)
        coefficients = np.polyfit(x_values, y_values.to_numpy(dtype=float), 1)
        fitted = np.polyval(coefficients, x_values)
        figure.add_trace(
            go.Scatter(
                x=chart_data.index,
                y=fitted,
                name="Linear direction",
                mode="lines",
                line={"color": AMBER, "width": 2, "dash": "dash"},
                hovertemplate="%{y:,.1f} children<extra></extra>",
            )
        )
    figure.update_layout(
        **chart_layout(f"Long-Term System Load Trend ({granularity})", "Children")
    )
    figure.update_xaxes(
        rangeslider={"visible": granularity == "Daily"},
        rangeselector={
            "buttons": [
                {"count": 3, "label": "3m", "step": "month", "stepmode": "backward"},
                {"count": 6, "label": "6m", "step": "month", "stepmode": "backward"},
                {"count": 1, "label": "1y", "step": "year", "stepmode": "backward"},
                {"step": "all", "label": "All"},
            ]
        },
    )
    figure.update_yaxes(rangemode="tozero")
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})


def render_flow_trend_chart(chart_data: pd.DataFrame, granularity: str) -> None:
    """Render apprehensions, transfers, and discharges over time."""
    figure = go.Figure()
    for column, label, color in (
        (INTAKE_COLUMN, "CBP apprehensions", BLUE),
        (TRANSFER_COLUMN, "Transfers", AMBER),
        (DISCHARGE_COLUMN, "Discharges", TEAL),
    ):
        figure.add_trace(
            go.Scatter(
                x=chart_data.index,
                y=chart_data[column],
                name=label,
                mode="lines",
                line={"color": color, "width": 1.9},
                hovertemplate="%{y:,.0f} children<extra></extra>",
            )
        )
    figure.update_layout(
        **chart_layout(f"System Flow Trends ({granularity})", "Children")
    )
    figure.update_yaxes(rangemode="tozero")
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})


def render_cumulative_pressure_chart(
    chart_data: pd.DataFrame,
    granularity: str,
) -> None:
    """Render cumulative transfers-minus-discharges pressure."""
    cumulative = pd.to_numeric(
        chart_data[NET_INTAKE_COLUMN], errors="coerce"
    ).fillna(0).cumsum()
    figure = go.Figure(
        go.Scatter(
            x=chart_data.index,
            y=cumulative,
            name="Cumulative net intake",
            mode="lines",
            fill="tozeroy",
            line={"color": PURPLE, "width": 2},
            fillcolor="rgba(118, 86, 166, 0.15)",
            hovertemplate="%{y:+,.0f} children<extra></extra>",
        )
    )
    figure.add_hline(y=0, line_color=SLATE_500, line_width=1)
    figure.update_layout(
        **chart_layout(
            f"Cumulative Net Intake Pressure ({granularity})",
            "Cumulative transfers minus discharges",
        )
    )
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})


def render_weekday_profile(metrics: pd.DataFrame) -> None:
    """Render average intake, transfer, and discharge flows by weekday."""
    profile = weekday_profile(metrics)
    figure = go.Figure()
    for column, label, color in (
        (INTAKE_COLUMN, "Apprehensions", BLUE),
        (TRANSFER_COLUMN, "Transfers", AMBER),
        (DISCHARGE_COLUMN, "Discharges", TEAL),
    ):
        figure.add_trace(
            go.Bar(
                x=profile.index,
                y=profile[column],
                name=label,
                marker_color=color,
                hovertemplate="%{y:,.1f} average<extra></extra>",
            )
        )
    figure.update_layout(
        **chart_layout("Average Daily Flows by Weekday", "Average children/day")
    )
    figure.update_layout(barmode="group")
    figure.update_xaxes(title="Day of week")
    figure.update_yaxes(rangemode="tozero")
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})


def render_monthly_heatmap(metrics: pd.DataFrame) -> None:
    """Render average monthly Total System Load for each selected year."""
    heatmap = monthly_heatmap_data(metrics)
    month_labels = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    figure = go.Figure(
        go.Heatmap(
            z=heatmap.values,
            x=month_labels,
            y=heatmap.index.astype(str),
            colorscale=[[0, "#E8F1F8"], [0.5, LIGHT_BLUE], [1, NAVY]],
            colorbar={"title": "Avg load"},
            hovertemplate="%{y} %{x}<br>%{z:,.0f} children<extra></extra>",
        )
    )
    figure.update_layout(
        title={"text": "Monthly System Load Seasonality", "font": {"color": NAVY}},
        height=420,
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        font={"family": "Arial, sans-serif", "color": SLATE_700},
        margin={"l": 60, "r": 30, "t": 65, "b": 50},
        xaxis={"title": "Calendar month"},
        yaxis={"title": "Year"},
    )
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})


def render_growth_distribution(chart_data: pd.DataFrame, granularity: str) -> None:
    """Render period-over-period system-load growth and its distribution."""
    growth = pd.to_numeric(
        chart_data[GROWTH_RATE_COLUMN], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    figure = go.Figure(
        go.Bar(
            x=chart_data.index,
            y=growth,
            name="Load growth",
            marker_color=np.where(growth.fillna(0).gt(0), RED, GREEN),
            hovertemplate="%{y:+.2f}%<extra></extra>",
        )
    )
    figure.add_hline(y=0, line_color=SLATE_500, line_width=1)
    figure.update_layout(
        **chart_layout(f"System Load Growth Rate ({granularity})", "Growth rate (%)")
    )
    figure.update_yaxes(ticksuffix="%")
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})


def render_anomaly_log(selected_data: pd.DataFrame) -> None:
    """Display logical constraint violations for the selected period."""
    with st.expander("Trends Data Quality & Validation Log", expanded=False):
        anomaly_columns = [
            TRANSFER_ANOMALY_COLUMN,
            DISCHARGE_ANOMALY_COLUMN,
        ]
        missing = [column for column in anomaly_columns if column not in selected_data]
        if missing:
            st.warning("Anomaly fields are unavailable: " + ", ".join(missing))
            return

        flagged = selected_data.loc[
            selected_data[anomaly_columns].fillna(False).any(axis=1)
        ]
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
    """Build the complete longitudinal trends page."""
    apply_page_styles()
    st.markdown(
        """
        <div class="page-header">
            <h1>Longitudinal Trends & Seasonality</h1>
            <p>Explore long-term system load, operational flows, cumulative
            pressure, recurring calendar patterns, and significant load changes.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Trend Controls")
        uploaded_file = st.file_uploader(
            "Upload HHS capacity data",
            type=["csv"],
            help="Leave empty to use the synthetic 2023-2025 dataset.",
            key="trends_csv_uploader",
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
        st.error(f"Unable to prepare trend data: {exc}")
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
            key="trends_date_range",
        )
        granularity = st.selectbox(
            "Chart granularity",
            ["Daily", "Weekly", "Monthly"],
            key="trends_granularity",
        )
        smoothing_days = st.slider(
            "Load smoothing window",
            min_value=3,
            max_value=90,
            value=30,
            help="Daily window used for the custom moving-average trend.",
        )
        show_linear_trend = st.checkbox(
            "Show linear direction",
            value=True,
            help="Adds a descriptive least-squares trend line; it is not a forecast.",
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
        daily_metrics = add_custom_trend(daily_metrics, smoothing_days)
        chart_data = aggregate_for_chart(daily_metrics, granularity)
        change_table = build_change_table(daily_metrics)
    except (DataValidationError, TypeError, ValueError) as exc:
        st.error(f"Unable to calculate longitudinal trends: {exc}")
        st.stop()

    period_change = calculate_period_change(daily_metrics)
    trend_slope = calculate_trend_slope(daily_metrics)
    peak_date = daily_metrics[TOTAL_LOAD_COLUMN].idxmax()
    peak_load = int(daily_metrics.loc[peak_date, TOTAL_LOAD_COLUMN])
    average_net_intake = float(daily_metrics[NET_INTAKE_COLUMN].mean())
    total_transfers = float(daily_metrics[TRANSFER_COLUMN].sum())
    period_offset = float(
        daily_metrics[DISCHARGE_COLUMN].sum() / (total_transfers + 1)
    )

    st.caption(
        f"Reporting period: {start_date:%d %b %Y}–{end_date:%d %b %Y} "
        f"• {len(selected_data):,} daily observations • {granularity} charts"
    )
    kpi_columns = st.columns(5)
    kpi_columns[0].metric(
        "Period Load Change",
        f"{period_change:+.1f}%",
        delta_color="inverse",
        help="Percentage change between selected-period endpoint loads.",
    )
    kpi_columns[1].metric(
        "Daily Trend Slope",
        f"{trend_slope:+.2f}",
        delta="children/day",
        delta_color="inverse",
        help="Least-squares descriptive load direction; not a forecast.",
    )
    kpi_columns[2].metric(
        "Peak System Load",
        f"{peak_load:,}",
        delta=f"{peak_date:%d %b %Y}",
        delta_color="off",
        help="Highest Total System Load in the selected period.",
    )
    kpi_columns[3].metric(
        "Average Net Intake",
        f"{average_net_intake:+.1f}/day",
        delta_color="inverse",
        help="Mean daily transfers minus HHS discharges.",
    )
    kpi_columns[4].metric(
        "Period Discharge Offset",
        f"{period_offset:.1%}",
        help="Aggregate discharges divided by aggregate transfers plus one.",
    )

    st.markdown(
        f"""
        <div class="trend-note">
            <strong>Trend interpretation:</strong> The fitted daily direction is
            {trend_slope:+.2f} children per day and selected-period load changed
            {period_change:+.1f}%. The regression line summarizes historical direction
            only and must not be interpreted as a demand forecast.
        </div>
        """,
        unsafe_allow_html=True,
    )

    load_tab, flows_tab, seasonality_tab, changes_tab = st.tabs(
        ["Load Trend", "Flow & Pressure Trends", "Seasonality", "Growth & Changes"]
    )
    with load_tab:
        render_load_trend_chart(
            chart_data,
            smoothing_days,
            show_linear_trend,
            granularity,
        )
    with flows_tab:
        render_flow_trend_chart(chart_data, granularity)
        render_cumulative_pressure_chart(chart_data, granularity)
    with seasonality_tab:
        render_weekday_profile(daily_metrics)
        render_monthly_heatmap(daily_metrics)
        st.caption(
            "Seasonality charts describe the selected historical period. Partial "
            "months and years may not be directly comparable with complete periods."
        )
    with changes_tab:
        render_growth_distribution(chart_data, granularity)
        st.subheader("Largest Absolute Daily Load Changes")
        st.dataframe(
            change_table,
            width="stretch",
            hide_index=True,
            column_config={
                "Growth Rate (%)": st.column_config.NumberColumn(format="%.2f%%")
            },
        )

    render_anomaly_log(selected_data)
    st.download_button(
        "Download displayed trend data",
        data=chart_data.reset_index().to_csv(index=False).encode("utf-8"),
        file_name=f"uac_trends_{granularity.lower()}_{start_date}_{end_date}.csv",
        mime="text/csv",
    )
    st.caption(
        "Descriptive trend analysis. Historical relationships and fitted directions "
        "do not predict future care loads or establish causation."
    )


if __name__ == "__main__":
    main()
