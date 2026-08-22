"""Executive insights page for the HHS UAC care-load analytics project.

Run directly with::

    streamlit run app/pages/insights.py
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


st.set_page_config(
    page_title="Executive Insights | HHS UAC Analytics",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_page_styles() -> None:
    """Apply a professional slate/navy theme to the insights page."""
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
                min-height: 126px;
                padding: .9rem 1rem;
            }}
            div[data-testid="stMetricLabel"] {{ color: {SLATE_500}; }}
            div[data-testid="stMetricValue"] {{ color: {NAVY}; }}
            .insight-card {{
                background-color: {WHITE};
                border: 1px solid {SLATE_200};
                border-left: 5px solid {BLUE};
                border-radius: 7px;
                color: {SLATE_700};
                min-height: 135px;
                padding: .9rem 1rem;
            }}
            .insight-card h4 {{ color: {NAVY}; margin: 0 0 .4rem 0; }}
            .insight-warning {{ border-left-color: {AMBER}; }}
            .insight-critical {{ border-left-color: {RED}; }}
            .insight-positive {{ border-left-color: {GREEN}; }}
            .method-note {{
                background-color: #EFF6FF;
                border: 1px solid #B9D5F0;
                border-radius: 7px;
                color: #173A5E;
                margin: .5rem 0 1rem 0;
                padding: .75rem 1rem;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_uploaded_csv(file_bytes: bytes) -> pd.DataFrame:
    """Parse uploaded CSV bytes using string-first ingestion."""
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


def safe_percentage_change(start_value: float, end_value: float) -> float:
    """Calculate percentage change without returning infinite values."""
    if not np.isfinite(start_value) or not np.isfinite(end_value) or start_value == 0:
        return 0.0
    return float((end_value - start_value) / abs(start_value) * 100)


def recent_period_change(metrics: pd.DataFrame, window_days: int) -> float:
    """Compare the latest window's mean load with the preceding window."""
    loads = pd.to_numeric(metrics[TOTAL_LOAD_COLUMN], errors="coerce").dropna()
    if len(loads) < 2:
        return 0.0

    effective_window = min(window_days, max(1, len(loads) // 2))
    current_mean = float(loads.iloc[-effective_window:].mean())
    previous_mean = float(loads.iloc[-2 * effective_window : -effective_window].mean())
    return safe_percentage_change(previous_mean, current_mean)


def aggregate_for_chart(metrics: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """Aggregate stock and flow measures appropriately for chart display."""
    if granularity == "Daily":
        return metrics.copy()

    frequency = pd.offsets.Week(weekday=6) if granularity == "Weekly" else pd.offsets.MonthEnd()
    stock_columns = [
        CBP_COLUMN,
        HHS_COLUMN,
        TOTAL_LOAD_COLUMN,
        ROLLING_7_COLUMN,
        ROLLING_14_COLUMN,
        GROWTH_RATE_COLUMN,
    ]
    flow_columns = [
        INTAKE_COLUMN,
        TRANSFER_COLUMN,
        DISCHARGE_COLUMN,
        NET_INTAKE_COLUMN,
    ]
    aggregations = {
        **{column: "last" for column in stock_columns if column in metrics},
        **{column: "sum" for column in flow_columns if column in metrics},
    }
    return metrics.resample(frequency).agg(aggregations).dropna(how="all")


def build_risk_signals(metrics: pd.DataFrame) -> pd.DataFrame:
    """Identify unusual load, growth, pressure, and logical-anomaly dates.

    Thresholds are distribution-aware and calculated only from the selected
    reporting period. These are screening signals, not forecasts or diagnoses.
    """
    if metrics.empty:
        return pd.DataFrame()

    load = pd.to_numeric(metrics[TOTAL_LOAD_COLUMN], errors="coerce")
    net_intake = pd.to_numeric(metrics[NET_INTAKE_COLUMN], errors="coerce")
    growth = pd.to_numeric(metrics[GROWTH_RATE_COLUMN], errors="coerce").abs()

    load_threshold = float(load.quantile(0.95))
    pressure_threshold = float(net_intake.quantile(0.90))
    growth_threshold = float(growth.dropna().quantile(0.95)) if growth.notna().any() else 0

    records: list[dict[str, object]] = []
    for reporting_date, row in metrics.iterrows():
        signals: list[str] = []
        row_load = float(
            pd.to_numeric(pd.Series([row[TOTAL_LOAD_COLUMN]]), errors="coerce").fillna(0).iloc[0]
        )
        row_pressure = float(
            pd.to_numeric(pd.Series([row[NET_INTAKE_COLUMN]]), errors="coerce").fillna(0).iloc[0]
        )
        row_growth = float(
            pd.to_numeric(pd.Series([row[GROWTH_RATE_COLUMN]]), errors="coerce").fillna(0).iloc[0]
        )

        if row_load >= load_threshold:
            signals.append("High system load")
        if row_pressure > 0 and row_pressure >= pressure_threshold:
            signals.append("Elevated net intake")
        if abs(row_growth) >= growth_threshold and growth_threshold > 0:
            signals.append("Unusual load change")
        transfer_anomaly = bool(row.get(TRANSFER_ANOMALY_COLUMN, False))
        discharge_anomaly = bool(row.get(DISCHARGE_ANOMALY_COLUMN, False))
        if transfer_anomaly:
            signals.append("Transfer constraint anomaly")
        if discharge_anomaly:
            signals.append("Discharge constraint anomaly")

        if not signals:
            continue

        logical_anomaly = transfer_anomaly or discharge_anomaly
        severity = "Critical" if logical_anomaly or len(signals) >= 3 else "Watch"
        records.append(
            {
                "Date": reporting_date,
                "Severity": severity,
                "Signals": "; ".join(signals),
                "Total System Load": int(row_load),
                "Net Daily Intake": int(row_pressure),
                "Growth Rate (%)": row_growth,
            }
        )

    if not records:
        return pd.DataFrame(
            columns=[
                "Date",
                "Severity",
                "Signals",
                "Total System Load",
                "Net Daily Intake",
                "Growth Rate (%)",
            ]
        )
    risk_frame = pd.DataFrame.from_records(records)
    severity_order = pd.CategoricalDtype(["Critical", "Watch"], ordered=True)
    risk_frame["Severity"] = risk_frame["Severity"].astype(severity_order)
    return risk_frame.sort_values(["Severity", "Date"], ascending=[True, False])


def build_correlation_matrix(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return a labeled correlation matrix for principal operational measures."""
    selected = metrics[
        [
            INTAKE_COLUMN,
            TRANSFER_COLUMN,
            DISCHARGE_COLUMN,
            CBP_COLUMN,
            HHS_COLUMN,
            TOTAL_LOAD_COLUMN,
            NET_INTAKE_COLUMN,
        ]
    ].apply(pd.to_numeric, errors="coerce")
    selected = selected.rename(
        columns={
            INTAKE_COLUMN: "Apprehensions",
            TRANSFER_COLUMN: "Transfers",
            DISCHARGE_COLUMN: "Discharges",
            CBP_COLUMN: "CBP Load",
            HHS_COLUMN: "HHS Load",
            TOTAL_LOAD_COLUMN: "System Load",
            NET_INTAKE_COLUMN: "Net Intake",
        }
    )
    return selected.corr(min_periods=2).fillna(0)


def chart_layout(title: str, y_title: str, *, height: int = 460) -> dict:
    """Return common Plotly layout settings."""
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


def render_insight_cards(
    metrics: pd.DataFrame,
    recent_change: float,
    cumulative_net_intake: int,
    offset_ratio: float,
    anomaly_count: int,
    comparison_days: int,
) -> None:
    """Render four plain-language, evidence-based analytical observations."""
    latest_load = int(metrics[TOTAL_LOAD_COLUMN].iloc[-1])
    load_class = (
        "insight-warning" if recent_change > 1 else "insight-positive" if recent_change < -1 else ""
    )
    load_direction = (
        "increased" if recent_change > 0 else "decreased" if recent_change < 0 else "held steady"
    )
    pressure_class = "insight-warning" if cumulative_net_intake > 0 else "insight-positive"
    pressure_text = (
        "Transfers exceeded discharges overall, indicating accumulated intake pressure."
        if cumulative_net_intake > 0
        else "Discharges matched or exceeded transfers overall, indicating net relief."
    )
    offset_class = "insight-positive" if offset_ratio >= 1 else "insight-warning"
    anomaly_class = "insight-critical" if anomaly_count else "insight-positive"

    columns = st.columns(4)
    with columns[0]:
        st.markdown(
            f"""
            <div class="insight-card {load_class}">
                <h4>Recent load direction</h4>
                The latest {comparison_days}-day average {load_direction} by
                <strong>{abs(recent_change):.1f}%</strong> versus the preceding window.
                Latest system load: <strong>{latest_load:,}</strong>.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with columns[1]:
        st.markdown(
            f"""
            <div class="insight-card {pressure_class}">
                <h4>Flow balance</h4>
                Cumulative net intake was <strong>{cumulative_net_intake:+,}</strong>.
                {pressure_text}
            </div>
            """,
            unsafe_allow_html=True,
        )
    with columns[2]:
        st.markdown(
            f"""
            <div class="insight-card {offset_class}">
                <h4>Discharge coverage</h4>
                Period discharge offset was <strong>{offset_ratio:.1%}</strong>.
                A value at or above 100% means aggregate discharges matched or
                exceeded transfers.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with columns[3]:
        st.markdown(
            f"""
            <div class="insight-card {anomaly_class}">
                <h4>Data confidence</h4>
                <strong>{anomaly_count:,}</strong> selected row(s) violated the
                transfer or discharge logical constraints and require review.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_load_trend(chart_data: pd.DataFrame, granularity: str) -> None:
    """Render system load with daily smoothing references."""
    figure = go.Figure()
    for column, label, color, dash, width in (
        (TOTAL_LOAD_COLUMN, "Total System Load", NAVY, "solid", 2.4),
        (ROLLING_7_COLUMN, "7-day average", BLUE, "dot", 2),
        (ROLLING_14_COLUMN, "14-day average", TEAL, "dash", 2),
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
        **chart_layout(f"System Load Trend ({granularity})", "Children under care")
    )
    figure.update_yaxes(rangemode="tozero")
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})


def render_flow_chart(chart_data: pd.DataFrame, granularity: str) -> None:
    """Render transfers, discharges, and net intake on a shared timeline."""
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
    figure.add_trace(
        go.Bar(
            x=chart_data.index,
            y=chart_data[NET_INTAKE_COLUMN],
            name="Net intake",
            marker_color=np.where(chart_data[NET_INTAKE_COLUMN].gt(0), RED, GREEN),
            opacity=0.35,
            hovertemplate="%{y:+,.0f}<extra></extra>",
        )
    )
    figure.add_hline(y=0, line_color=SLATE_500, line_width=1)
    figure.update_layout(**chart_layout(f"Transfer–Discharge Drivers ({granularity})", "Children"))
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})


def render_correlation_heatmap(metrics: pd.DataFrame) -> None:
    """Render correlations among major system stock and flow measures."""
    correlation = build_correlation_matrix(metrics)
    figure = go.Figure(
        go.Heatmap(
            z=correlation.values,
            x=correlation.columns,
            y=correlation.index,
            zmin=-1,
            zmax=1,
            colorscale=[
                [0, RED],
                [0.5, WHITE],
                [1, BLUE],
            ],
            text=np.round(correlation.values, 2),
            texttemplate="%{text:.2f}",
            hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
            colorbar={"title": "Correlation"},
        )
    )
    figure.update_layout(
        title={"text": "Operational Measure Correlations", "font": {"color": NAVY}},
        height=530,
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        margin={"l": 95, "r": 30, "t": 65, "b": 95},
        font={"family": "Arial, sans-serif", "color": SLATE_700},
    )
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})
    st.caption(
        "Correlation describes association, not causation. Values near +1 move "
        "together; values near −1 move in opposite directions."
    )


def render_risk_timeline(metrics: pd.DataFrame, risk_signals: pd.DataFrame) -> None:
    """Render the system-load timeline with screening-signal markers."""
    figure = go.Figure(
        go.Scatter(
            x=metrics.index,
            y=metrics[TOTAL_LOAD_COLUMN],
            name="Total System Load",
            mode="lines",
            line={"color": NAVY, "width": 2},
            hovertemplate="%{y:,.0f} children<extra></extra>",
        )
    )
    if not risk_signals.empty:
        for severity, color, size in (
            ("Watch", AMBER, 8),
            ("Critical", RED, 11),
        ):
            subset = risk_signals.loc[risk_signals["Severity"].astype("string").eq(severity)]
            if subset.empty:
                continue
            figure.add_trace(
                go.Scatter(
                    x=subset["Date"],
                    y=subset["Total System Load"],
                    name=f"{severity} signal",
                    mode="markers",
                    marker={"color": color, "size": size, "line_width": 0.5},
                    customdata=subset[["Signals", "Net Daily Intake"]],
                    hovertemplate=(
                        "%{y:,.0f} children<br>%{customdata[0]}"
                        "<br>Net intake: %{customdata[1]:+,.0f}<extra></extra>"
                    ),
                )
            )
    figure.update_layout(**chart_layout("System Load Screening Signals", "Children under care"))
    figure.update_yaxes(rangemode="tozero")
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})


def render_validation_log(selected_data: pd.DataFrame) -> None:
    """Display logical constraint violations in an expandable audit section."""
    with st.expander("Insights Data Quality & Validation Log", expanded=False):
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
    """Build the executive insights page."""
    apply_page_styles()
    st.markdown(
        """
        <div class="page-header">
            <h1>Executive Insights & System Signals</h1>
            <p>Translate care-load, transfer, discharge, and data-quality measures
            into concise operational findings and screening signals.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Insights Controls")
        uploaded_file = st.file_uploader(
            "Upload HHS capacity data",
            type=["csv"],
            help="Leave empty to use the synthetic 2023-2025 dataset.",
            key="insights_csv_uploader",
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
        st.error(f"Unable to prepare insights data: {exc}")
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
            key="insights_date_range",
        )
        granularity = st.selectbox(
            "Chart granularity",
            ["Daily", "Weekly", "Monthly"],
            key="insights_granularity",
        )
        comparison_days = st.slider(
            "Recent comparison window",
            min_value=3,
            max_value=30,
            value=7,
            help="Compare the latest average load with the preceding window.",
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
        risk_signals = build_risk_signals(daily_metrics)
    except (DataValidationError, TypeError, ValueError) as exc:
        st.error(f"Unable to calculate analytical insights: {exc}")
        st.stop()

    latest_load = int(daily_metrics[TOTAL_LOAD_COLUMN].iloc[-1])
    period_change = safe_percentage_change(
        float(daily_metrics[TOTAL_LOAD_COLUMN].iloc[0]),
        float(daily_metrics[TOTAL_LOAD_COLUMN].iloc[-1]),
    )
    recent_change = recent_period_change(daily_metrics, comparison_days)
    cumulative_net_intake = int(daily_metrics[NET_INTAKE_COLUMN].sum())
    average_offset_ratio = float(
        daily_metrics[DISCHARGE_COLUMN].sum() / (daily_metrics[TRANSFER_COLUMN].sum() + 1)
    )
    anomaly_count = int(
        selected_data[[TRANSFER_ANOMALY_COLUMN, DISCHARGE_ANOMALY_COLUMN]]
        .fillna(False)
        .any(axis=1)
        .sum()
    )

    st.caption(
        f"Reporting period: {start_date:%d %b %Y}–{end_date:%d %b %Y} "
        f"• {len(selected_data):,} daily observations • {granularity} charts"
    )
    kpi_columns = st.columns(5)
    kpi_columns[0].metric(
        "Latest System Load",
        f"{latest_load:,}",
        delta=f"{period_change:+.1f}% over period",
        delta_color="inverse",
        help="Latest CBP plus HHS active load; delta compares period endpoints.",
    )
    kpi_columns[1].metric(
        "Recent Load Shift",
        f"{recent_change:+.1f}%",
        help=f"Latest {comparison_days}-day mean versus the preceding window.",
    )
    kpi_columns[2].metric(
        "Cumulative Net Intake",
        f"{cumulative_net_intake:+,}",
        help="Selected-period transfers minus discharges.",
    )
    kpi_columns[3].metric(
        "Care Load Volatility",
        f"{kpis['care_load_volatility_index']:.2f}%",
        help="Standard deviation of daily Total System Load growth.",
    )
    kpi_columns[4].metric(
        "Risk Signal Days",
        f"{len(risk_signals):,}",
        help="Dates meeting distribution-aware load, intake, growth, or anomaly rules.",
    )

    overview_tab, drivers_tab, risks_tab = st.tabs(
        ["Executive Summary", "Drivers & Relationships", "Risk Signals"]
    )
    with overview_tab:
        render_insight_cards(
            daily_metrics,
            recent_change,
            cumulative_net_intake,
            average_offset_ratio,
            anomaly_count,
            comparison_days,
        )
        render_load_trend(chart_data, granularity)
    with drivers_tab:
        render_flow_chart(chart_data, granularity)
        render_correlation_heatmap(daily_metrics)
    with risks_tab:
        st.markdown(
            """
            <div class="method-note">
                <strong>Screening method:</strong> Signals identify selected-period
                upper-tail system load, net intake, absolute daily growth, and logical
                constraint anomalies. They support review and are not forecasts.
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_risk_timeline(daily_metrics, risk_signals)
        if risk_signals.empty:
            st.success("No screening signals were detected in the selected period.")
        else:
            display = risk_signals.copy()
            display["Date"] = pd.to_datetime(display["Date"]).dt.date
            st.dataframe(
                display,
                width="stretch",
                hide_index=True,
                column_config={"Growth Rate (%)": st.column_config.NumberColumn(format="%.2f%%")},
            )
            st.download_button(
                "Download risk signal log",
                data=display.to_csv(index=False).encode("utf-8"),
                file_name=f"uac_risk_signals_{start_date}_{end_date}.csv",
                mime="text/csv",
            )

    render_validation_log(selected_data)
    st.caption(
        "Descriptive decision-support view. Correlations and screening signals do "
        "not establish causation or predict future care loads."
    )


if __name__ == "__main__":
    main()
