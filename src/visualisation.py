"""Reusable Plotly visualizations for HHS UAC capacity analytics.

The functions in this module contain no Streamlit calls.  They accept raw,
preprocessed, or metric-enriched DataFrames and return fully configured Plotly
figures that can be rendered by Streamlit, notebooks, APIs, or HTML reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app_utils import (
    BACKLOG_STREAK_COLUMN,
    CBP_COLUMN,
    DATE_COLUMN,
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
    compute_capacity_metrics,
    validate_and_clean_data,
)
from backend.analytics import (
    AnalyticsError,
    TimeGranularity,
    calculate_backlog_episodes,
    calculate_capacity_scenario,
    filter_reporting_period,
    resample_metrics,
)
from src.validation import DatasetValidationResult, ValidationSeverity


PLOTLY_RENDER_CONFIG: dict[str, Any] = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {
        "format": "png",
        "filename": "hhs_uac_capacity_chart",
        "scale": 2,
    },
}


class VisualizationError(ValueError):
    """Raised when input data cannot support a requested visualization."""


@dataclass(frozen=True)
class GovernmentChartTheme:
    """Accessible slate/navy palette and common chart typography."""

    navy: str = "#163B65"
    slate_950: str = "#172033"
    slate_700: str = "#3F5063"
    slate_500: str = "#66788A"
    slate_200: str = "#D9E1EA"
    white: str = "#FFFFFF"
    blue: str = "#2E75B6"
    light_blue: str = "#77A9D4"
    teal: str = "#167C80"
    amber: str = "#D97706"
    red: str = "#B42318"
    green: str = "#238636"
    purple: str = "#7656A6"
    font_family: str = "Arial, sans-serif"


@dataclass(frozen=True)
class VisualizationConfig:
    """Shared display and reporting-period configuration for chart builders."""

    start_date: str | date | datetime | pd.Timestamp | None = None
    end_date: str | date | datetime | pd.Timestamp | None = None
    granularity: str | TimeGranularity = TimeGranularity.DAILY
    height: int = 440
    backlog_threshold_days: int = 3
    comparison_mode: str = "stacked"
    show_anomalies: bool = True
    show_range_slider: bool = False
    theme: GovernmentChartTheme = field(default_factory=GovernmentChartTheme)

    def __post_init__(self) -> None:
        TimeGranularity.parse(self.granularity)
        if self.height < 250:
            raise ValueError("height must be at least 250 pixels.")
        if self.backlog_threshold_days < 1:
            raise ValueError("backlog_threshold_days must be at least 1.")
        if self.comparison_mode not in {"stacked", "lines"}:
            raise ValueError("comparison_mode must be 'stacked' or 'lines'.")


@dataclass
class VisualizationData:
    """Daily analytical data and its correctly aggregated chart view."""

    daily_metrics: pd.DataFrame
    chart_metrics: pd.DataFrame
    config: VisualizationConfig

    def copy(self) -> VisualizationData:
        """Return a defensive copy suitable for local chart customization."""
        return VisualizationData(
            daily_metrics=self.daily_metrics.copy(),
            chart_metrics=self.chart_metrics.copy(),
            config=self.config,
        )


@dataclass
class DashboardFigures:
    """Core dashboard charts generated from one normalized data pass."""

    system_load: go.Figure
    care_load_comparison: go.Figure
    net_intake_backlog: go.Figure
    operational_flows: go.Figure
    growth_volatility: go.Figure
    data: VisualizationData

    def as_dict(self) -> dict[str, go.Figure]:
        """Return figures under stable names for tabs or report sections."""
        return {
            "system_load": self.system_load,
            "care_load_comparison": self.care_load_comparison,
            "net_intake_backlog": self.net_intake_backlog,
            "operational_flows": self.operational_flows,
            "growth_volatility": self.growth_volatility,
        }


_METRIC_COLUMNS = {
    TOTAL_LOAD_COLUMN,
    NET_INTAKE_COLUMN,
    GROWTH_RATE_COLUMN,
    ROLLING_7_COLUMN,
    ROLLING_14_COLUMN,
    OFFSET_RATIO_COLUMN,
    BACKLOG_STREAK_COLUMN,
}


def _prepare_daily_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw, cleaned, or derived input into sorted daily metrics."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if data.empty:
        raise VisualizationError("No observations are available to visualize.")

    frame = data.copy()
    if _METRIC_COLUMNS.issubset(frame.columns):
        if DATE_COLUMN in frame.columns:
            dates = pd.to_datetime(frame[DATE_COLUMN], errors="coerce")
            if dates.isna().any():
                raise VisualizationError("Metric input contains invalid reporting dates.")
            frame[DATE_COLUMN] = dates
            frame = frame.set_index(DATE_COLUMN)
        elif not isinstance(frame.index, pd.DatetimeIndex):
            raise VisualizationError(
                "Metric input must provide Date as a column or DatetimeIndex."
            )
        frame.index = pd.DatetimeIndex(frame.index, name=DATE_COLUMN)
        if frame.index.has_duplicates:
            raise VisualizationError("Metric input contains duplicate reporting dates.")
        return frame.sort_index()

    raw = frame.reset_index() if DATE_COLUMN not in frame.columns else frame
    try:
        cleaned = validate_and_clean_data(raw)
        metrics = compute_capacity_metrics(cleaned)
    except (DataValidationError, TypeError, ValueError) as exc:
        raise VisualizationError(f"Unable to prepare visualization data: {exc}") from exc
    metrics.index = pd.DatetimeIndex(metrics.index, name=DATE_COLUMN)
    return metrics.sort_index()


def prepare_visualization_data(
    data: pd.DataFrame,
    config: VisualizationConfig | None = None,
) -> VisualizationData:
    """Prepare an inclusive reporting period and stock/flow-aware aggregation."""
    request = config or VisualizationConfig()
    try:
        daily = filter_reporting_period(
            _prepare_daily_metrics(data),
            request.start_date,
            request.end_date,
        )
        chart = resample_metrics(daily, request.granularity)
    except (AnalyticsError, DataValidationError, TypeError, ValueError) as exc:
        raise VisualizationError(f"Unable to prepare chart data: {exc}") from exc
    return VisualizationData(daily_metrics=daily, chart_metrics=chart, config=request)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise VisualizationError("Missing visualization column(s): " + ", ".join(missing))


def apply_government_theme(
    figure: go.Figure,
    *,
    title: str,
    y_title: str,
    config: VisualizationConfig,
    secondary_y_title: str | None = None,
) -> go.Figure:
    """Apply the shared accessible theme to a Plotly figure in place."""
    theme = config.theme
    figure.update_layout(
        title={"text": title, "font": {"size": 18, "color": theme.navy}},
        height=config.height,
        paper_bgcolor=theme.white,
        plot_bgcolor=theme.white,
        font={"family": theme.font_family, "color": theme.slate_700},
        hovermode="x unified",
        margin={"l": 60, "r": 35, "t": 72, "b": 55},
        legend={
            "orientation": "h",
            "x": 0,
            "y": 1.08,
            "bgcolor": "rgba(255,255,255,0.8)",
        },
        xaxis={
            "title": "Reporting date",
            "showgrid": False,
            "rangeslider": {"visible": config.show_range_slider},
        },
        yaxis={
            "title": y_title,
            "gridcolor": theme.slate_200,
            "zerolinecolor": theme.slate_500,
            "separatethousands": True,
        },
        hoverlabel={"bgcolor": theme.white, "font_color": theme.slate_950},
    )
    if secondary_y_title:
        figure.update_layout(
            yaxis2={
                "title": secondary_y_title,
                "overlaying": "y",
                "side": "right",
                "showgrid": False,
                "rangemode": "tozero",
            }
        )
    return figure


def _add_anomaly_markers(
    figure: go.Figure,
    daily: pd.DataFrame,
    config: VisualizationConfig,
) -> None:
    """Overlay logical-anomaly markers without connecting unrelated dates."""
    if not config.show_anomalies or TOTAL_LOAD_COLUMN not in daily.columns:
        return
    theme = config.theme
    for column, name, symbol, color in (
        (
            TRANSFER_ANOMALY_COLUMN,
            "Transfer > CBP custody",
            "x",
            theme.red,
        ),
        (
            DISCHARGE_ANOMALY_COLUMN,
            "Discharge > HHS care",
            "diamond-open",
            theme.amber,
        ),
    ):
        if column not in daily.columns:
            continue
        mask = daily[column].fillna(False).astype(bool)
        if not mask.any():
            continue
        figure.add_trace(
            go.Scatter(
                x=daily.index[mask],
                y=daily.loc[mask, TOTAL_LOAD_COLUMN],
                mode="markers",
                name=name,
                marker={"color": color, "size": 9, "symbol": symbol},
                hovertemplate=(
                    "%{x|%b %d, %Y}<br>System load: %{y:,.0f}<br>"
                    + f"<b>{name}</b>"
                    + "<extra></extra>"
                ),
            )
        )


def _add_backlog_regions(
    figure: go.Figure,
    daily: pd.DataFrame,
    config: VisualizationConfig,
) -> int:
    """Shade elevated positive-pressure episodes and return their count."""
    episodes = calculate_backlog_episodes(
        daily,
        threshold_days=config.backlog_threshold_days,
    )
    if episodes.empty:
        return 0
    elevated = episodes.loc[episodes["Elevated"].fillna(False)]
    for position, (_, episode) in enumerate(elevated.iterrows()):
        figure.add_vrect(
            x0=episode["Episode Start"],
            x1=pd.Timestamp(episode["Episode End"]) + pd.Timedelta(days=1),
            fillcolor=config.theme.amber,
            opacity=0.10,
            line_width=0,
            layer="below",
            annotation_text="Elevated backlog" if position == 0 else None,
            annotation_position="top left",
        )
    return int(len(elevated))


def _system_load_figure(prepared: VisualizationData) -> go.Figure:
    chart = prepared.chart_metrics
    config = prepared.config
    _require_columns(
        chart,
        (TOTAL_LOAD_COLUMN, ROLLING_7_COLUMN, ROLLING_14_COLUMN),
    )
    theme = config.theme
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=chart.index,
            y=chart[TOTAL_LOAD_COLUMN],
            mode="lines",
            name="Total System Load",
            line={"color": theme.navy, "width": 2.8},
            hovertemplate="%{x|%b %d, %Y}<br>Total load: %{y:,.0f}<extra></extra>",
        )
    )
    for column, name, color, dash in (
        (ROLLING_7_COLUMN, "7-day average", theme.blue, "dot"),
        (ROLLING_14_COLUMN, "14-day average", theme.teal, "dash"),
    ):
        figure.add_trace(
            go.Scatter(
                x=chart.index,
                y=chart[column],
                mode="lines",
                name=name,
                line={"color": color, "width": 2, "dash": dash},
                hovertemplate=f"%{{x|%b %d, %Y}}<br>{name}: %{{y:,.0f}}<extra></extra>",
            )
        )
    _add_anomaly_markers(figure, prepared.daily_metrics, config)
    return apply_government_theme(
        figure,
        title="Total System Care Load",
        y_title="Children under care",
        config=config,
    )


def create_system_load_chart(
    data: pd.DataFrame,
    config: VisualizationConfig | None = None,
) -> go.Figure:
    """Create Total System Load with 7-day and 14-day moving averages."""
    return _system_load_figure(prepare_visualization_data(data, config))


def _care_load_comparison_figure(prepared: VisualizationData) -> go.Figure:
    chart = prepared.chart_metrics
    config = prepared.config
    _require_columns(chart, (CBP_COLUMN, HHS_COLUMN))
    theme = config.theme
    figure = go.Figure()
    stacked = config.comparison_mode == "stacked"
    for column, name, color in (
        (CBP_COLUMN, "CBP custody", theme.blue),
        (HHS_COLUMN, "HHS care", theme.teal),
    ):
        figure.add_trace(
            go.Scatter(
                x=chart.index,
                y=chart[column],
                mode="lines",
                name=name,
                stackgroup="care" if stacked else None,
                groupnorm="" if stacked else None,
                line={"color": color, "width": 2},
                fill="tonexty" if stacked else None,
                hovertemplate=f"%{{x|%b %d, %Y}}<br>{name}: %{{y:,.0f}}<extra></extra>",
            )
        )
    return apply_government_theme(
        figure,
        title="Active CBP and HHS Care Loads",
        y_title="Children in active care",
        config=config,
    )


def create_care_load_comparison_chart(
    data: pd.DataFrame,
    config: VisualizationConfig | None = None,
) -> go.Figure:
    """Create a stacked-area or line comparison of active CBP and HHS loads."""
    return _care_load_comparison_figure(prepare_visualization_data(data, config))


def _net_intake_backlog_figure(prepared: VisualizationData) -> go.Figure:
    chart = prepared.chart_metrics
    config = prepared.config
    _require_columns(chart, (NET_INTAKE_COLUMN, BACKLOG_STREAK_COLUMN))
    theme = config.theme
    values = pd.to_numeric(chart[NET_INTAKE_COLUMN], errors="coerce").fillna(0)
    colors = np.where(values.gt(0), theme.red, theme.green)
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Bar(
            x=chart.index,
            y=values,
            name="Net intake pressure",
            marker={"color": colors, "line": {"width": 0}},
            customdata=np.where(values.gt(0), "Pressure", "Relief"),
            hovertemplate=(
                "%{x|%b %d, %Y}<br>Net intake: %{y:+,.0f}<br>"
                "State: %{customdata}<extra></extra>"
            ),
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=chart.index,
            y=chart[BACKLOG_STREAK_COLUMN],
            mode="lines",
            name="Backlog streak",
            line={"color": theme.purple, "width": 2},
            hovertemplate="%{x|%b %d, %Y}<br>Streak: %{y:,.0f} days<extra></extra>",
        ),
        secondary_y=True,
    )
    _add_backlog_regions(figure, prepared.daily_metrics, config)
    figure.add_hline(y=0, line={"color": theme.slate_500, "width": 1})
    apply_government_theme(
        figure,
        title="Net Intake Pressure and Backlog Accumulation",
        y_title="Net children per period",
        secondary_y_title="Consecutive pressure days",
        config=config,
    )
    figure.update_yaxes(rangemode="tozero", secondary_y=True)
    return figure


def create_net_intake_backlog_chart(
    data: pd.DataFrame,
    config: VisualizationConfig | None = None,
) -> go.Figure:
    """Create pressure bars, backlog streak line, and elevated-period shading."""
    return _net_intake_backlog_figure(prepare_visualization_data(data, config))


def _operational_flows_figure(prepared: VisualizationData) -> go.Figure:
    chart = prepared.chart_metrics
    config = prepared.config
    _require_columns(chart, (INTAKE_COLUMN, TRANSFER_COLUMN, DISCHARGE_COLUMN))
    theme = config.theme
    figure = go.Figure()
    for column, name, color in (
        (INTAKE_COLUMN, "CBP apprehensions", theme.slate_500),
        (TRANSFER_COLUMN, "Transfers to HHS", theme.blue),
        (DISCHARGE_COLUMN, "HHS discharges", theme.green),
    ):
        figure.add_trace(
            go.Bar(
                x=chart.index,
                y=chart[column],
                name=name,
                marker_color=color,
                hovertemplate=f"%{{x|%b %d, %Y}}<br>{name}: %{{y:,.0f}}<extra></extra>",
            )
        )
    figure.update_layout(barmode="group")
    return apply_government_theme(
        figure,
        title="Operational Intake, Transfer, and Discharge Flows",
        y_title="Children per period",
        config=config,
    )


def create_operational_flow_chart(
    data: pd.DataFrame,
    config: VisualizationConfig | None = None,
) -> go.Figure:
    """Create a grouped comparison of the three operational flow measures."""
    return _operational_flows_figure(prepare_visualization_data(data, config))


def _growth_volatility_figure(
    prepared: VisualizationData,
    volatility_window_days: int,
) -> go.Figure:
    if volatility_window_days < 2:
        raise ValueError("volatility_window_days must be at least 2.")
    chart = prepared.chart_metrics
    config = prepared.config
    _require_columns(chart, (GROWTH_RATE_COLUMN,))
    theme = config.theme
    growth = pd.to_numeric(chart[GROWTH_RATE_COLUMN], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    volatility = growth.rolling(volatility_window_days, min_periods=2).std(ddof=0)
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scatter(
            x=chart.index,
            y=growth,
            mode="lines",
            name="Care load growth",
            line={"color": theme.blue, "width": 1.8},
            hovertemplate="%{x|%b %d, %Y}<br>Growth: %{y:+.2f}%<extra></extra>",
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=chart.index,
            y=volatility,
            mode="lines",
            name=f"{volatility_window_days}-period volatility",
            line={"color": theme.purple, "width": 2.2},
            fill="tozeroy",
            fillcolor="rgba(118,86,166,0.10)",
            hovertemplate="%{x|%b %d, %Y}<br>Volatility: %{y:.2f}%<extra></extra>",
        ),
        secondary_y=True,
    )
    figure.add_hline(y=0, line={"color": theme.slate_500, "width": 1})
    apply_government_theme(
        figure,
        title="Care Load Growth and Volatility",
        y_title="Growth rate (%)",
        secondary_y_title="Volatility index (%)",
        config=config,
    )
    figure.update_yaxes(rangemode="tozero", secondary_y=True)
    return figure


def create_growth_volatility_chart(
    data: pd.DataFrame,
    config: VisualizationConfig | None = None,
    *,
    volatility_window_days: int = 30,
) -> go.Figure:
    """Create care-load growth with a rolling volatility overlay."""
    prepared = prepare_visualization_data(data, config)
    return _growth_volatility_figure(prepared, volatility_window_days)


def create_capacity_utilization_chart(
    data: pd.DataFrame,
    *,
    cbp_capacity: int,
    hhs_capacity: int,
    warning_threshold: float = 80.0,
    critical_threshold: float = 95.0,
    config: VisualizationConfig | None = None,
) -> go.Figure:
    """Create scenario utilization lines against user-supplied capacity ceilings."""
    prepared = prepare_visualization_data(data, config)
    try:
        scenario = calculate_capacity_scenario(
            prepared.daily_metrics,
            cbp_capacity=cbp_capacity,
            hhs_capacity=hhs_capacity,
            warning_threshold=warning_threshold,
            critical_threshold=critical_threshold,
        )
    except (AnalyticsError, TypeError, ValueError) as exc:
        raise VisualizationError(f"Unable to calculate capacity scenario: {exc}") from exc

    request = prepared.config
    if TimeGranularity.parse(request.granularity) is TimeGranularity.DAILY:
        chart = scenario.metrics
    else:
        frequency = (
            pd.offsets.Week(weekday=6)
            if TimeGranularity.parse(request.granularity) is TimeGranularity.WEEKLY
            else pd.offsets.MonthEnd()
        )
        chart = scenario.metrics.resample(frequency).last().dropna(how="all")

    theme = request.theme
    figure = go.Figure()
    for column, name, color in (
        ("CBP Capacity Utilization", "CBP utilization", theme.blue),
        ("HHS Capacity Utilization", "HHS utilization", theme.teal),
        ("Total Capacity Utilization", "Combined utilization", theme.navy),
    ):
        figure.add_trace(
            go.Scatter(
                x=chart.index,
                y=chart[column],
                mode="lines",
                name=name,
                line={"color": color, "width": 2.5 if "Combined" in name else 1.8},
                hovertemplate=f"%{{x|%b %d, %Y}}<br>{name}: %{{y:.1f}}%<extra></extra>",
            )
        )
    figure.add_hrect(
        y0=warning_threshold,
        y1=critical_threshold,
        fillcolor=theme.amber,
        opacity=0.08,
        line_width=0,
        layer="below",
    )
    figure.add_hrect(
        y0=critical_threshold,
        y1=max(110, float(chart["Total Capacity Utilization"].max()) * 1.05),
        fillcolor=theme.red,
        opacity=0.08,
        line_width=0,
        layer="below",
    )
    figure.add_hline(
        y=warning_threshold,
        line={"color": theme.amber, "dash": "dash", "width": 1.5},
        annotation_text=f"Warning {warning_threshold:g}%",
    )
    figure.add_hline(
        y=critical_threshold,
        line={"color": theme.red, "dash": "dash", "width": 1.5},
        annotation_text=f"Critical {critical_threshold:g}%",
    )
    return apply_government_theme(
        figure,
        title="Planning Capacity Utilization Scenario",
        y_title="Utilization (%)",
        config=request,
    )


def create_validation_summary_chart(
    result: DatasetValidationResult,
    *,
    theme: GovernmentChartTheme | None = None,
    height: int = 360,
) -> go.Figure:
    """Create a finding-type count chart from a structured validation result."""
    if not isinstance(result, DatasetValidationResult):
        raise TypeError("result must be a DatasetValidationResult.")
    if height < 250:
        raise ValueError("height must be at least 250 pixels.")
    palette = theme or GovernmentChartTheme()
    severities = [
        ValidationSeverity.CRITICAL,
        ValidationSeverity.ERROR,
        ValidationSeverity.WARNING,
        ValidationSeverity.INFO,
    ]
    colors = [palette.red, "#D64545", palette.amber, palette.blue]
    counts = [
        sum(item.severity is severity for item in result.report.findings)
        for severity in severities
    ]
    affected = [
        sum(
            item.affected_rows
            for item in result.report.findings
            if item.severity is severity
        )
        for severity in severities
    ]
    figure = go.Figure(
        go.Bar(
            x=[severity.value.title() for severity in severities],
            y=counts,
            marker_color=colors,
            customdata=affected,
            text=counts,
            textposition="outside",
            hovertemplate=(
                "%{x}<br>Finding types: %{y:,}<br>Reported affected rows: "
                "%{customdata:,}<extra></extra>"
            ),
        )
    )
    config = VisualizationConfig(height=height, theme=palette)
    apply_government_theme(
        figure,
        title=f"Data Quality Findings — {result.report.status}",
        y_title="Finding types",
        config=config,
    )
    figure.update_layout(showlegend=False)
    figure.update_xaxes(title="Severity")
    return figure


def create_dashboard_figures(
    data: pd.DataFrame,
    config: VisualizationConfig | None = None,
    *,
    volatility_window_days: int = 30,
) -> DashboardFigures:
    """Build the five core dashboard figures from one prepared dataset."""
    prepared = prepare_visualization_data(data, config)
    return DashboardFigures(
        system_load=_system_load_figure(prepared),
        care_load_comparison=_care_load_comparison_figure(prepared),
        net_intake_backlog=_net_intake_backlog_figure(prepared),
        operational_flows=_operational_flows_figure(prepared),
        growth_volatility=_growth_volatility_figure(
            prepared,
            volatility_window_days,
        ),
        data=prepared,
    )


def figure_to_html_bytes(
    figure: go.Figure,
    *,
    include_plotlyjs: bool | str = "cdn",
    full_html: bool = True,
) -> bytes:
    """Serialize a Plotly figure to portable UTF-8 HTML bytes."""
    if not isinstance(figure, go.Figure):
        raise TypeError("figure must be a plotly.graph_objects.Figure.")
    return figure.to_html(
        include_plotlyjs=include_plotlyjs,
        full_html=full_html,
        config=PLOTLY_RENDER_CONFIG,
    ).encode("utf-8")
