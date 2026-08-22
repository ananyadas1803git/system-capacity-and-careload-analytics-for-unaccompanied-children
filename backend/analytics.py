"""Core analytics orchestration for the HHS UAC capacity system.

This module sits between data preparation in :mod:`app_utils` and presentation
layers such as Streamlit or an API. It provides deterministic, UI-independent
analysis services that can also be used in tests, batch jobs, and notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import BinaryIO, TextIO

import numpy as np
import pandas as pd

from app_utils import (
    BACKLOG_STREAK_COLUMN,
    CBP_COLUMN,
    DISCHARGE_ANOMALY_COLUMN,
    DISCHARGE_COLUMN,
    GROWTH_RATE_COLUMN,
    HHS_COLUMN,
    INTAKE_COLUMN,
    NET_INTAKE_COLUMN,
    OFFSET_RATIO_COLUMN,
    QUALITY_FLAG_COLUMN,
    ROLLING_14_COLUMN,
    ROLLING_7_COLUMN,
    TOTAL_LOAD_COLUMN,
    TRANSFER_ANOMALY_COLUMN,
    TRANSFER_COLUMN,
    DataValidationError,
    ValidationReport,
    calculate_kpis,
    compute_capacity_metrics,
    read_csv_data,
    validate_and_clean_data,
)


class AnalyticsError(RuntimeError):
    """Raised when a valid analytical request cannot be completed."""


class TimeGranularity(str, Enum):
    """Supported chart and reporting aggregation levels."""

    DAILY = "Daily"
    WEEKLY = "Weekly"
    MONTHLY = "Monthly"

    @classmethod
    def parse(cls, value: str | TimeGranularity) -> TimeGranularity:
        """Normalize a string or enum into a supported granularity."""
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().casefold()
        for member in cls:
            if member.value.casefold() == normalized:
                return member
        valid = ", ".join(member.value for member in cls)
        raise ValueError(f"Unsupported granularity '{value}'. Expected one of: {valid}.")


@dataclass(frozen=True)
class AnalysisConfig:
    """Configuration for a single capacity-analysis request.

    Attributes:
        start_date: Inclusive reporting-period start. ``None`` uses the first
            available date.
        end_date: Inclusive reporting-period end. ``None`` uses the last
            available date.
        granularity: Daily, weekly, or monthly presentation aggregation.
        backlog_threshold_days: Minimum positive-intake streak duration used to
            classify an elevated backlog episode.
    """

    start_date: str | date | datetime | pd.Timestamp | None = None
    end_date: str | date | datetime | pd.Timestamp | None = None
    granularity: str | TimeGranularity = TimeGranularity.DAILY
    backlog_threshold_days: int = 3

    def __post_init__(self) -> None:
        if self.backlog_threshold_days < 1:
            raise ValueError("backlog_threshold_days must be at least 1.")


@dataclass
class AnalysisResult:
    """Complete output of the analytical pipeline."""

    daily_metrics: pd.DataFrame
    chart_metrics: pd.DataFrame
    kpis: dict[str, int | float]
    backlog_episodes: pd.DataFrame
    anomaly_rows: pd.DataFrame
    operational_summary: dict[str, int | float | str]
    validation_report: ValidationReport | None
    config: AnalysisConfig

    def copy(self) -> AnalysisResult:
        """Return a defensive copy suitable for downstream mutation."""
        return AnalysisResult(
            daily_metrics=self.daily_metrics.copy(),
            chart_metrics=self.chart_metrics.copy(),
            kpis=dict(self.kpis),
            backlog_episodes=self.backlog_episodes.copy(),
            anomaly_rows=self.anomaly_rows.copy(),
            operational_summary=dict(self.operational_summary),
            validation_report=self.validation_report,
            config=self.config,
        )


@dataclass
class CapacityScenarioResult:
    """Output of a user-defined capacity-planning scenario."""

    metrics: pd.DataFrame
    summary: dict[str, int | float | str]
    stress_episodes: pd.DataFrame


def _to_timestamp(
    value: str | date | datetime | pd.Timestamp | None,
    field_name: str,
) -> pd.Timestamp | None:
    """Convert a date-like input into a normalized pandas timestamp."""
    if value is None:
        return None
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} is not a valid date: {value!r}") from exc
    if pd.isna(timestamp):
        raise ValueError(f"{field_name} is not a valid date: {value!r}")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp.normalize()


def _ensure_datetime_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted copy with a unique DatetimeIndex named ``Date``."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")
    if frame.empty:
        raise AnalyticsError("No data is available for analysis.")

    result = frame.copy()
    if "Date" in result.columns:
        result["Date"] = pd.to_datetime(result["Date"], errors="coerce")
        if result["Date"].isna().any():
            raise AnalyticsError("One or more analytical rows have an invalid Date.")
        result = result.set_index("Date")
    elif not isinstance(result.index, pd.DatetimeIndex):
        raise AnalyticsError("Analytical data must use Date as a column or DatetimeIndex.")

    result.index = pd.DatetimeIndex(result.index, name="Date")
    if result.index.has_duplicates:
        raise AnalyticsError("Analytical data contains duplicate reporting dates.")
    return result.sort_index()


def filter_reporting_period(
    frame: pd.DataFrame,
    start_date: str | date | datetime | pd.Timestamp | None = None,
    end_date: str | date | datetime | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Filter an analytical DataFrame to an inclusive reporting period."""
    indexed = _ensure_datetime_index(frame)
    start = _to_timestamp(start_date, "start_date") or indexed.index.min()
    end = _to_timestamp(end_date, "end_date") or indexed.index.max()

    if start > end:
        raise ValueError("start_date must be on or before end_date.")
    filtered = indexed.loc[start:end].copy()
    if filtered.empty:
        raise AnalyticsError(f"No observations exist between {start.date()} and {end.date()}.")
    return filtered


def resample_metrics(
    daily_metrics: pd.DataFrame,
    granularity: str | TimeGranularity,
) -> pd.DataFrame:
    """Aggregate daily analytics for presentation.

    Daily operational flows are summed. Active CBP/HHS loads are stocks, so the
    final observation represents each week or month. Growth is recalculated from
    aggregated endpoint loads. The 7-day and 14-day averages remain daily
    statistics sampled at the period endpoint.
    """
    frame = _ensure_datetime_index(daily_metrics)
    level = TimeGranularity.parse(granularity)
    if level is TimeGranularity.DAILY:
        return frame

    frequency: pd.DateOffset = (
        pd.offsets.Week(weekday=6) if level is TimeGranularity.WEEKLY else pd.offsets.MonthEnd()
    )

    stock_columns = [
        CBP_COLUMN,
        HHS_COLUMN,
        TOTAL_LOAD_COLUMN,
        ROLLING_7_COLUMN,
        ROLLING_14_COLUMN,
        BACKLOG_STREAK_COLUMN,
    ]
    flow_columns = [
        INTAKE_COLUMN,
        TRANSFER_COLUMN,
        DISCHARGE_COLUMN,
        NET_INTAKE_COLUMN,
    ]
    aggregations: dict[str, str] = {
        **{column: "last" for column in stock_columns if column in frame.columns},
        **{column: "sum" for column in flow_columns if column in frame.columns},
    }

    for boolean_column in (
        TRANSFER_ANOMALY_COLUMN,
        DISCHARGE_ANOMALY_COLUMN,
        "Is Imputed Date",
        "Is_Imputed_Date",
    ):
        if boolean_column in frame.columns:
            aggregations[boolean_column] = "max"

    aggregated = frame.resample(frequency).agg(aggregations).dropna(how="all")
    if aggregated.empty:
        raise AnalyticsError("Aggregation produced no reporting periods.")

    required_loads = {CBP_COLUMN, HHS_COLUMN}
    if required_loads.issubset(aggregated.columns):
        aggregated[TOTAL_LOAD_COLUMN] = aggregated[CBP_COLUMN] + aggregated[HHS_COLUMN]
    aggregated[NET_INTAKE_COLUMN] = aggregated[TRANSFER_COLUMN] - aggregated[DISCHARGE_COLUMN]
    aggregated[GROWTH_RATE_COLUMN] = (
        aggregated[TOTAL_LOAD_COLUMN]
        .pct_change(fill_method=None)
        .replace([np.inf, -np.inf], np.nan)
        .mul(100)
    )
    aggregated[OFFSET_RATIO_COLUMN] = aggregated[DISCHARGE_COLUMN].div(
        aggregated[TRANSFER_COLUMN] + 1
    )
    return aggregated


def calculate_backlog_episodes(
    daily_metrics: pd.DataFrame,
    threshold_days: int = 3,
) -> pd.DataFrame:
    """Summarize every continuous period with positive Net Daily Intake."""
    if threshold_days < 1:
        raise ValueError("threshold_days must be at least 1.")

    frame = _ensure_datetime_index(daily_metrics)
    if NET_INTAKE_COLUMN not in frame.columns:
        raise AnalyticsError(f"Missing required metric: {NET_INTAKE_COLUMN}.")

    pressure = pd.to_numeric(frame[NET_INTAKE_COLUMN], errors="coerce").fillna(0)
    positive = pressure.gt(0)
    groups = positive.ne(positive.shift(fill_value=False)).cumsum()
    positive_rows = pd.DataFrame({NET_INTAKE_COLUMN: pressure, "Episode Group": groups}).loc[
        positive
    ]

    columns = [
        "Episode Start",
        "Episode End",
        "Duration (Days)",
        "Cumulative Net Intake",
        "Peak Daily Pressure",
        "Average Daily Pressure",
        "Elevated",
        "Status",
    ]
    if positive_rows.empty:
        return pd.DataFrame(columns=columns)

    latest_date = frame.index.max()
    records: list[dict[str, object]] = []
    for _, episode in positive_rows.groupby("Episode Group"):
        start = episode.index.min()
        end = episode.index.max()
        values = episode[NET_INTAKE_COLUMN]
        duration = int(len(episode))
        records.append(
            {
                "Episode Start": start,
                "Episode End": end,
                "Duration (Days)": duration,
                "Cumulative Net Intake": int(values.sum()),
                "Peak Daily Pressure": int(values.max()),
                "Average Daily Pressure": float(values.mean()),
                "Elevated": duration >= threshold_days,
                "Status": "Active" if end == latest_date else "Closed",
            }
        )

    result = pd.DataFrame.from_records(records, columns=columns).sort_values(
        ["Duration (Days)", "Cumulative Net Intake"],
        ascending=[False, False],
    )
    result.attrs.clear()
    return result.reset_index(drop=True)


def extract_anomaly_rows(daily_metrics: pd.DataFrame) -> pd.DataFrame:
    """Return rows violating the transfer or discharge logical constraints."""
    frame = _ensure_datetime_index(daily_metrics)
    anomaly_columns = [
        TRANSFER_ANOMALY_COLUMN,
        DISCHARGE_ANOMALY_COLUMN,
    ]
    missing = [column for column in anomaly_columns if column not in frame.columns]
    if missing:
        raise AnalyticsError("Missing anomaly field(s): " + ", ".join(missing))

    mask = frame[anomaly_columns].fillna(False).astype(bool).any(axis=1)
    output_columns = [
        CBP_COLUMN,
        TRANSFER_COLUMN,
        HHS_COLUMN,
        DISCHARGE_COLUMN,
        *anomaly_columns,
    ]
    if QUALITY_FLAG_COLUMN in frame.columns:
        output_columns.append(QUALITY_FLAG_COLUMN)
    result = frame.loc[mask, output_columns].copy()
    result.attrs.clear()
    return result


def build_operational_summary(
    daily_metrics: pd.DataFrame,
    kpis: dict[str, int | float],
    backlog_episodes: pd.DataFrame,
    anomaly_rows: pd.DataFrame,
    backlog_threshold_days: int,
) -> dict[str, int | float | str]:
    """Build a JSON-friendly selected-period operational summary."""
    frame = _ensure_datetime_index(daily_metrics)
    imputed_column = next(
        (column for column in ("Is Imputed Date", "Is_Imputed_Date") if column in frame.columns),
        None,
    )
    imputed_dates = (
        int(frame[imputed_column].fillna(False).astype(bool).sum()) if imputed_column else 0
    )
    peak_date = frame[TOTAL_LOAD_COLUMN].idxmax()
    elevated_episodes = (
        int(backlog_episodes["Elevated"].fillna(False).sum()) if not backlog_episodes.empty else 0
    )
    current_streak = int(frame[BACKLOG_STREAK_COLUMN].iloc[-1])

    summary: dict[str, int | float | str] = {
        "period_start": frame.index.min().date().isoformat(),
        "period_end": frame.index.max().date().isoformat(),
        "daily_observations": int(len(frame)),
        "latest_system_load": int(kpis["total_children_under_care"]),
        "average_system_load": float(frame[TOTAL_LOAD_COLUMN].mean()),
        "peak_system_load": int(frame[TOTAL_LOAD_COLUMN].max()),
        "peak_system_load_date": peak_date.date().isoformat(),
        "latest_net_intake": int(kpis["net_intake_pressure"]),
        "cumulative_net_intake": int(frame[NET_INTAKE_COLUMN].sum()),
        "positive_pressure_days": int(frame[NET_INTAKE_COLUMN].gt(0).sum()),
        "care_load_volatility_index": float(kpis["care_load_volatility_index"]),
        "longest_backlog_streak": int(kpis["backlog_accumulation_rate"]),
        "current_backlog_streak": current_streak,
        "backlog_threshold_days": int(backlog_threshold_days),
        "elevated_backlog_episodes": elevated_episodes,
        "latest_discharge_offset_ratio": float(kpis["discharge_offset_ratio"]),
        "total_transfers": int(frame[TRANSFER_COLUMN].sum()),
        "total_discharges": int(frame[DISCHARGE_COLUMN].sum()),
        "logical_anomaly_rows": int(len(anomaly_rows)),
        "imputed_dates": imputed_dates,
    }
    return summary


def _build_stress_episodes(
    scenario_metrics: pd.DataFrame,
    warning_threshold: float,
) -> pd.DataFrame:
    """Summarize continuous combined-utilization warning periods."""
    utilization_column = "Total Capacity Utilization"
    headroom_column = "Total Capacity Headroom"
    above = scenario_metrics[utilization_column].ge(warning_threshold)
    groups = above.ne(above.shift(fill_value=False)).cumsum()
    stressed = scenario_metrics.loc[above].copy()
    stressed["Episode Group"] = groups.loc[above]

    columns = [
        "Episode Start",
        "Episode End",
        "Duration (Days)",
        "Peak Utilization (%)",
        "Minimum Headroom",
        "Peak System Load",
        "Status",
    ]
    if stressed.empty:
        return pd.DataFrame(columns=columns)

    latest_date = scenario_metrics.index.max()
    records: list[dict[str, object]] = []
    for _, episode in stressed.groupby("Episode Group"):
        start = episode.index.min()
        end = episode.index.max()
        records.append(
            {
                "Episode Start": start,
                "Episode End": end,
                "Duration (Days)": int(len(episode)),
                "Peak Utilization (%)": float(episode[utilization_column].max()),
                "Minimum Headroom": int(episode[headroom_column].min()),
                "Peak System Load": int(episode[TOTAL_LOAD_COLUMN].max()),
                "Status": "Active" if end == latest_date else "Closed",
            }
        )
    result = pd.DataFrame.from_records(records, columns=columns).sort_values(
        ["Peak Utilization (%)", "Duration (Days)"],
        ascending=[False, False],
    )
    result.attrs.clear()
    return result.reset_index(drop=True)


def calculate_capacity_scenario(
    daily_metrics: pd.DataFrame,
    cbp_capacity: int,
    hhs_capacity: int,
    warning_threshold: float = 80.0,
    critical_threshold: float = 95.0,
) -> CapacityScenarioResult:
    """Evaluate active loads against user-supplied planning capacities.

    Capacity inputs are scenario assumptions, not inferred official values.
    Utilization percentages may exceed 100 when active load is above the selected
    planning ceiling.
    """
    if cbp_capacity <= 0 or hhs_capacity <= 0:
        raise ValueError("cbp_capacity and hhs_capacity must be positive integers.")
    if not 0 < warning_threshold < critical_threshold:
        raise ValueError("Thresholds must satisfy 0 < warning_threshold < critical_threshold.")

    frame = _ensure_datetime_index(daily_metrics)
    required = [CBP_COLUMN, HHS_COLUMN, TOTAL_LOAD_COLUMN]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise AnalyticsError("Missing capacity field(s): " + ", ".join(missing))

    result = frame.copy()
    total_capacity = cbp_capacity + hhs_capacity
    result["CBP Capacity Utilization"] = result[CBP_COLUMN].div(cbp_capacity).mul(100)
    result["HHS Capacity Utilization"] = result[HHS_COLUMN].div(hhs_capacity).mul(100)
    result["Total Capacity Utilization"] = result[TOTAL_LOAD_COLUMN].div(total_capacity).mul(100)
    result["CBP Capacity Headroom"] = cbp_capacity - result[CBP_COLUMN]
    result["HHS Capacity Headroom"] = hhs_capacity - result[HHS_COLUMN]
    result["Total Capacity Headroom"] = total_capacity - result[TOTAL_LOAD_COLUMN]
    result["Capacity Status"] = np.select(
        [
            result["Total Capacity Utilization"].ge(critical_threshold),
            result["Total Capacity Utilization"].ge(warning_threshold),
        ],
        ["Critical", "Warning"],
        default="Within planning range",
    )

    stress_episodes = _build_stress_episodes(result, warning_threshold)
    latest = result.iloc[-1]
    peak_date = result["Total Capacity Utilization"].idxmax()
    summary: dict[str, int | float | str] = {
        "cbp_planning_capacity": int(cbp_capacity),
        "hhs_planning_capacity": int(hhs_capacity),
        "combined_planning_capacity": int(total_capacity),
        "warning_threshold_percent": float(warning_threshold),
        "critical_threshold_percent": float(critical_threshold),
        "latest_total_utilization_percent": float(latest["Total Capacity Utilization"]),
        "latest_total_headroom": int(latest["Total Capacity Headroom"]),
        "latest_capacity_status": str(latest["Capacity Status"]),
        "peak_total_utilization_percent": float(result["Total Capacity Utilization"].max()),
        "peak_utilization_date": peak_date.date().isoformat(),
        "warning_days": int(result["Total Capacity Utilization"].ge(warning_threshold).sum()),
        "critical_days": int(result["Total Capacity Utilization"].ge(critical_threshold).sum()),
        "stress_episodes": int(len(stress_episodes)),
    }
    return CapacityScenarioResult(
        metrics=result,
        summary=summary,
        stress_episodes=stress_episodes,
    )


class CapacityAnalyticsEngine:
    """Reusable service object for the complete capacity-analysis workflow."""

    def __init__(self, raw_data: pd.DataFrame) -> None:
        """Validate and prepare raw HHS-style input data once."""
        if not isinstance(raw_data, pd.DataFrame):
            raise TypeError("raw_data must be a pandas DataFrame.")
        try:
            prepared = validate_and_clean_data(raw_data)
        except (DataValidationError, TypeError, ValueError) as exc:
            raise AnalyticsError(f"Unable to prepare source data: {exc}") from exc

        self._prepared_data = _ensure_datetime_index(prepared)
        report = prepared.attrs.get("validation_report")
        self._validation_report = report if isinstance(report, ValidationReport) else None

    @classmethod
    def from_csv(
        cls,
        source: str | Path | bytes | BinaryIO | TextIO,
    ) -> CapacityAnalyticsEngine:
        """Construct an engine from a path, bytes, or file-like CSV source."""
        try:
            raw_data = read_csv_data(source)
        except (DataValidationError, TypeError, ValueError) as exc:
            raise AnalyticsError(f"Unable to read source CSV: {exc}") from exc
        return cls(raw_data)

    @property
    def prepared_data(self) -> pd.DataFrame:
        """Return a defensive copy of validated daily source data."""
        return self._prepared_data.copy()

    @property
    def validation_report(self) -> ValidationReport | None:
        """Return the structured validation report, if available."""
        return self._validation_report

    def run(self, config: AnalysisConfig | None = None) -> AnalysisResult:
        """Execute filtering, metrics, KPI, backlog, and anomaly calculations."""
        request = config or AnalysisConfig()
        try:
            filtered = filter_reporting_period(
                self._prepared_data,
                request.start_date,
                request.end_date,
            )
            daily_metrics = compute_capacity_metrics(filtered)
            daily_metrics = _ensure_datetime_index(daily_metrics)
            kpis = calculate_kpis(daily_metrics)
            episodes = calculate_backlog_episodes(
                daily_metrics,
                request.backlog_threshold_days,
            )
            anomalies = extract_anomaly_rows(daily_metrics)
            chart_metrics = resample_metrics(daily_metrics, request.granularity)
            summary = build_operational_summary(
                daily_metrics,
                kpis,
                episodes,
                anomalies,
                request.backlog_threshold_days,
            )
        except (AnalyticsError, DataValidationError, TypeError, ValueError) as exc:
            raise AnalyticsError(f"Capacity analysis failed: {exc}") from exc

        return AnalysisResult(
            daily_metrics=daily_metrics,
            chart_metrics=chart_metrics,
            kpis=kpis,
            backlog_episodes=episodes,
            anomaly_rows=anomalies,
            operational_summary=summary,
            validation_report=self._validation_report,
            config=request,
        )


def run_capacity_analysis(
    raw_data: pd.DataFrame,
    *,
    start_date: str | date | datetime | pd.Timestamp | None = None,
    end_date: str | date | datetime | pd.Timestamp | None = None,
    granularity: str | TimeGranularity = TimeGranularity.DAILY,
    backlog_threshold_days: int = 3,
) -> AnalysisResult:
    """Functional entry point for the complete capacity-analysis pipeline."""
    config = AnalysisConfig(
        start_date=start_date,
        end_date=end_date,
        granularity=granularity,
        backlog_threshold_days=backlog_threshold_days,
    )
    return CapacityAnalyticsEngine(raw_data).run(config)
