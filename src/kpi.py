"""KPI business rules for the HHS UAC capacity analytics system.

The calculations in this module are presentation-independent. They enrich the
five core values from :mod:`app_utils` with comparison baselines, directions,
status classifications, trend series, alert messages, and calculation metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from app_utils import (
    BACKLOG_STREAK_COLUMN,
    DATE_COLUMN,
    DISCHARGE_COLUMN,
    GROWTH_RATE_COLUMN,
    NET_INTAKE_COLUMN,
    OFFSET_RATIO_COLUMN,
    TOTAL_LOAD_COLUMN,
    TRANSFER_COLUMN,
    DataValidationError,
    calculate_kpis,
    compute_capacity_metrics,
    validate_and_clean_data,
)


TOTAL_CARE_KEY = "total_children_under_care"
NET_PRESSURE_KEY = "net_intake_pressure"
VOLATILITY_KEY = "care_load_volatility_index"
BACKLOG_KEY = "backlog_accumulation_rate"
OFFSET_KEY = "discharge_offset_ratio"
KPI_KEYS = (
    TOTAL_CARE_KEY,
    NET_PRESSURE_KEY,
    VOLATILITY_KEY,
    BACKLOG_KEY,
    OFFSET_KEY,
)


class KPIError(ValueError):
    """Raised when a KPI request cannot be calculated safely."""


class KPIStatus(str, Enum):
    """Operational classification attached to a KPI value."""

    FAVORABLE = "favorable"
    STABLE = "stable"
    WATCH = "watch"
    CRITICAL = "critical"
    INSUFFICIENT_DATA = "insufficient_data"


class KPIDirection(str, Enum):
    """Direction of a KPI relative to its comparison baseline."""

    INCREASE = "increase"
    DECREASE = "decrease"
    UNCHANGED = "unchanged"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class KPIConfig:
    """Thresholds and comparison settings for KPI classification."""

    comparison_window_days: int = 30
    load_growth_watch_percent: float = 1.0
    load_growth_critical_percent: float = 5.0
    net_intake_watch: float = 25.0
    net_intake_critical: float = 75.0
    volatility_watch_percent: float = 1.0
    volatility_critical_percent: float = 2.0
    backlog_watch_days: int = 3
    backlog_critical_days: int = 7
    offset_watch_ratio: float = 0.90
    offset_critical_ratio: float = 0.75
    trend_volatility_window_days: int = 30

    def __post_init__(self) -> None:
        if self.comparison_window_days < 1:
            raise ValueError("comparison_window_days must be at least 1.")
        if self.trend_volatility_window_days < 2:
            raise ValueError("trend_volatility_window_days must be at least 2.")
        if not 0 <= self.load_growth_watch_percent < self.load_growth_critical_percent:
            raise ValueError("Load thresholds must satisfy 0 <= watch < critical.")
        if not 0 <= self.net_intake_watch < self.net_intake_critical:
            raise ValueError("Net-intake thresholds must satisfy 0 <= watch < critical.")
        if not 0 <= self.volatility_watch_percent < self.volatility_critical_percent:
            raise ValueError("Volatility thresholds must satisfy 0 <= watch < critical.")
        if not 1 <= self.backlog_watch_days < self.backlog_critical_days:
            raise ValueError("Backlog thresholds must satisfy 1 <= watch < critical.")
        if not 0 <= self.offset_critical_ratio < self.offset_watch_ratio <= 1:
            raise ValueError("Offset thresholds must satisfy 0 <= critical < watch <= 1.")


@dataclass(frozen=True)
class KPIResult:
    """One calculated KPI with comparison and classification metadata."""

    key: str
    name: str
    value: int | float
    formatted_value: str
    unit: str
    description: str
    formula: str
    as_of_date: str
    status: KPIStatus
    direction: KPIDirection
    comparison_value: int | float | None
    delta: int | float | None
    delta_percent: float | None
    higher_is_better: bool | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary representation."""
        return {
            "key": self.key,
            "name": self.name,
            "value": self.value,
            "formatted_value": self.formatted_value,
            "unit": self.unit,
            "description": self.description,
            "formula": self.formula,
            "as_of_date": self.as_of_date,
            "status": self.status.value,
            "direction": self.direction.value,
            "comparison_value": self.comparison_value,
            "delta": self.delta,
            "delta_percent": self.delta_percent,
            "higher_is_better": self.higher_is_better,
        }


@dataclass(frozen=True)
class KPIAlert:
    """Operational alert generated from a KPI status."""

    kpi_key: str
    severity: str
    title: str
    message: str
    as_of_date: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-friendly alert dictionary."""
        return {
            "kpi_key": self.kpi_key,
            "severity": self.severity,
            "title": self.title,
            "message": self.message,
            "as_of_date": self.as_of_date,
        }


@dataclass
class KPIDashboardResult:
    """Complete KPI output for dashboards, APIs, and reports."""

    metrics: pd.DataFrame
    trend_frame: pd.DataFrame
    kpis: dict[str, KPIResult]
    alerts: list[KPIAlert]
    summary_table: pd.DataFrame
    config: KPIConfig

    def copy(self) -> KPIDashboardResult:
        """Return a defensive copy for downstream use."""
        return KPIDashboardResult(
            metrics=self.metrics.copy(),
            trend_frame=self.trend_frame.copy(),
            kpis=dict(self.kpis),
            alerts=list(self.alerts),
            summary_table=self.summary_table.copy(),
            config=self.config,
        )


KPI_DEFINITIONS: dict[str, dict[str, str | bool | None]] = {
    TOTAL_CARE_KEY: {
        "name": "Total Children Under Care",
        "unit": "children",
        "description": "Latest active CBP custody plus active HHS care load.",
        "formula": "Children in CBP custody + Children in HHS Care",
        "higher_is_better": False,
    },
    NET_PRESSURE_KEY: {
        "name": "Net Intake Pressure",
        "unit": "children/day",
        "description": "Latest transfers into HHS minus HHS discharges.",
        "formula": "Transfers out of CBP custody - Discharges from HHS Care",
        "higher_is_better": False,
    },
    VOLATILITY_KEY: {
        "name": "Care Load Volatility Index",
        "unit": "%",
        "description": "Population standard deviation of daily care-load growth.",
        "formula": "Population SD of daily Total System Load growth (%)",
        "higher_is_better": False,
    },
    BACKLOG_KEY: {
        "name": "Backlog Accumulation Rate",
        "unit": "consecutive days",
        "description": "Longest selected-period run of positive Net Daily Intake.",
        "formula": "Longest consecutive run where Net Daily Intake > 0",
        "higher_is_better": False,
    },
    OFFSET_KEY: {
        "name": "Discharge Offset Ratio",
        "unit": "ratio",
        "description": "Latest discharges relative to transfers into HHS.",
        "formula": "Discharges from HHS Care / (Transfers out of CBP custody + 1)",
        "higher_is_better": True,
    },
}


def _prepare_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw, cleaned, or derived input into daily analytical metrics."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if data.empty:
        raise KPIError("KPI input data is empty.")

    frame = data.copy()
    derived_columns = {
        TOTAL_LOAD_COLUMN,
        NET_INTAKE_COLUMN,
        GROWTH_RATE_COLUMN,
        OFFSET_RATIO_COLUMN,
        BACKLOG_STREAK_COLUMN,
    }
    if derived_columns.issubset(frame.columns):
        if DATE_COLUMN in frame.columns:
            frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN], errors="coerce")
            if frame[DATE_COLUMN].isna().any():
                raise KPIError("Derived KPI input contains invalid dates.")
            frame = frame.set_index(DATE_COLUMN)
        elif not isinstance(frame.index, pd.DatetimeIndex):
            raise KPIError("Derived KPI input must use a DatetimeIndex or Date column.")
        frame.index = pd.DatetimeIndex(frame.index, name=DATE_COLUMN)
        if frame.index.has_duplicates:
            raise KPIError("Derived KPI input contains duplicate dates.")
        return frame.sort_index()

    raw_frame = frame.reset_index() if DATE_COLUMN not in frame.columns else frame
    try:
        cleaned = validate_and_clean_data(raw_frame)
        metrics = compute_capacity_metrics(cleaned)
    except (DataValidationError, TypeError, ValueError) as exc:
        raise KPIError(f"Unable to prepare KPI metrics: {exc}") from exc
    metrics.index = pd.DatetimeIndex(metrics.index, name=DATE_COLUMN)
    return metrics.sort_index()


def _filter_as_of(
    metrics: pd.DataFrame,
    as_of_date: str | date | datetime | pd.Timestamp | None,
) -> pd.DataFrame:
    """Restrict metrics to information available on or before an as-of date."""
    if as_of_date is None:
        return metrics.copy()
    try:
        as_of = pd.Timestamp(as_of_date)
    except (TypeError, ValueError, OverflowError) as exc:
        raise KPIError(f"Invalid as_of_date: {as_of_date!r}") from exc
    if pd.isna(as_of):
        raise KPIError(f"Invalid as_of_date: {as_of_date!r}")
    if as_of.tzinfo is not None:
        as_of = as_of.tz_localize(None)
    filtered = metrics.loc[: as_of.normalize()].copy()
    if filtered.empty:
        raise KPIError("No observations are available on or before as_of_date.")
    return filtered


def _direction(delta: float | None, tolerance: float = 1e-12) -> KPIDirection:
    """Classify a numeric delta into increase, decrease, or unchanged."""
    if delta is None or not np.isfinite(delta):
        return KPIDirection.UNAVAILABLE
    if delta > tolerance:
        return KPIDirection.INCREASE
    if delta < -tolerance:
        return KPIDirection.DECREASE
    return KPIDirection.UNCHANGED


def _delta_percent(value: float, comparison: float | None) -> float | None:
    """Calculate comparison-relative percentage change safely."""
    if comparison is None or not np.isfinite(comparison) or comparison == 0:
        return None
    result = (value - comparison) / abs(comparison) * 100
    return float(result) if np.isfinite(result) else None


def _longest_positive_streak(values: pd.Series) -> int:
    """Return the longest consecutive run of positive numeric values."""
    positive = pd.to_numeric(values, errors="coerce").fillna(0).gt(0)
    groups = positive.ne(positive.shift(fill_value=False)).cumsum()
    streaks = positive.groupby(groups).cumsum()
    return int(streaks.max()) if not streaks.empty else 0


def _comparison_baselines(
    metrics: pd.DataFrame,
    config: KPIConfig,
) -> dict[str, float | None]:
    """Calculate prior-window baselines for all five KPIs."""
    if len(metrics) < 2:
        return {key: None for key in KPI_KEYS}

    window = min(config.comparison_window_days, max(1, len(metrics) // 2))
    previous = metrics.iloc[-2 * window : -window]
    if previous.empty:
        return {key: None for key in KPI_KEYS}

    growth = (
        pd.to_numeric(previous[GROWTH_RATE_COLUMN], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    return {
        TOTAL_CARE_KEY: float(previous[TOTAL_LOAD_COLUMN].mean()),
        NET_PRESSURE_KEY: float(previous[NET_INTAKE_COLUMN].mean()),
        VOLATILITY_KEY: float(growth.std(ddof=0)) if not growth.empty else 0.0,
        BACKLOG_KEY: float(_longest_positive_streak(previous[NET_INTAKE_COLUMN])),
        OFFSET_KEY: float(previous[DISCHARGE_COLUMN].sum() / (previous[TRANSFER_COLUMN].sum() + 1)),
    }


def _classify_status(
    key: str,
    value: float,
    comparison: float | None,
    config: KPIConfig,
) -> KPIStatus:
    """Apply KPI-specific operational status thresholds."""
    if not np.isfinite(value):
        return KPIStatus.INSUFFICIENT_DATA

    if key == TOTAL_CARE_KEY:
        change = _delta_percent(value, comparison)
        if change is None:
            return KPIStatus.STABLE
        if change >= config.load_growth_critical_percent:
            return KPIStatus.CRITICAL
        if change >= config.load_growth_watch_percent:
            return KPIStatus.WATCH
        if change <= -config.load_growth_watch_percent:
            return KPIStatus.FAVORABLE
        return KPIStatus.STABLE

    if key == NET_PRESSURE_KEY:
        if value >= config.net_intake_critical:
            return KPIStatus.CRITICAL
        if value >= config.net_intake_watch:
            return KPIStatus.WATCH
        if value <= 0:
            return KPIStatus.FAVORABLE
        return KPIStatus.STABLE

    if key == VOLATILITY_KEY:
        if value >= config.volatility_critical_percent:
            return KPIStatus.CRITICAL
        if value >= config.volatility_watch_percent:
            return KPIStatus.WATCH
        return KPIStatus.STABLE

    if key == BACKLOG_KEY:
        if value >= config.backlog_critical_days:
            return KPIStatus.CRITICAL
        if value >= config.backlog_watch_days:
            return KPIStatus.WATCH
        if value == 0:
            return KPIStatus.FAVORABLE
        return KPIStatus.STABLE

    if key == OFFSET_KEY:
        if value < config.offset_critical_ratio:
            return KPIStatus.CRITICAL
        if value < config.offset_watch_ratio:
            return KPIStatus.WATCH
        if value >= 1:
            return KPIStatus.FAVORABLE
        return KPIStatus.STABLE

    raise KPIError(f"Unknown KPI key: {key}")


def _formatted_value(key: str, value: float) -> str:
    """Format one KPI value consistently for human-readable output."""
    if key in {TOTAL_CARE_KEY, NET_PRESSURE_KEY}:
        sign = "+" if key == NET_PRESSURE_KEY else ""
        return f"{value:{sign},.0f}"
    if key == VOLATILITY_KEY:
        return f"{value:.2f}%"
    if key == BACKLOG_KEY:
        return f"{value:,.0f} day(s)"
    if key == OFFSET_KEY:
        return f"{value:.1%}"
    return str(value)


def calculate_kpi_trends(
    metrics: pd.DataFrame,
    volatility_window_days: int = 30,
) -> pd.DataFrame:
    """Create a daily time series for the five KPI concepts."""
    if volatility_window_days < 2:
        raise ValueError("volatility_window_days must be at least 2.")
    frame = _prepare_metrics(metrics)
    trend = pd.DataFrame(index=frame.index)
    trend[TOTAL_CARE_KEY] = pd.to_numeric(frame[TOTAL_LOAD_COLUMN], errors="coerce")
    trend[NET_PRESSURE_KEY] = pd.to_numeric(frame[NET_INTAKE_COLUMN], errors="coerce")
    growth = pd.to_numeric(frame[GROWTH_RATE_COLUMN], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    trend[VOLATILITY_KEY] = growth.rolling(
        volatility_window_days,
        min_periods=2,
    ).std(ddof=0)
    trend[BACKLOG_KEY] = pd.to_numeric(frame[BACKLOG_STREAK_COLUMN], errors="coerce").fillna(0)
    trend[OFFSET_KEY] = pd.to_numeric(frame[OFFSET_RATIO_COLUMN], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    trend.index = pd.DatetimeIndex(trend.index, name=DATE_COLUMN)
    trend.attrs.clear()
    return trend


def _build_alerts(kpis: dict[str, KPIResult]) -> list[KPIAlert]:
    """Generate concise alerts for watch and critical KPI statuses."""
    alerts: list[KPIAlert] = []
    messages = {
        TOTAL_CARE_KEY: "System load is rising relative to its prior comparison window.",
        NET_PRESSURE_KEY: "Transfers are materially exceeding HHS discharges.",
        VOLATILITY_KEY: "Daily care-load changes are unusually variable.",
        BACKLOG_KEY: "A sustained positive-net-intake episode requires review.",
        OFFSET_KEY: "Discharges are not sufficiently offsetting transfers into HHS.",
    }
    for key, result in kpis.items():
        if result.status not in {KPIStatus.WATCH, KPIStatus.CRITICAL}:
            continue
        alerts.append(
            KPIAlert(
                kpi_key=key,
                severity=result.status.value,
                title=f"{result.status.value.title()}: {result.name}",
                message=f"{messages[key]} Current value: {result.formatted_value}.",
                as_of_date=result.as_of_date,
            )
        )
    return alerts


def _summary_table(kpis: dict[str, KPIResult]) -> pd.DataFrame:
    """Build a display-ready tabular KPI summary."""
    rows = [
        {
            "KPI": result.name,
            "Value": result.formatted_value,
            "Status": result.status.value,
            "Direction": result.direction.value,
            "Comparison Value": result.comparison_value,
            "Delta": result.delta,
            "Delta Percent": result.delta_percent,
            "Unit": result.unit,
            "As Of": result.as_of_date,
            "Formula": result.formula,
        }
        for result in kpis.values()
    ]
    table = pd.DataFrame.from_records(rows)
    table.attrs.clear()
    return table


class CapacityKPICalculator:
    """Reusable calculator for classified capacity KPI snapshots and trends."""

    def __init__(self, config: KPIConfig | None = None) -> None:
        self.config = config or KPIConfig()

    def calculate(
        self,
        data: pd.DataFrame,
        *,
        as_of_date: str | date | datetime | pd.Timestamp | None = None,
    ) -> KPIDashboardResult:
        """Calculate KPI values, trends, comparisons, statuses, and alerts."""
        metrics = _filter_as_of(_prepare_metrics(data), as_of_date)
        try:
            core_values = calculate_kpis(metrics)
        except (DataValidationError, TypeError, ValueError) as exc:
            raise KPIError(f"Unable to calculate core KPI values: {exc}") from exc

        comparisons = _comparison_baselines(metrics, self.config)
        as_of = metrics.index.max().date().isoformat()
        kpi_results: dict[str, KPIResult] = {}
        for key in KPI_KEYS:
            raw_value = core_values[key]
            value = float(raw_value)
            comparison = comparisons[key]
            delta = (
                value - comparison if comparison is not None and np.isfinite(comparison) else None
            )
            definition = KPI_DEFINITIONS[key]
            result = KPIResult(
                key=key,
                name=str(definition["name"]),
                value=int(value)
                if key in {TOTAL_CARE_KEY, NET_PRESSURE_KEY, BACKLOG_KEY}
                else value,
                formatted_value=_formatted_value(key, value),
                unit=str(definition["unit"]),
                description=str(definition["description"]),
                formula=str(definition["formula"]),
                as_of_date=as_of,
                status=_classify_status(
                    key,
                    value,
                    comparison,
                    self.config,
                ),
                direction=_direction(delta),
                comparison_value=comparison,
                delta=delta,
                delta_percent=_delta_percent(value, comparison),
                higher_is_better=definition["higher_is_better"],  # type: ignore[arg-type]
            )
            kpi_results[key] = result

        trend_frame = calculate_kpi_trends(
            metrics,
            self.config.trend_volatility_window_days,
        )
        alerts = _build_alerts(kpi_results)
        return KPIDashboardResult(
            metrics=metrics,
            trend_frame=trend_frame,
            kpis=kpi_results,
            alerts=alerts,
            summary_table=_summary_table(kpi_results),
            config=self.config,
        )


def calculate_kpi_dashboard(
    data: pd.DataFrame,
    config: KPIConfig | None = None,
    *,
    as_of_date: str | date | datetime | pd.Timestamp | None = None,
) -> KPIDashboardResult:
    """Functional entry point for the complete KPI business-rules pipeline."""
    return CapacityKPICalculator(config).calculate(data, as_of_date=as_of_date)


def kpi_results_to_dict(result: KPIDashboardResult) -> dict[str, Any]:
    """Serialize a KPI dashboard result without embedding large trend data."""
    if not isinstance(result, KPIDashboardResult):
        raise TypeError("result must be a KPIDashboardResult.")
    return {
        "as_of_date": result.metrics.index.max().date().isoformat(),
        "observation_count": int(len(result.metrics)),
        "kpis": {key: value.to_dict() for key, value in result.kpis.items()},
        "alerts": [alert.to_dict() for alert in result.alerts],
        "config": {
            "comparison_window_days": result.config.comparison_window_days,
            "trend_volatility_window_days": (result.config.trend_volatility_window_days),
        },
    }
