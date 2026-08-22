"""Feature engineering for HHS UAC capacity forecasting and risk models.

The pipeline is deterministic and deliberately separates feature columns from
future target columns. Historical rolling features are shifted by one day by
default so the value being summarized cannot leak into its own predictors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app_utils import (
    CBP_COLUMN,
    DATE_COLUMN,
    DISCHARGE_ANOMALY_COLUMN,
    DISCHARGE_COLUMN,
    GROWTH_RATE_COLUMN,
    HHS_COLUMN,
    INTAKE_COLUMN,
    NET_INTAKE_COLUMN,
    TOTAL_LOAD_COLUMN,
    TRANSFER_ANOMALY_COLUMN,
    TRANSFER_COLUMN,
    DataValidationError,
    compute_capacity_metrics,
    validate_and_clean_data,
)


BASE_SIGNAL_COLUMNS = (
    TOTAL_LOAD_COLUMN,
    NET_INTAKE_COLUMN,
    INTAKE_COLUMN,
    CBP_COLUMN,
    HHS_COLUMN,
    TRANSFER_COLUMN,
    DISCHARGE_COLUMN,
    GROWTH_RATE_COLUMN,
)

SIGNAL_NAMES = {
    TOTAL_LOAD_COLUMN: "total_system_load",
    NET_INTAKE_COLUMN: "net_daily_intake",
    INTAKE_COLUMN: "cbp_apprehensions",
    CBP_COLUMN: "cbp_active_load",
    HHS_COLUMN: "hhs_active_load",
    TRANSFER_COLUMN: "cbp_transfers",
    DISCHARGE_COLUMN: "hhs_discharges",
    GROWTH_RATE_COLUMN: "care_load_growth_rate",
}


class FeatureEngineeringError(ValueError):
    """Raised when an ML feature matrix cannot be constructed safely."""


@dataclass(frozen=True)
class FeatureEngineeringConfig:
    """Configuration for deterministic feature and target generation.

    Attributes:
        lag_days: Historical day offsets created for each base signal.
        rolling_windows: Windows used for historical mean, standard deviation,
            minimum, maximum, and slope features.
        ema_spans: Exponential moving-average spans for system load and net
            intake.
        target_horizon_days: Forecast horizon for future target columns.
        leakage_safe_rolling: Shift rolling/EMA inputs by one day before
            calculation.
        include_targets: Add future supervised-learning targets.
        drop_incomplete_rows: Remove rows with missing feature or target values.
    """

    lag_days: tuple[int, ...] = (1, 7, 14, 28)
    rolling_windows: tuple[int, ...] = (7, 14, 30)
    ema_spans: tuple[int, ...] = (7, 14, 30)
    target_horizon_days: int = 7
    leakage_safe_rolling: bool = True
    include_targets: bool = True
    drop_incomplete_rows: bool = False

    def __post_init__(self) -> None:
        for field_name, values in (
            ("lag_days", self.lag_days),
            ("rolling_windows", self.rolling_windows),
            ("ema_spans", self.ema_spans),
        ):
            if not values:
                raise ValueError(f"{field_name} must contain at least one value.")
            if any(not isinstance(value, int) or value < 1 for value in values):
                raise ValueError(f"Every {field_name} value must be a positive integer.")
            if len(set(values)) != len(values):
                raise ValueError(f"{field_name} values must be unique.")
        if self.target_horizon_days < 1:
            raise ValueError("target_horizon_days must be at least 1.")


@dataclass
class FeatureEngineeringResult:
    """Feature pipeline output with explicit feature/target separation."""

    frame: pd.DataFrame
    feature_columns: list[str]
    target_columns: list[str]
    source_columns: list[str]
    dropped_rows: int
    config: FeatureEngineeringConfig

    def copy(self) -> FeatureEngineeringResult:
        """Return a defensive copy for downstream experimentation."""
        return FeatureEngineeringResult(
            frame=self.frame.copy(),
            feature_columns=list(self.feature_columns),
            target_columns=list(self.target_columns),
            source_columns=list(self.source_columns),
            dropped_rows=self.dropped_rows,
            config=self.config,
        )


def _prepare_base_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw, cleaned, or already-derived data into daily metrics."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if data.empty:
        raise FeatureEngineeringError("The feature-engineering input is empty.")

    frame = data.copy()
    derived_required = {
        TOTAL_LOAD_COLUMN,
        NET_INTAKE_COLUMN,
        GROWTH_RATE_COLUMN,
    }
    if derived_required.issubset(frame.columns):
        if DATE_COLUMN in frame.columns:
            dates = pd.to_datetime(frame[DATE_COLUMN], errors="coerce")
            if dates.isna().any():
                raise FeatureEngineeringError("Derived input contains invalid dates.")
            frame[DATE_COLUMN] = dates
            frame = frame.set_index(DATE_COLUMN)
        elif not isinstance(frame.index, pd.DatetimeIndex):
            raise FeatureEngineeringError(
                "Derived input must use Date as a column or DatetimeIndex."
            )
        frame.index = pd.DatetimeIndex(frame.index, name=DATE_COLUMN)
        if frame.index.has_duplicates:
            raise FeatureEngineeringError("Derived input contains duplicate dates.")
        return frame.sort_index()

    raw_frame = frame.reset_index() if DATE_COLUMN not in frame.columns else frame
    try:
        cleaned = validate_and_clean_data(raw_frame)
        metrics = compute_capacity_metrics(cleaned)
    except (DataValidationError, TypeError, ValueError) as exc:
        raise FeatureEngineeringError(f"Unable to prepare source metrics: {exc}") from exc
    metrics.index = pd.DatetimeIndex(metrics.index, name=DATE_COLUMN)
    return metrics.sort_index()


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide numeric series while converting zero denominators to missing."""
    numerator_values = pd.to_numeric(numerator, errors="coerce")
    denominator_values = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    return numerator_values.div(denominator_values).replace([np.inf, -np.inf], np.nan)


def _rolling_slope(values: np.ndarray) -> float:
    """Return the least-squares slope for a complete rolling window."""
    numeric = np.asarray(values, dtype=float)
    if len(numeric) < 2 or not np.isfinite(numeric).all():
        return np.nan
    return float(np.polyfit(np.arange(len(numeric), dtype=float), numeric, 1)[0])


def _calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Create numeric and cyclical calendar features."""
    calendar = pd.DataFrame(index=index)
    day_of_week = index.dayofweek
    month = index.month
    day_of_year = index.dayofyear

    calendar["calendar_day_of_week"] = day_of_week
    calendar["calendar_month"] = month
    calendar["calendar_quarter"] = index.quarter
    calendar["calendar_day_of_month"] = index.day
    calendar["calendar_day_of_year"] = day_of_year
    calendar["calendar_week_of_year"] = index.isocalendar().week.to_numpy(dtype=int)
    calendar["calendar_is_weekend"] = (day_of_week >= 5).astype("int8")
    calendar["calendar_is_month_start"] = index.is_month_start.astype("int8")
    calendar["calendar_is_month_end"] = index.is_month_end.astype("int8")
    calendar["calendar_dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    calendar["calendar_dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    calendar["calendar_month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    calendar["calendar_month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    calendar["calendar_doy_sin"] = np.sin(2 * np.pi * (day_of_year - 1) / 365.25)
    calendar["calendar_doy_cos"] = np.cos(2 * np.pi * (day_of_year - 1) / 365.25)
    return calendar


def _operational_features(metrics: pd.DataFrame) -> pd.DataFrame:
    """Create same-day operational composition and flow-balance features."""
    features = pd.DataFrame(index=metrics.index)
    features["operational_transfer_to_intake_ratio"] = _safe_ratio(
        metrics[TRANSFER_COLUMN],
        metrics[INTAKE_COLUMN] + 1,
    )
    features["operational_discharge_to_transfer_ratio"] = _safe_ratio(
        metrics[DISCHARGE_COLUMN],
        metrics[TRANSFER_COLUMN] + 1,
    )
    features["operational_cbp_load_share"] = _safe_ratio(
        metrics[CBP_COLUMN],
        metrics[TOTAL_LOAD_COLUMN],
    )
    features["operational_hhs_load_share"] = _safe_ratio(
        metrics[HHS_COLUMN],
        metrics[TOTAL_LOAD_COLUMN],
    )
    features["operational_intake_transfer_gap"] = (
        metrics[INTAKE_COLUMN] - metrics[TRANSFER_COLUMN]
    )
    features["operational_transfer_discharge_gap"] = (
        metrics[TRANSFER_COLUMN] - metrics[DISCHARGE_COLUMN]
    )
    features["operational_load_per_apprehension"] = _safe_ratio(
        metrics[TOTAL_LOAD_COLUMN],
        metrics[INTAKE_COLUMN] + 1,
    )
    features["operational_cbp_hhs_load_ratio"] = _safe_ratio(
        metrics[CBP_COLUMN],
        metrics[HHS_COLUMN] + 1,
    )
    return features


def _lag_features(
    metrics: pd.DataFrame,
    lag_days: tuple[int, ...],
) -> pd.DataFrame:
    """Create historical lag features for all base operational signals."""
    features = pd.DataFrame(index=metrics.index)
    for lag in sorted(lag_days):
        for column in BASE_SIGNAL_COLUMNS:
            features[f"lag_{lag}d_{SIGNAL_NAMES[column]}"] = pd.to_numeric(
                metrics[column], errors="coerce"
            ).shift(lag)
    return features


def _rolling_features(
    metrics: pd.DataFrame,
    windows: tuple[int, ...],
    leakage_safe: bool,
) -> pd.DataFrame:
    """Create historical rolling statistics and trend slopes."""
    features = pd.DataFrame(index=metrics.index)
    rolling_signals = (
        TOTAL_LOAD_COLUMN,
        NET_INTAKE_COLUMN,
        INTAKE_COLUMN,
        TRANSFER_COLUMN,
        DISCHARGE_COLUMN,
    )
    for column in rolling_signals:
        values = pd.to_numeric(metrics[column], errors="coerce")
        history = values.shift(1) if leakage_safe else values
        signal_name = SIGNAL_NAMES[column]
        for window in sorted(windows):
            rolling = history.rolling(window=window, min_periods=window)
            prefix = f"rolling_{window}d_{signal_name}"
            features[f"{prefix}_mean"] = rolling.mean()
            features[f"{prefix}_std"] = rolling.std(ddof=0)
            features[f"{prefix}_min"] = rolling.min()
            features[f"{prefix}_max"] = rolling.max()

    load_history = pd.to_numeric(
        metrics[TOTAL_LOAD_COLUMN], errors="coerce"
    )
    if leakage_safe:
        load_history = load_history.shift(1)
    for window in sorted(windows):
        features[f"rolling_{window}d_total_system_load_slope"] = (
            load_history.rolling(window=window, min_periods=window).apply(
                _rolling_slope,
                raw=True,
            )
        )
    return features


def _exponential_features(
    metrics: pd.DataFrame,
    spans: tuple[int, ...],
    leakage_safe: bool,
) -> pd.DataFrame:
    """Create exponentially weighted historical load and pressure trends."""
    features = pd.DataFrame(index=metrics.index)
    for column in (TOTAL_LOAD_COLUMN, NET_INTAKE_COLUMN):
        values = pd.to_numeric(metrics[column], errors="coerce")
        history = values.shift(1) if leakage_safe else values
        signal_name = SIGNAL_NAMES[column]
        for span in sorted(spans):
            features[f"ema_{span}d_{signal_name}"] = history.ewm(
                span=span,
                adjust=False,
                min_periods=span,
            ).mean()
    return features


def _momentum_features(metrics: pd.DataFrame) -> pd.DataFrame:
    """Create short- and medium-term change, momentum, and acceleration features."""
    features = pd.DataFrame(index=metrics.index)
    load = pd.to_numeric(metrics[TOTAL_LOAD_COLUMN], errors="coerce")
    net_intake = pd.to_numeric(metrics[NET_INTAKE_COLUMN], errors="coerce")
    features["momentum_total_load_change_1d"] = load.diff(1)
    features["momentum_total_load_change_7d"] = load.diff(7)
    features["momentum_total_load_change_14d"] = load.diff(14)
    features["momentum_total_load_pct_change_7d"] = (
        load.pct_change(7, fill_method=None).replace([np.inf, -np.inf], np.nan)
    )
    features["momentum_total_load_pct_change_14d"] = (
        load.pct_change(14, fill_method=None).replace([np.inf, -np.inf], np.nan)
    )
    features["momentum_net_intake_change_1d"] = net_intake.diff(1)
    features["momentum_net_intake_change_7d"] = net_intake.diff(7)
    features["momentum_load_acceleration"] = load.diff(1).diff(1)
    features["momentum_net_intake_acceleration"] = net_intake.diff(1).diff(1)
    return features


def _quality_features(metrics: pd.DataFrame) -> pd.DataFrame:
    """Create model-visible data-quality indicators."""
    features = pd.DataFrame(index=metrics.index)
    transfer_anomaly = metrics.get(
        TRANSFER_ANOMALY_COLUMN,
        pd.Series(False, index=metrics.index),
    ).fillna(False).astype(bool)
    discharge_anomaly = metrics.get(
        DISCHARGE_ANOMALY_COLUMN,
        pd.Series(False, index=metrics.index),
    ).fillna(False).astype(bool)
    imputed_column = next(
        (
            column
            for column in ("Is Imputed Date", "Is_Imputed_Date")
            if column in metrics.columns
        ),
        None,
    )
    imputed = (
        metrics[imputed_column].fillna(False).astype(bool)
        if imputed_column
        else pd.Series(False, index=metrics.index)
    )
    features["quality_transfer_anomaly"] = transfer_anomaly.astype("int8")
    features["quality_discharge_anomaly"] = discharge_anomaly.astype("int8")
    features["quality_any_logical_anomaly"] = (
        transfer_anomaly | discharge_anomaly
    ).astype("int8")
    features["quality_is_imputed_date"] = imputed.astype("int8")
    return features


def _target_features(
    metrics: pd.DataFrame,
    horizon_days: int,
) -> pd.DataFrame:
    """Create future load, growth, intake, and backlog-risk targets."""
    targets = pd.DataFrame(index=metrics.index)
    load = pd.to_numeric(metrics[TOTAL_LOAD_COLUMN], errors="coerce")
    net_intake = pd.to_numeric(metrics[NET_INTAKE_COLUMN], errors="coerce")
    future_load = load.shift(-horizon_days)
    future_net_intake = net_intake.shift(-horizon_days)

    targets[f"target_total_load_t_plus_{horizon_days}d"] = future_load
    targets[f"target_load_change_t_plus_{horizon_days}d"] = future_load - load
    targets[f"target_load_growth_t_plus_{horizon_days}d"] = (
        future_load.sub(load).div(load.replace(0, np.nan))
    )
    targets[f"target_net_intake_t_plus_{horizon_days}d"] = future_net_intake
    targets[f"target_positive_pressure_t_plus_{horizon_days}d"] = (
        future_net_intake.gt(0).where(future_net_intake.notna()).astype("Float64")
    )

    future_pressure_columns = [net_intake.shift(-step) for step in range(1, horizon_days + 1)]
    future_pressure = pd.concat(future_pressure_columns, axis=1)
    targets[f"target_cumulative_net_intake_next_{horizon_days}d"] = (
        future_pressure.sum(axis=1, min_count=horizon_days)
    )
    targets[f"target_sustained_pressure_next_{horizon_days}d"] = (
        future_pressure.gt(0)
        .all(axis=1)
        .where(future_pressure.notna().all(axis=1))
        .astype("Float64")
    )
    return targets


class CapacityFeatureEngineer:
    """State-free transformer for capacity forecasting and risk features."""

    def __init__(self, config: FeatureEngineeringConfig | None = None) -> None:
        self.config = config or FeatureEngineeringConfig()

    def transform(self, data: pd.DataFrame) -> FeatureEngineeringResult:
        """Build features and optional future targets from daily source data."""
        metrics = _prepare_base_metrics(data)
        missing_signals = [
            column for column in BASE_SIGNAL_COLUMNS if column not in metrics.columns
        ]
        if missing_signals:
            raise FeatureEngineeringError(
                "Missing base analytical signal(s): " + ", ".join(missing_signals)
            )

        source_columns = list(metrics.columns)
        feature_parts = [
            _calendar_features(metrics.index),
            _operational_features(metrics),
            _lag_features(metrics, self.config.lag_days),
            _rolling_features(
                metrics,
                self.config.rolling_windows,
                self.config.leakage_safe_rolling,
            ),
            _exponential_features(
                metrics,
                self.config.ema_spans,
                self.config.leakage_safe_rolling,
            ),
            _momentum_features(metrics),
            _quality_features(metrics),
        ]
        engineered = pd.concat([metrics, *feature_parts], axis=1)
        feature_columns = [
            column
            for part in feature_parts
            for column in part.columns
        ]

        target_columns: list[str] = []
        if self.config.include_targets:
            targets = _target_features(metrics, self.config.target_horizon_days)
            engineered = pd.concat([engineered, targets], axis=1)
            target_columns = list(targets.columns)

        engineered = engineered.replace([np.inf, -np.inf], np.nan)
        original_rows = len(engineered)
        if self.config.drop_incomplete_rows:
            required_complete = feature_columns + target_columns
            engineered = engineered.dropna(subset=required_complete)
        dropped_rows = original_rows - len(engineered)

        engineered.index = pd.DatetimeIndex(engineered.index, name=DATE_COLUMN)
        engineered.attrs = {
            "feature_count": len(feature_columns),
            "target_count": len(target_columns),
            "target_horizon_days": self.config.target_horizon_days,
            "leakage_safe_rolling": self.config.leakage_safe_rolling,
        }
        return FeatureEngineeringResult(
            frame=engineered,
            feature_columns=feature_columns,
            target_columns=target_columns,
            source_columns=source_columns,
            dropped_rows=dropped_rows,
            config=self.config,
        )


def build_feature_matrix(
    data: pd.DataFrame,
    config: FeatureEngineeringConfig | None = None,
) -> FeatureEngineeringResult:
    """Functional entry point for the complete feature-engineering pipeline."""
    return CapacityFeatureEngineer(config).transform(data)


def split_features_and_target(
    result: FeatureEngineeringResult,
    target_column: str | None = None,
    *,
    drop_missing: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Extract a leakage-safe feature matrix and one supervised target series."""
    if not isinstance(result, FeatureEngineeringResult):
        raise TypeError("result must be a FeatureEngineeringResult.")
    if not result.target_columns:
        raise FeatureEngineeringError("No target columns were generated.")

    selected_target = target_column or result.target_columns[0]
    if selected_target not in result.target_columns:
        raise FeatureEngineeringError(
            f"Unknown target '{selected_target}'. Available targets: "
            + ", ".join(result.target_columns)
        )

    combined = result.frame[result.feature_columns + [selected_target]].copy()
    if drop_missing:
        combined = combined.dropna()
    features = combined[result.feature_columns].copy()
    target = combined[selected_target].copy()
    features.attrs.clear()
    target.attrs.clear()
    return features, target


def chronological_train_test_split(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    test_size: float = 0.2,
    gap_rows: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split time-series data chronologically without shuffling.

    ``gap_rows`` removes observations between train and test sets, helping reduce
    boundary leakage when targets use multi-day forecast horizons.
    """
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame.")
    if not isinstance(target, pd.Series):
        raise TypeError("target must be a pandas Series.")
    if features.empty or target.empty:
        raise FeatureEngineeringError("Features and target must not be empty.")
    if not features.index.equals(target.index):
        raise FeatureEngineeringError("Features and target indexes must match exactly.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if gap_rows < 0:
        raise ValueError("gap_rows must be non-negative.")

    test_rows = max(1, int(np.ceil(len(features) * test_size)))
    test_start = len(features) - test_rows
    train_end = test_start - gap_rows
    if train_end < 1:
        raise FeatureEngineeringError(
            "The requested test_size and gap_rows leave no training observations."
        )

    x_train = features.iloc[:train_end].copy()
    x_test = features.iloc[test_start:].copy()
    y_train = target.iloc[:train_end].copy()
    y_test = target.iloc[test_start:].copy()
    return x_train, x_test, y_train, y_test


def feature_manifest(result: FeatureEngineeringResult) -> pd.DataFrame:
    """Return a model-audit manifest for generated features and targets."""
    if not isinstance(result, FeatureEngineeringResult):
        raise TypeError("result must be a FeatureEngineeringResult.")

    records: list[dict[str, Any]] = []
    for column in result.feature_columns + result.target_columns:
        if column.startswith("calendar_"):
            category = "calendar"
        elif column.startswith("operational_"):
            category = "operational"
        elif column.startswith("lag_"):
            category = "lag"
        elif column.startswith("rolling_"):
            category = "rolling"
        elif column.startswith("ema_"):
            category = "exponential"
        elif column.startswith("momentum_"):
            category = "momentum"
        elif column.startswith("quality_"):
            category = "quality"
        elif column.startswith("target_"):
            category = "target"
        else:
            category = "other"

        series = result.frame[column]
        records.append(
            {
                "Column": column,
                "Role": "target" if column in result.target_columns else "feature",
                "Category": category,
                "Dtype": str(series.dtype),
                "Missing Values": int(series.isna().sum()),
                "Missing Rate": float(series.isna().mean()),
                "Unique Values": int(series.nunique(dropna=True)),
            }
        )
    manifest = pd.DataFrame.from_records(records)
    manifest.attrs.clear()
    return manifest
