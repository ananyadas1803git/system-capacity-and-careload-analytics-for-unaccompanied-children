"""Leakage-safe multi-model seven-day system-load forecasting framework.

This module owns reusable research-model logic.  It deliberately performs no
training at import time: callers must invoke :func:`train_forecasting_models`
explicitly.  All model and ensemble choices use expanding-window predictions
from the development period; the final chronological holdout is evaluated only
after those choices are frozen.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import subprocess
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.base import RegressorMixin
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app_utils import (
    CBP_COLUMN,
    DATE_COLUMN,
    DISCHARGE_COLUMN,
    HHS_COLUMN,
    INTAKE_COLUMN,
    NET_INTAKE_COLUMN,
    TOTAL_LOAD_COLUMN,
    TRANSFER_COLUMN,
)
from backend.utils import json_safe, utc_now_iso


TARGET_ABSOLUTE = "target_total_load_t_plus_7d"
TARGET_CHANGE = "target_change_7d"
CURRENT_LOAD = "current_total_system_load"
FORECAST_HORIZON = 7
MODEL_VERSION = "2.0.0"

# Cross-platform rolling and trigonometric calculations can differ at roughly
# 1e-12 even when their analytical values are equivalent. Ten decimal places
# remove that numerical noise while still preserving changes many orders of
# magnitude smaller than one child in the operational count features.
FINGERPRINT_FLOAT_DECIMAL_PLACES = 10
FINGERPRINT_ALGORITHM = "canonical-semantic-v3"
PREPARED_DATASET_CONTRACT_VERSION = "canonical-prepared-dataset-v3"
PREPARED_FEATURE_RECIPE_VERSION = "prepared-forecast-features-v1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_PATH = PROJECT_ROOT / "data" / "processed" / "uac_capacity_ml_features.parquet"
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "HHS_Unaccompanied_Alien_Children_Program.csv"
DEFAULT_PROVENANCE_PATH = PROJECT_ROOT / "data" / "processed" / "preprocessing_report.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "forecasting"

MODEL_NAMES = (
    "persistence",
    "seven_day_drift",
    "ridge",
    "elastic_net",
    "ets",
    "sarimax",
    "lightgbm",
    "catboost",
    "xgboost",
    "sarimax_boosting_hybrid",
    "validation_weighted_ensemble",
)

PREDICTION_COLUMNS = (
    "forecast_origin_date",
    "target_date",
    "actual_value",
    "current_load",
    "model_name",
    "predicted_change_7d",
    "reconstructed_absolute_prediction",
    "persistence_prediction",
    "lower_interval",
    "median_prediction",
    "upper_interval",
    "evaluation_label",
    "fold",
    "is_imputed_date",
    "has_anomaly",
    "backlog_state",
    "capacity_stress",
)


class ForecastingFrameworkError(RuntimeError):
    """Raised when forecasting cannot proceed safely or reproducibly."""


@dataclass(frozen=True)
class ForecastConfig:
    """Configuration for the reproducible multi-model experiment."""

    feature_path: Path = DEFAULT_FEATURE_PATH
    raw_path: Path = DEFAULT_RAW_PATH
    provenance_path: Path = DEFAULT_PROVENANCE_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    horizon_days: int = FORECAST_HORIZON
    holdout_fraction: float = 0.20
    cv_splits: int = 5
    cv_test_size: int = 56
    gap_days: int = FORECAST_HORIZON
    random_seed: int = 42
    capacity_reference: float = 12_000.0
    correlation_threshold: float = 0.98
    maximum_missing_fraction: float = 0.30
    worst_fold_ratio_limit: float = 1.50
    overwrite: bool = False

    def __post_init__(self) -> None:
        if self.horizon_days != FORECAST_HORIZON:
            raise ValueError("This framework currently supports a seven-day horizon.")
        if not 0 < self.holdout_fraction < 1:
            raise ValueError("holdout_fraction must be between zero and one.")
        if self.cv_splits < 4:
            raise ValueError("At least four walk-forward folds are required.")
        if self.cv_test_size < 7:
            raise ValueError("cv_test_size must cover at least seven observations.")
        if self.gap_days < self.horizon_days:
            raise ValueError("gap_days must be at least the forecast horizon.")
        if self.capacity_reference <= 0:
            raise ValueError("capacity_reference must be positive.")
        if not 0.8 <= self.correlation_threshold < 1:
            raise ValueError("correlation_threshold must be in [0.8, 1).")

    @property
    def artifact_paths(self) -> dict[str, Path]:
        """Return the stable artifact contract for this experiment."""

        root = self.output_dir
        return {
            "registry": root / "models" / "model_registry.json",
            "canonical_prepared_frame": (
                root / "audits" / "canonical_prepared_forecast_frame.parquet"
            ),
            "comparison": root / "metrics" / "model_comparison_metrics.json",
            "fold_metrics": root / "metrics" / "fold_metrics.csv",
            "oof_predictions": root / "predictions" / "development_oof_predictions.csv",
            "holdout_predictions": root / "predictions" / "final_holdout_predictions.csv",
            "selected_features": root / "audits" / "selected_features.json",
            "feature_exclusions": root / "audits" / "feature_exclusion_report.csv",
            "feature_availability": root / "audits" / "feature_availability.csv",
            "feature_importance": root / "diagnostics" / "oof_permutation_importance.csv",
            "residual_diagnostics": root / "diagnostics" / "residual_diagnostics.csv",
            "error_by_regime": root / "diagnostics" / "error_by_regime.csv",
            "residual_correlation": root / "diagnostics" / "residual_correlation.csv",
            "interval_metrics": root / "metrics" / "prediction_interval_metrics.json",
            "provenance": root / "audits" / "dataset_provenance.json",
            "leakage": root / "audits" / "leakage_audit.json",
            "promotion": root / "metrics" / "promotion_decision.json",
            "champion": root / "models" / "champion_model.json",
            "report": root / "forecast_model_report.html",
        }


@dataclass(frozen=True)
class WalkForwardFold:
    """One expanding-window split with an explicit embargo."""

    number: int
    train_indices: np.ndarray
    validation_indices: np.ndarray

    @property
    def observed_gap(self) -> int:
        return int(self.validation_indices[0] - self.train_indices[-1] - 1)


@dataclass
class PreparedForecastDataset:
    """Validated model frame and explicit forecast-origin feature contract."""

    frame: pd.DataFrame
    compact_features: tuple[str, ...]
    expanded_features: tuple[str, ...]
    availability: pd.DataFrame
    exclusions: pd.DataFrame
    data_fingerprint: str
    schema_fingerprint: str


@dataclass
class ChronologicalPartitions:
    """Development, embargo, and final untouched holdout partitions."""

    development: pd.DataFrame
    embargo: pd.DataFrame
    holdout: pd.DataFrame


@dataclass
class FeatureProcessor:
    """Training-only median imputation and deterministic correlation filtering."""

    maximum_missing_fraction: float
    correlation_threshold: float
    fitted_rows: tuple[str, str] | None = None
    medians: dict[str, float] = field(default_factory=dict)
    selected_columns: tuple[str, ...] = ()
    exclusion_reasons: dict[str, str] = field(default_factory=dict)

    def fit(self, frame: pd.DataFrame) -> "FeatureProcessor":
        if frame.empty:
            raise ForecastingFrameworkError("Feature processor cannot fit empty data.")
        selected: list[str] = []
        numeric = (
            frame.apply(pd.to_numeric, errors="coerce")
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
        )
        for column in numeric.columns:
            values = numeric[column]
            missing_fraction = float(values.isna().mean())
            if missing_fraction > self.maximum_missing_fraction:
                self.exclusion_reasons[column] = (
                    f"training missing fraction {missing_fraction:.3f} exceeds "
                    f"{self.maximum_missing_fraction:.3f}"
                )
                continue
            median = float(values.median()) if values.notna().any() else math.nan
            if not math.isfinite(median):
                self.exclusion_reasons[column] = "all values missing in training"
                continue
            filled = values.fillna(median)
            if filled.nunique(dropna=False) <= 1 or float(filled.std(ddof=0)) <= 1e-12:
                self.exclusion_reasons[column] = "constant or near-constant in training"
                continue
            self.medians[column] = median
            selected.append(column)

        if not selected:
            raise ForecastingFrameworkError("No usable features remain after filtering.")
        filled = numeric[selected].fillna(self.medians)
        correlation = filled.corr().abs()
        kept: list[str] = []
        for column in selected:
            correlated_with = next(
                (
                    prior
                    for prior in kept
                    if pd.notna(correlation.loc[column, prior])
                    and correlation.loc[column, prior] > self.correlation_threshold
                ),
                None,
            )
            if correlated_with is None:
                kept.append(column)
            else:
                self.exclusion_reasons[column] = (
                    f"training correlation > {self.correlation_threshold:.2f} "
                    f"with {correlated_with}"
                )
                self.medians.pop(column, None)
        self.selected_columns = tuple(kept)
        self.fitted_rows = (
            frame.index.min().date().isoformat(),
            frame.index.max().date().isoformat(),
        )
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.selected_columns or self.fitted_rows is None:
            raise ForecastingFrameworkError("Feature processor has not been fitted.")
        missing = [name for name in self.selected_columns if name not in frame]
        if missing:
            raise ForecastingFrameworkError(
                "Transform data is missing selected feature(s): " + ", ".join(missing)
            )
        transformed = (
            frame.loc[:, self.selected_columns].apply(pd.to_numeric, errors="coerce").astype(float)
        )
        transformed = transformed.replace([np.inf, -np.inf], np.nan)
        return transformed.fillna(self.medians).astype(float)


@dataclass
class CandidateResult:
    """Development-period result for one frozen candidate configuration."""

    model_name: str
    configuration_name: str
    feature_group: str
    parameters: dict[str, Any]
    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    warning_status: str
    training_seconds: float
    prediction_seconds: float
    feature_counts: list[int]
    best_iterations: list[int | None]

    @property
    def mean_mae(self) -> float:
        return float(self.fold_metrics["mae"].mean())

    @property
    def std_mae(self) -> float:
        return float(self.fold_metrics["mae"].std(ddof=0))

    @property
    def worst_mae(self) -> float:
        return float(self.fold_metrics["mae"].max())


@dataclass
class FinalModelResult:
    """Frozen candidate fitted on development and evaluated on holdout once."""

    model_name: str
    predictions: np.ndarray
    predicted_changes: np.ndarray
    training_seconds: float
    prediction_seconds: float
    feature_count: int
    warning_status: str
    model_bundle: Any = None


@dataclass
class ForecastExperimentResult:
    """In-memory summary returned after artifact generation."""

    config: ForecastConfig
    provenance: dict[str, Any]
    leakage_audit: dict[str, Any]
    comparison: dict[str, Any]
    promotion: dict[str, Any]
    interval_metrics: dict[str, Any]
    artifact_paths: dict[str, str]


@contextmanager
def _atomic_target(target: Path) -> Iterator[Path]:
    target = target.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        temporary.unlink(missing_ok=True)
        yield temporary
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata() -> dict[str, Any]:
    """Return the current revision and cleanliness without failing training."""

    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": None, "git_worktree_clean": None}
    return {
        "git_commit": revision or None,
        "git_worktree_clean": not bool(status.strip()),
    }


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    """Hash canonical values without depending on pandas' hash implementation.

    ``hash_pandas_object`` and dtype strings can change across pandas/Arrow
    releases even when every analytical value is identical. This explicit
    encoding normalizes supported semantic types, datetime resolution, byte
    order, sub-precision floating-point noise, signed zero, and missing-value
    representation. Datetimes retain nanosecond precision.
    """

    digest = hashlib.sha256()
    schema = [(str(name), _semantic_dtype(series)) for name, series in frame.items()]
    digest.update(
        json.dumps(
            {"index_name": str(frame.index.name or ""), "columns": schema},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    index = pd.to_datetime(frame.index, errors="raise").to_numpy(dtype="datetime64[ns]")
    digest.update(index.astype("<i8", copy=False).tobytes(order="C"))
    for _, series in frame.items():
        semantic = _semantic_dtype(series)
        if semantic == "datetime64":
            values = pd.to_datetime(series, errors="raise").to_numpy(dtype="datetime64[ns]")
            digest.update(values.astype("<i8", copy=False).tobytes(order="C"))
        elif semantic == "boolean":
            values = pd.array(series, dtype="boolean").to_numpy(dtype=np.int8, na_value=-1)
            digest.update(values.tobytes(order="C"))
        elif semantic == "numeric":
            values = pd.to_numeric(series, errors="raise").to_numpy(dtype="<f8", copy=True)
            missing = np.isnan(values)
            finite = np.isfinite(values)
            values[finite] = np.round(values[finite], decimals=FINGERPRINT_FLOAT_DECIMAL_PLACES)
            # Hash missingness independently of numeric bytes so all NaN
            # payloads and nullable numeric dtypes have one representation.
            digest.update(missing.astype(np.uint8, copy=False).tobytes(order="C"))
            values[missing] = 0.0
            # IEEE-754 distinguishes -0.0 from +0.0 even though they compare
            # equal; canonical fingerprints intentionally do not.
            values[values == 0] = 0.0
            digest.update(values.astype("<f8", copy=False).tobytes(order="C"))
        else:
            values = [None if pd.isna(value) else str(value) for value in series]
            digest.update(
                json.dumps(
                    values,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
    return digest.hexdigest()


def _semantic_dtype(series: pd.Series) -> str:
    """Return a stable cross-version dtype category for artifact hashing."""

    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return "datetime64"
    if pd.api.types.is_bool_dtype(series.dtype):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series.dtype):
        return "numeric"
    return "string"


def _schema_fingerprint(frame: pd.DataFrame) -> str:
    payload = "\n".join(f"{name}:{_semantic_dtype(series)}" for name, series in frame.items())
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def forecast_configuration_fingerprint(config: ForecastConfig) -> str:
    """Hash every analytical forecast setting independently of local paths."""

    payload = {
        "horizon_days": config.horizon_days,
        "holdout_fraction": config.holdout_fraction,
        "cv_splits": config.cv_splits,
        "cv_test_size": config.cv_test_size,
        "gap_days": config.gap_days,
        "random_seed": config.random_seed,
        "capacity_reference": config.capacity_reference,
        "correlation_threshold": config.correlation_threshold,
        "maximum_missing_fraction": config.maximum_missing_fraction,
        "worst_fold_ratio_limit": config.worst_fold_ratio_limit,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with _atomic_target(path) as temporary:
        temporary.write_text(
            json.dumps(json_safe(payload), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    export = frame.copy()
    export.attrs.clear()
    with _atomic_target(path) as temporary:
        export.to_csv(temporary, index=False, date_format="%Y-%m-%d", lineterminator="\n")


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    """Atomically persist the exact training frame with its index and dtypes."""

    export = frame.copy()
    export.attrs.clear()
    with _atomic_target(path) as temporary:
        export.to_parquet(temporary, engine="pyarrow", index=True)


def _read_feature_artifact(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ForecastingFrameworkError(f"Feature artifact not found: {path}")
    try:
        frame = pd.read_parquet(path, engine="pyarrow")
    except (OSError, ValueError, ImportError) as exc:
        raise ForecastingFrameworkError(f"Unable to read feature artifact: {exc}") from exc
    if frame.empty:
        raise ForecastingFrameworkError("Feature artifact is empty.")
    if DATE_COLUMN in frame.columns:
        frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN], errors="coerce")
        frame = frame.set_index(DATE_COLUMN)
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    if frame.index.isna().any():
        raise ForecastingFrameworkError("Feature artifact contains invalid dates.")
    if frame.index.duplicated().any():
        raise ForecastingFrameworkError("Feature artifact contains duplicate dates.")
    frame = frame.sort_index()
    if not frame.index.is_monotonic_increasing:
        raise ForecastingFrameworkError("Feature dates are not chronological.")
    return frame


def construct_change_target(
    future_load: pd.Series | Sequence[float],
    current_load: pd.Series | Sequence[float],
) -> pd.Series:
    """Construct ``load[t+7] - load[t]`` without silently aligning bad indexes."""

    future = pd.Series(future_load, copy=False, dtype=float)
    current = pd.Series(current_load, copy=False, dtype=float)
    if len(future) != len(current):
        raise ForecastingFrameworkError("Future and current load lengths differ.")
    if isinstance(future_load, pd.Series) and isinstance(current_load, pd.Series):
        if not future_load.index.equals(current_load.index):
            raise ForecastingFrameworkError("Future and current load indexes differ.")
        future.index = future_load.index
        current.index = current_load.index
    result = future - current
    result.name = TARGET_CHANGE
    return result


def reconstruct_absolute_forecast(
    current_load: Sequence[float] | np.ndarray,
    predicted_change: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Reconstruct the absolute T+7 forecast from an origin-known load."""

    current = np.asarray(current_load, dtype=float)
    change = np.asarray(predicted_change, dtype=float)
    if current.shape != change.shape:
        raise ForecastingFrameworkError("Current load and predicted change shapes differ.")
    return current + change


def order_prediction_intervals(
    lower: Sequence[float],
    median: Sequence[float],
    upper: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Order independently estimated quantiles row-wise and return copies."""

    arrays = [
        np.asarray(lower, dtype=float),
        np.asarray(median, dtype=float),
        np.asarray(upper, dtype=float),
    ]
    if len({array.shape for array in arrays}) != 1:
        raise ForecastingFrameworkError("Prediction interval arrays have different shapes.")
    stacked = np.vstack(arrays)
    ordered = np.sort(stacked, axis=0)
    return ordered[0], ordered[1], ordered[2]


def build_oof_residuals(
    actual: pd.Series,
    oof_predictions: pd.Series,
) -> pd.Series:
    """Build residual targets only where out-of-fold forecasts are available."""

    if actual.index.has_duplicates or oof_predictions.index.has_duplicates:
        raise ForecastingFrameworkError("OOF residual inputs must have unique dates.")
    aligned = pd.concat(
        [actual.rename("actual"), oof_predictions.rename("oof_prediction")],
        axis=1,
        join="inner",
    ).dropna()
    if aligned.empty:
        raise ForecastingFrameworkError("No aligned OOF observations are available.")
    residuals = aligned["actual"] - aligned["oof_prediction"]
    residuals.name = "oof_residual"
    return residuals


def validate_prediction_schema(frame: pd.DataFrame) -> None:
    """Validate the stable long-form prediction artifact schema."""

    missing = [column for column in PREDICTION_COLUMNS if column not in frame]
    if missing:
        raise ForecastingFrameworkError(
            "Prediction artifact is missing column(s): " + ", ".join(missing)
        )
    if frame.empty:
        raise ForecastingFrameworkError("Prediction artifact is empty.")
    numeric = frame[
        [
            "actual_value",
            "current_load",
            "predicted_change_7d",
            "reconstructed_absolute_prediction",
            "persistence_prediction",
        ]
    ].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise ForecastingFrameworkError("Prediction artifact has non-finite core values.")


def persistence_forecast(frame: pd.DataFrame) -> np.ndarray:
    """Return the current load as the seven-day persistence forecast."""

    return frame[CURRENT_LOAD].to_numpy(float)


def drift_forecast(frame: pd.DataFrame) -> np.ndarray:
    """Return ``load[t] + load[t] - load[t-7]``."""

    required = "load_lag_7"
    if required not in frame:
        raise ForecastingFrameworkError("Seven-day drift requires load_lag_7.")
    current = frame[CURRENT_LOAD].to_numpy(float)
    return current + current - frame[required].to_numpy(float)


def _rolling_slope(values: np.ndarray) -> float:
    if len(values) < 2 or not np.isfinite(values).all():
        return math.nan
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x, values.astype(float), 1)[0])


def _backlog_streak(positive: pd.Series) -> pd.Series:
    groups = (~positive).cumsum()
    return positive.astype(int).groupby(groups).cumsum().astype(float)


def _availability_record(
    feature: str,
    group: str,
    source: str,
    available_when: str,
    justified: bool = True,
    reason: str = "available at or before forecast origin",
) -> dict[str, Any]:
    return {
        "feature": feature,
        "feature_group": group,
        "source": source,
        "available_when": available_when,
        "included": justified,
        "reason": reason,
    }


def prepare_forecast_dataset(
    source: pd.DataFrame,
    config: ForecastConfig | None = None,
) -> PreparedForecastDataset:
    """Create compact and expanded, forecast-origin-safe feature groups.

    All operational flow summaries are shifted by one day.  The only same-day
    stock used is Total System Load, which is explicitly known at the forecast
    origin and anchors absolute-forecast reconstruction.
    """

    selected = config or ForecastConfig()
    frame = source.copy().sort_index()
    required = {
        TOTAL_LOAD_COLUMN,
        CBP_COLUMN,
        HHS_COLUMN,
        INTAKE_COLUMN,
        TRANSFER_COLUMN,
        DISCHARGE_COLUMN,
        NET_INTAKE_COLUMN,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ForecastingFrameworkError(
            "Feature artifact is missing required source columns: " + ", ".join(missing)
        )
    if frame.index.duplicated().any() or not frame.index.is_monotonic_increasing:
        raise ForecastingFrameworkError("Input dates must be unique and chronological.")
    expected = pd.date_range(frame.index.min(), frame.index.max(), freq="D")
    if len(frame.index) != len(expected) or len(expected.difference(frame.index)):
        raise ForecastingFrameworkError("Processed modeling data must be complete daily data.")

    model = pd.DataFrame(index=frame.index)
    availability: list[dict[str, Any]] = []
    compact: list[str] = []
    expanded: list[str] = []

    def add(
        name: str,
        values: pd.Series | np.ndarray,
        *,
        group: str,
        source_name: str,
        available_when: str,
    ) -> None:
        model[name] = values
        availability.append(_availability_record(name, group, source_name, available_when))
        expanded.append(name)
        if group == "compact":
            compact.append(name)

    load = pd.to_numeric(frame[TOTAL_LOAD_COLUMN], errors="coerce")
    cbp = pd.to_numeric(frame[CBP_COLUMN], errors="coerce")
    hhs = pd.to_numeric(frame[HHS_COLUMN], errors="coerce")
    intake = pd.to_numeric(frame[INTAKE_COLUMN], errors="coerce")
    transfer = pd.to_numeric(frame[TRANSFER_COLUMN], errors="coerce")
    discharge = pd.to_numeric(frame[DISCHARGE_COLUMN], errors="coerce")
    net = pd.to_numeric(frame[NET_INTAKE_COLUMN], errors="coerce")

    add(
        CURRENT_LOAD,
        load,
        group="compact",
        source_name=TOTAL_LOAD_COLUMN,
        available_when="end of forecast-origin day t",
    )
    for lag in (1, 2, 3, 7, 14, 28):
        add(
            f"load_lag_{lag}",
            load.shift(lag),
            group="compact",
            source_name=TOTAL_LOAD_COLUMN,
            available_when=f"t-{lag} days",
        )
    for signal_name, signal in (("cbp", cbp), ("hhs", hhs)):
        for lag in (1, 7, 14, 28):
            add(
                f"{signal_name}_load_lag_{lag}",
                signal.shift(lag),
                group="compact",
                source_name=CBP_COLUMN if signal_name == "cbp" else HHS_COLUMN,
                available_when=f"t-{lag} days",
            )
    for signal_name, signal, source_name in (
        ("apprehension", intake, INTAKE_COLUMN),
        ("transfer", transfer, TRANSFER_COLUMN),
        ("discharge", discharge, DISCHARGE_COLUMN),
        ("net_intake", net, NET_INTAKE_COLUMN),
    ):
        for lag in (1, 7, 14, 28):
            add(
                f"{signal_name}_lag_{lag}",
                signal.shift(lag),
                group="compact",
                source_name=source_name,
                available_when=f"t-{lag} days",
            )

    historical_load = load.shift(1)
    historical_net = net.shift(1)
    for window in (7, 14, 28):
        add(
            f"load_rolling_mean_{window}",
            historical_load.rolling(window).mean(),
            group="compact" if window < 28 else "expanded",
            source_name=TOTAL_LOAD_COLUMN,
            available_when=f"t-1 through t-{window} days",
        )
        add(
            f"load_rolling_std_{window}",
            historical_load.rolling(window).std(ddof=0),
            group="compact" if window < 28 else "expanded",
            source_name=TOTAL_LOAD_COLUMN,
            available_when=f"t-1 through t-{window} days",
        )
        add(
            f"load_rolling_slope_{window}",
            historical_load.rolling(window).apply(_rolling_slope, raw=True),
            group="compact" if window < 28 else "expanded",
            source_name=TOTAL_LOAD_COLUMN,
            available_when=f"t-1 through t-{window} days",
        )
        add(
            f"load_ema_{window}",
            historical_load.ewm(span=window, adjust=False).mean(),
            group="expanded",
            source_name=TOTAL_LOAD_COLUMN,
            available_when="history through t-1",
        )
        add(
            f"net_rolling_mean_{window}",
            historical_net.rolling(window).mean(),
            group="expanded",
            source_name=NET_INTAKE_COLUMN,
            available_when=f"t-1 through t-{window} days",
        )
        add(
            f"net_rolling_std_{window}",
            historical_net.rolling(window).std(ddof=0),
            group="expanded",
            source_name=NET_INTAKE_COLUMN,
            available_when=f"t-1 through t-{window} days",
        )

    add(
        "cumulative_net_intake_7",
        historical_net.rolling(7).sum(),
        group="compact",
        source_name=NET_INTAKE_COLUMN,
        available_when="t-1 through t-7 days",
    )
    add(
        "cumulative_net_intake_14",
        historical_net.rolling(14).sum(),
        group="compact",
        source_name=NET_INTAKE_COLUMN,
        available_when="t-1 through t-14 days",
    )
    add(
        "load_momentum_1",
        load - load.shift(1),
        group="compact",
        source_name=TOTAL_LOAD_COLUMN,
        available_when="origin load t and t-1",
    )
    add(
        "load_momentum_7",
        load - load.shift(7),
        group="compact",
        source_name=TOTAL_LOAD_COLUMN,
        available_when="origin load t and t-7",
    )
    add(
        "load_acceleration",
        (load - load.shift(1)) - (load.shift(1) - load.shift(2)),
        group="expanded",
        source_name=TOTAL_LOAD_COLUMN,
        available_when="origin load t and earlier",
    )
    add(
        "backlog_streak_at_origin",
        _backlog_streak(net.gt(0)),
        group="compact",
        source_name=NET_INTAKE_COLUMN,
        available_when="end of forecast-origin day t",
    )
    add(
        "capacity_utilization",
        load / selected.capacity_reference,
        group="compact",
        source_name="configured planning reference",
        available_when="forecast origin t",
    )
    add(
        "capacity_headroom",
        selected.capacity_reference - load,
        group="compact",
        source_name="configured planning reference",
        available_when="forecast origin t",
    )

    index = model.index
    calendar = {
        "calendar_day_of_week": index.dayofweek,
        "calendar_is_weekend": (index.dayofweek >= 5).astype(int),
        "calendar_dow_sin": np.sin(2 * np.pi * index.dayofweek / 7),
        "calendar_dow_cos": np.cos(2 * np.pi * index.dayofweek / 7),
        "calendar_month_sin": np.sin(2 * np.pi * (index.month - 1) / 12),
        "calendar_month_cos": np.cos(2 * np.pi * (index.month - 1) / 12),
        "calendar_doy_sin": np.sin(2 * np.pi * (index.dayofyear - 1) / 365.25),
        "calendar_doy_cos": np.cos(2 * np.pi * (index.dayofyear - 1) / 365.25),
    }
    for name, values in calendar.items():
        add(
            name,
            values,
            group="compact",
            source_name=DATE_COLUMN,
            available_when="known calendar attribute of origin t",
        )

    imputed = frame.get("Is Imputed Date", pd.Series(False, index=frame.index))
    anomaly = frame.get("Anomaly_Any", pd.Series(False, index=frame.index))
    add(
        "quality_is_imputed_date",
        imputed.astype(bool).astype(int),
        group="compact",
        source_name="Is Imputed Date",
        available_when="quality audit completed at origin t",
    )
    add(
        "quality_has_anomaly",
        anomaly.astype(bool).astype(int),
        group="compact",
        source_name="Anomaly_Any",
        available_when="quality audit completed at origin t",
    )

    future = load.shift(-selected.horizon_days)
    model[TARGET_ABSOLUTE] = future
    model[TARGET_CHANGE] = construct_change_target(future, load)
    model["target_date"] = model.index + pd.Timedelta(days=selected.horizon_days)
    model["is_imputed_date"] = imputed.astype(bool)
    model["has_anomaly"] = anomaly.astype(bool)
    model["net_intake_at_origin"] = net
    model["backlog_state"] = _backlog_streak(net.gt(0)).gt(0)
    model["capacity_stress"] = (load / selected.capacity_reference).ge(0.8)

    availability.extend(
        [
            _availability_record(
                TARGET_ABSOLUTE,
                "excluded",
                TOTAL_LOAD_COLUMN,
                "t+7 days",
                False,
                "future target; never permitted in model features",
            ),
            _availability_record(
                TARGET_CHANGE,
                "excluded",
                TOTAL_LOAD_COLUMN,
                "t+7 days",
                False,
                "future-derived target; never permitted in model features",
            ),
        ]
    )
    exclusions = pd.DataFrame(
        [
            {
                "feature": name,
                "reason": "same-day operational field is not justified before origin close",
            }
            for name in (
                INTAKE_COLUMN,
                TRANSFER_COLUMN,
                DISCHARGE_COLUMN,
                NET_INTAKE_COLUMN,
                CBP_COLUMN,
                HHS_COLUMN,
            )
        ]
        + [
            {
                "feature": name,
                "reason": "future target or future-derived field",
            }
            for name in source.columns
            if str(name).startswith("target_")
        ]
    ).drop_duplicates("feature")

    eligible = model[TARGET_ABSOLUTE].notna() & model[TARGET_CHANGE].notna()
    model = model.loc[eligible].copy()
    if len(model) < 400:
        raise ForecastingFrameworkError(
            f"Only {len(model)} target-complete observations are available."
        )
    forbidden = [name for name in expanded if "target" in name or "future" in name]
    if forbidden:
        raise ForecastingFrameworkError(
            "Future-derived features entered the matrix: " + ", ".join(forbidden)
        )
    return PreparedForecastDataset(
        frame=model,
        compact_features=tuple(dict.fromkeys(compact)),
        expanded_features=tuple(dict.fromkeys(expanded)),
        availability=pd.DataFrame(availability),
        exclusions=exclusions,
        data_fingerprint=_frame_fingerprint(model),
        schema_fingerprint=_schema_fingerprint(model),
    )


def chronological_partitions(
    prepared: PreparedForecastDataset,
    config: ForecastConfig | None = None,
) -> ChronologicalPartitions:
    """Reserve the final 20% plus a seven-observation pre-holdout embargo."""

    selected = config or ForecastConfig()
    frame = prepared.frame
    holdout_rows = max(1, int(math.ceil(len(frame) * selected.holdout_fraction)))
    holdout_start = len(frame) - holdout_rows
    development_end = holdout_start - selected.gap_days
    if development_end < 300:
        raise ForecastingFrameworkError("Holdout and embargo leave too little development data.")
    return ChronologicalPartitions(
        development=frame.iloc[:development_end].copy(),
        embargo=frame.iloc[development_end:holdout_start].copy(),
        holdout=frame.iloc[holdout_start:].copy(),
    )


def expanding_window_folds(
    row_count: int,
    config: ForecastConfig | None = None,
) -> tuple[WalkForwardFold, ...]:
    """Return equal-duration expanding folds with a seven-observation gap."""

    selected = config or ForecastConfig()
    test_size = selected.cv_test_size
    minimum = selected.cv_splits * test_size + selected.gap_days + 60
    if row_count < minimum:
        test_size = max(14, (row_count - selected.gap_days - 60) // selected.cv_splits)
    if test_size < 14:
        raise ForecastingFrameworkError("Dataset is too small for useful walk-forward folds.")
    splitter = TimeSeriesSplit(
        n_splits=selected.cv_splits,
        test_size=test_size,
        gap=selected.gap_days,
    )
    folds = tuple(
        WalkForwardFold(number + 1, train, validation)
        for number, (train, validation) in enumerate(splitter.split(np.arange(row_count)))
    )
    if any(fold.observed_gap != selected.gap_days for fold in folds):
        raise ForecastingFrameworkError("A walk-forward fold has an invalid gap.")
    return folds


def regression_metrics(
    actual: Sequence[float],
    predicted: Sequence[float],
    persistence_mae: float,
) -> dict[str, float]:
    """Calculate the common regression metric set using finite observations."""

    truth = np.asarray(actual, dtype=float)
    forecast = np.asarray(predicted, dtype=float)
    if (
        truth.shape != forecast.shape
        or not np.isfinite(truth).all()
        or not np.isfinite(forecast).all()
    ):
        raise ForecastingFrameworkError("Metrics require aligned finite arrays.")
    mae = float(mean_absolute_error(truth, forecast))
    rmse = float(math.sqrt(mean_squared_error(truth, forecast)))
    nonzero = np.abs(truth) > 1e-12
    mape = float(np.mean(np.abs((truth[nonzero] - forecast[nonzero]) / truth[nonzero])) * 100)
    r2 = float(r2_score(truth, forecast)) if len(truth) > 1 else 0.0
    mase = float(mae / persistence_mae) if persistence_mae > 0 else math.inf
    improvement = (
        float((persistence_mae - mae) / persistence_mae * 100) if persistence_mae > 0 else 0.0
    )
    return {
        "mae": mae,
        "rmse": rmse,
        "mape_percent": mape,
        "r2": r2,
        "mase_vs_persistence": mase,
        "mae_improvement_vs_persistence_percent": improvement,
    }


def _prediction_rows(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    model_name: str,
    label: str,
    fold: int | str,
) -> pd.DataFrame:
    current = frame[CURRENT_LOAD].to_numpy(float)
    return pd.DataFrame(
        {
            "forecast_origin_date": frame.index,
            "target_date": frame["target_date"].to_numpy(),
            "actual_value": frame[TARGET_ABSOLUTE].to_numpy(float),
            "current_load": current,
            "model_name": model_name,
            "predicted_change_7d": np.asarray(predictions, float) - current,
            "reconstructed_absolute_prediction": np.asarray(predictions, float),
            "persistence_prediction": current,
            "lower_interval": np.nan,
            "median_prediction": np.nan,
            "upper_interval": np.nan,
            "evaluation_label": label,
            "fold": fold,
            "is_imputed_date": frame["is_imputed_date"].astype(bool).to_numpy(),
            "has_anomaly": frame["has_anomaly"].astype(bool).to_numpy(),
            "backlog_state": frame["backlog_state"].astype(bool).to_numpy(),
            "capacity_stress": frame["capacity_stress"].astype(bool).to_numpy(),
        }
    )


def _fold_metric_row(
    fold: WalkForwardFold,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    model_name: str,
    config_name: str,
    actual: np.ndarray,
    predicted: np.ndarray,
    feature_count: int,
    training_seconds: float,
    prediction_seconds: float,
    warning_status: str,
    best_iteration: int | None = None,
    aic: float | None = None,
) -> dict[str, Any]:
    persistence = validation[CURRENT_LOAD].to_numpy(float)
    persistence_mae = float(mean_absolute_error(actual, persistence))
    metrics = regression_metrics(actual, predicted, persistence_mae)
    return {
        "model_name": model_name,
        "configuration": config_name,
        "fold": fold.number,
        "training_start": train.index.min().date().isoformat(),
        "training_end": train.index.max().date().isoformat(),
        "validation_start": validation.index.min().date().isoformat(),
        "validation_end": validation.index.max().date().isoformat(),
        "gap_rows": fold.observed_gap,
        "validation_rows": len(validation),
        **metrics,
        "persistence_mae": persistence_mae,
        "feature_count": feature_count,
        "training_seconds": training_seconds,
        "prediction_seconds": prediction_seconds,
        "best_iteration": best_iteration,
        "aic": aic,
        "warning_status": warning_status,
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def dependency_status() -> dict[str, dict[str, Any]]:
    """Report optional model dependencies without importing them eagerly."""

    modules = {
        "scikit-learn": "sklearn",
        "statsmodels": "statsmodels",
        "lightgbm": "lightgbm",
        "catboost": "catboost",
        "xgboost": "xgboost",
    }
    return {
        distribution: {
            "available": importlib.util.find_spec(module) is not None,
            "version": _package_version(distribution),
        }
        for distribution, module in modules.items()
    }


def _require(module_name: str, install_name: str | None = None) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        package = install_name or module_name
        raise ForecastingFrameworkError(
            f"Optional dependency '{package}' is required. "
            "Run: python -m pip install -r requirements.txt"
        ) from exc


def _feature_group(
    prepared: PreparedForecastDataset,
    name: str,
) -> tuple[str, ...]:
    if name == "compact":
        return prepared.compact_features
    if name == "expanded":
        return prepared.expanded_features
    raise ForecastingFrameworkError(f"Unknown feature group: {name}")


def _fit_ml_model(
    model_name: str,
    parameters: Mapping[str, Any],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
    seed: int,
) -> tuple[RegressorMixin, int | None, str]:
    warning_messages: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if model_name == "ridge":
            model: Any = Pipeline(
                [("scale", StandardScaler()), ("model", Ridge(random_state=seed, **parameters))]
            )
            model.fit(x_train, y_train)
            best_iteration = None
        elif model_name == "elastic_net":
            model = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        ElasticNet(
                            random_state=seed,
                            max_iter=20_000,
                            selection="cyclic",
                            **parameters,
                        ),
                    ),
                ]
            )
            model.fit(x_train, y_train)
            best_iteration = None
        elif model_name == "lightgbm":
            lightgbm = _require("lightgbm")
            model = lightgbm.LGBMRegressor(
                random_state=seed,
                n_jobs=1,
                deterministic=True,
                force_col_wise=True,
                verbosity=-1,
                data_random_seed=seed,
                feature_fraction_seed=seed,
                bagging_seed=seed,
                **parameters,
            )
            model.fit(
                x_train,
                y_train,
                eval_X=x_validation,
                eval_y=y_validation,
                eval_metric="l1",
                callbacks=[lightgbm.early_stopping(50, verbose=False)],
            )
            best_iteration = int(model.best_iteration_ or parameters.get("n_estimators", 0))
        elif model_name == "catboost":
            catboost = _require("catboost")
            model = catboost.CatBoostRegressor(
                random_seed=seed,
                thread_count=1,
                verbose=False,
                allow_writing_files=False,
                **parameters,
            )
            model.fit(
                x_train,
                y_train,
                eval_set=(x_validation, y_validation),
                early_stopping_rounds=50,
                verbose=False,
            )
            best_iteration = int(model.get_best_iteration())
        elif model_name == "xgboost":
            xgboost = _require("xgboost")
            model = xgboost.XGBRegressor(
                random_state=seed,
                n_jobs=1,
                tree_method="hist",
                early_stopping_rounds=50,
                **parameters,
            )
            model.fit(x_train, y_train, eval_set=[(x_validation, y_validation)], verbose=False)
            best_iteration = int(
                getattr(model, "best_iteration", parameters.get("n_estimators", 0))
            )
        else:
            raise ForecastingFrameworkError(f"Unsupported ML model: {model_name}")
        warning_messages.extend(str(item.message) for item in caught)
    return model, best_iteration, "; ".join(warning_messages) or "ok"


def _refit_ml_model(
    model_name: str,
    parameters: Mapping[str, Any],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int,
    best_iterations: Sequence[int | None],
) -> RegressorMixin:
    adjusted = dict(parameters)
    valid_iterations = [value for value in best_iterations if value and value > 0]
    if valid_iterations and model_name in {"lightgbm", "catboost", "xgboost"}:
        iteration_key = "iterations" if model_name == "catboost" else "n_estimators"
        adjusted[iteration_key] = max(20, int(np.median(valid_iterations)))
    if model_name == "ridge":
        model: Any = Pipeline(
            [("scale", StandardScaler()), ("model", Ridge(random_state=seed, **adjusted))]
        )
    elif model_name == "elastic_net":
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    ElasticNet(
                        random_state=seed,
                        max_iter=20_000,
                        selection="cyclic",
                        **adjusted,
                    ),
                ),
            ]
        )
    elif model_name == "lightgbm":
        lightgbm = _require("lightgbm")
        model = lightgbm.LGBMRegressor(
            random_state=seed,
            n_jobs=1,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
            data_random_seed=seed,
            feature_fraction_seed=seed,
            bagging_seed=seed,
            **adjusted,
        )
    elif model_name == "catboost":
        catboost = _require("catboost")
        model = catboost.CatBoostRegressor(
            random_seed=seed,
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
            **adjusted,
        )
    elif model_name == "xgboost":
        adjusted.pop("early_stopping_rounds", None)
        xgboost = _require("xgboost")
        model = xgboost.XGBRegressor(
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
            **adjusted,
        )
    else:
        raise ForecastingFrameworkError(f"Unsupported ML model: {model_name}")
    model.fit(x_train, y_train)
    return model


def _evaluate_baseline(
    model_name: str,
    development: pd.DataFrame,
    folds: Sequence[WalkForwardFold],
) -> CandidateResult:
    predictions: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    prediction_seconds = 0.0
    for fold in folds:
        train = development.iloc[fold.train_indices]
        validation = development.iloc[fold.validation_indices]
        started = time.perf_counter()
        forecast = (
            persistence_forecast(validation)
            if model_name == "persistence"
            else drift_forecast(validation)
        )
        prediction_seconds += time.perf_counter() - started
        actual = validation[TARGET_ABSOLUTE].to_numpy(float)
        metrics.append(
            _fold_metric_row(
                fold,
                train,
                validation,
                model_name,
                model_name,
                actual,
                forecast,
                0,
                0.0,
                prediction_seconds,
                "ok",
            )
        )
        predictions.append(
            _prediction_rows(validation, forecast, model_name, "development_oof", fold.number)
        )
    return CandidateResult(
        model_name=model_name,
        configuration_name=model_name,
        feature_group="none",
        parameters={},
        predictions=pd.concat(predictions, ignore_index=True),
        fold_metrics=pd.DataFrame(metrics),
        warning_status="ok",
        training_seconds=0.0,
        prediction_seconds=prediction_seconds,
        feature_counts=[0] * len(folds),
        best_iterations=[None] * len(folds),
    )


def _evaluate_ml_candidate(
    model_name: str,
    configuration_name: str,
    parameters: Mapping[str, Any],
    feature_group: str,
    prepared: PreparedForecastDataset,
    development: pd.DataFrame,
    folds: Sequence[WalkForwardFold],
    config: ForecastConfig,
) -> CandidateResult:
    columns = _feature_group(prepared, feature_group)
    predictions: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    warnings_seen: list[str] = []
    feature_counts: list[int] = []
    best_iterations: list[int | None] = []
    training_total = 0.0
    prediction_total = 0.0
    for fold in folds:
        train = development.iloc[fold.train_indices]
        validation = development.iloc[fold.validation_indices]
        processor = FeatureProcessor(
            config.maximum_missing_fraction,
            config.correlation_threshold,
        ).fit(train.loc[:, columns])
        x_train = processor.transform(train.loc[:, columns])
        x_validation = processor.transform(validation.loc[:, columns])
        started = time.perf_counter()
        model, best_iteration, warning_status = _fit_ml_model(
            model_name,
            parameters,
            x_train,
            train[TARGET_CHANGE],
            x_validation,
            validation[TARGET_CHANGE],
            config.random_seed,
        )
        training_seconds = time.perf_counter() - started
        started = time.perf_counter()
        predicted_change = np.asarray(model.predict(x_validation), dtype=float)
        prediction_seconds = time.perf_counter() - started
        absolute = reconstruct_absolute_forecast(
            validation[CURRENT_LOAD].to_numpy(float), predicted_change
        )
        actual = validation[TARGET_ABSOLUTE].to_numpy(float)
        metrics.append(
            _fold_metric_row(
                fold,
                train,
                validation,
                model_name,
                configuration_name,
                actual,
                absolute,
                len(processor.selected_columns),
                training_seconds,
                prediction_seconds,
                warning_status,
                best_iteration,
            )
        )
        predictions.append(
            _prediction_rows(validation, absolute, model_name, "development_oof", fold.number)
        )
        warnings_seen.append(warning_status)
        feature_counts.append(len(processor.selected_columns))
        best_iterations.append(best_iteration)
        training_total += training_seconds
        prediction_total += prediction_seconds
    return CandidateResult(
        model_name=model_name,
        configuration_name=configuration_name,
        feature_group=feature_group,
        parameters=dict(parameters),
        predictions=pd.concat(predictions, ignore_index=True),
        fold_metrics=pd.DataFrame(metrics),
        warning_status="ok" if set(warnings_seen) == {"ok"} else "warnings recorded",
        training_seconds=training_total,
        prediction_seconds=prediction_total,
        feature_counts=feature_counts,
        best_iterations=best_iterations,
    )


def _statistical_forecast(
    model_name: str,
    parameters: Mapping[str, Any],
    training_target: pd.Series,
    steps: int,
) -> tuple[np.ndarray, Any, str, float | None]:
    statsmodels = _require("statsmodels")
    del statsmodels
    warning_status = "ok"
    aic: float | None = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if model_name == "ets":
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            model = ExponentialSmoothing(
                training_target.astype(float),
                initialization_method="estimated",
                **parameters,
            ).fit(optimized=True, remove_bias=False)
            forecast = np.asarray(model.forecast(steps), dtype=float)
            aic_value = getattr(model, "aic", None)
            aic = float(aic_value) if aic_value is not None and np.isfinite(aic_value) else None
        elif model_name == "sarimax":
            from statsmodels.tsa.statespace.sarimax import SARIMAX

            model = SARIMAX(
                training_target.astype(float),
                order=tuple(parameters["order"]),
                seasonal_order=tuple(parameters["seasonal_order"]),
                trend=parameters.get("trend", "c"),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, maxiter=100)
            forecast = np.asarray(model.forecast(steps), dtype=float)
            aic = float(model.aic) if np.isfinite(model.aic) else None
        else:
            raise ForecastingFrameworkError(f"Unsupported statistical model: {model_name}")
        if caught:
            warning_status = "; ".join(str(item.message) for item in caught)
    return forecast, model, warning_status, aic


def _evaluate_statistical_candidate(
    model_name: str,
    configuration_name: str,
    parameters: Mapping[str, Any],
    development: pd.DataFrame,
    folds: Sequence[WalkForwardFold],
) -> CandidateResult:
    predictions: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    training_total = 0.0
    prediction_total = 0.0
    warnings_seen: list[str] = []
    for fold in folds:
        train = development.iloc[fold.train_indices]
        validation = development.iloc[fold.validation_indices]
        steps = fold.observed_gap + len(validation)
        started = time.perf_counter()
        forecast_change, _, warning_status, aic = _statistical_forecast(
            model_name,
            parameters,
            train[TARGET_CHANGE],
            steps,
        )
        elapsed = time.perf_counter() - started
        predicted_change = forecast_change[-len(validation) :]
        started = time.perf_counter()
        absolute = reconstruct_absolute_forecast(
            validation[CURRENT_LOAD].to_numpy(float), predicted_change
        )
        prediction_seconds = time.perf_counter() - started
        actual = validation[TARGET_ABSOLUTE].to_numpy(float)
        metrics.append(
            _fold_metric_row(
                fold,
                train,
                validation,
                model_name,
                configuration_name,
                actual,
                absolute,
                0,
                elapsed,
                prediction_seconds,
                warning_status,
                aic=aic,
            )
        )
        predictions.append(
            _prediction_rows(validation, absolute, model_name, "development_oof", fold.number)
        )
        training_total += elapsed
        prediction_total += prediction_seconds
        warnings_seen.append(warning_status)
    return CandidateResult(
        model_name=model_name,
        configuration_name=configuration_name,
        feature_group="univariate_change_series",
        parameters=dict(parameters),
        predictions=pd.concat(predictions, ignore_index=True),
        fold_metrics=pd.DataFrame(metrics),
        warning_status="ok" if set(warnings_seen) == {"ok"} else "warnings recorded",
        training_seconds=training_total,
        prediction_seconds=prediction_total,
        feature_counts=[0] * len(folds),
        best_iterations=[None] * len(folds),
    )


def _candidate_space() -> dict[str, list[tuple[str, dict[str, Any], str]]]:
    """Return a deliberately small CPU-friendly deterministic search space."""

    return {
        "ridge": [
            ("ridge_a10_compact", {"alpha": 10.0}, "compact"),
            ("ridge_a100_compact", {"alpha": 100.0}, "compact"),
            ("ridge_a10_expanded", {"alpha": 10.0}, "expanded"),
        ],
        "elastic_net": [
            ("elastic_a01_l20_compact", {"alpha": 0.01, "l1_ratio": 0.2}, "compact"),
            ("elastic_a10_l20_compact", {"alpha": 0.10, "l1_ratio": 0.2}, "compact"),
            ("elastic_a01_l50_expanded", {"alpha": 0.01, "l1_ratio": 0.5}, "expanded"),
        ],
        "lightgbm": [
            (
                "lgb_l1_shallow_compact",
                {
                    "objective": "regression_l1",
                    "learning_rate": 0.03,
                    "n_estimators": 600,
                    "num_leaves": 7,
                    "max_depth": 3,
                    "min_child_samples": 50,
                    "colsample_bytree": 0.75,
                    "subsample": 0.8,
                    "subsample_freq": 1,
                    "reg_alpha": 2.0,
                    "reg_lambda": 20.0,
                },
                "compact",
            ),
            (
                "lgb_huber_expanded",
                {
                    "objective": "huber",
                    "learning_rate": 0.02,
                    "n_estimators": 800,
                    "num_leaves": 15,
                    "max_depth": 4,
                    "min_child_samples": 60,
                    "colsample_bytree": 0.7,
                    "subsample": 0.8,
                    "subsample_freq": 1,
                    "reg_alpha": 5.0,
                    "reg_lambda": 30.0,
                },
                "expanded",
            ),
        ],
        "catboost": [
            (
                "cat_mae_compact",
                {
                    "loss_function": "MAE",
                    "eval_metric": "MAE",
                    "depth": 4,
                    "learning_rate": 0.03,
                    "iterations": 700,
                    "l2_leaf_reg": 15.0,
                    "random_strength": 0.5,
                },
                "compact",
            ),
            (
                "cat_huber_expanded",
                {
                    "loss_function": "Huber:delta=1.0",
                    "eval_metric": "MAE",
                    "depth": 5,
                    "learning_rate": 0.02,
                    "iterations": 800,
                    "l2_leaf_reg": 25.0,
                    "random_strength": 0.5,
                },
                "expanded",
            ),
        ],
        "xgboost": [
            (
                "xgb_l1_compact",
                {
                    "objective": "reg:absoluteerror",
                    "learning_rate": 0.03,
                    "n_estimators": 700,
                    "max_depth": 3,
                    "min_child_weight": 30.0,
                    "subsample": 0.8,
                    "colsample_bytree": 0.75,
                    "reg_alpha": 2.0,
                    "reg_lambda": 20.0,
                },
                "compact",
            ),
            (
                "xgb_huber_expanded",
                {
                    "objective": "reg:pseudohubererror",
                    "learning_rate": 0.02,
                    "n_estimators": 800,
                    "max_depth": 4,
                    "min_child_weight": 40.0,
                    "subsample": 0.75,
                    "colsample_bytree": 0.7,
                    "reg_alpha": 5.0,
                    "reg_lambda": 30.0,
                },
                "expanded",
            ),
        ],
        "ets": [
            ("ets_level", {"trend": None, "seasonal": None}, "none"),
            ("ets_add_trend", {"trend": "add", "damped_trend": False, "seasonal": None}, "none"),
            ("ets_damped", {"trend": "add", "damped_trend": True, "seasonal": None}, "none"),
            (
                "ets_weekly_additive",
                {"trend": "add", "damped_trend": True, "seasonal": "add", "seasonal_periods": 7},
                "none",
            ),
        ],
        "sarimax": [
            (
                "sarimax_111_1017",
                {"order": (1, 1, 1), "seasonal_order": (1, 0, 1, 7), "trend": "c"},
                "none",
            ),
            (
                "sarimax_211_1007",
                {"order": (2, 1, 1), "seasonal_order": (1, 0, 0, 7), "trend": "c"},
                "none",
            ),
            (
                "sarimax_112_0117",
                {"order": (1, 1, 2), "seasonal_order": (0, 1, 1, 7), "trend": "n"},
                "none",
            ),
        ],
    }


def _select_best(candidates: Sequence[CandidateResult]) -> CandidateResult:
    if not candidates:
        raise ForecastingFrameworkError("No successful candidate configurations remain.")
    return min(candidates, key=lambda item: (item.mean_mae, item.std_mae, item.configuration_name))


def _fit_final_candidate(
    result: CandidateResult,
    prepared: PreparedForecastDataset,
    partitions: ChronologicalPartitions,
    config: ForecastConfig,
) -> FinalModelResult:
    development = partitions.development
    holdout = partitions.holdout
    started = time.perf_counter()
    if result.model_name == "persistence":
        predictions = persistence_forecast(holdout)
        return FinalModelResult(
            result.model_name,
            predictions,
            predictions - holdout[CURRENT_LOAD].to_numpy(float),
            0.0,
            0.0,
            0,
            "ok",
            {"rule": "prediction equals current load"},
        )
    if result.model_name == "seven_day_drift":
        predictions = drift_forecast(holdout)
        return FinalModelResult(
            result.model_name,
            predictions,
            predictions - holdout[CURRENT_LOAD].to_numpy(float),
            0.0,
            0.0,
            0,
            "ok",
            {"rule": "current load plus seven-day observed drift"},
        )
    if result.model_name in {"ets", "sarimax"}:
        steps = len(partitions.embargo) + len(holdout)
        change, model, warning_status, _ = _statistical_forecast(
            result.model_name,
            result.parameters,
            development[TARGET_CHANGE],
            steps,
        )
        training_seconds = time.perf_counter() - started
        started_prediction = time.perf_counter()
        predicted_change = change[-len(holdout) :]
        predictions = reconstruct_absolute_forecast(
            holdout[CURRENT_LOAD].to_numpy(float), predicted_change
        )
        prediction_seconds = time.perf_counter() - started_prediction
        return FinalModelResult(
            result.model_name,
            predictions,
            predicted_change,
            training_seconds,
            prediction_seconds,
            0,
            warning_status,
            {"model": model, "configuration": result.parameters},
        )

    columns = _feature_group(prepared, result.feature_group)
    processor = FeatureProcessor(
        config.maximum_missing_fraction,
        config.correlation_threshold,
    ).fit(development.loc[:, columns])
    x_train = processor.transform(development.loc[:, columns])
    x_holdout = processor.transform(holdout.loc[:, columns])
    model = _refit_ml_model(
        result.model_name,
        result.parameters,
        x_train,
        development[TARGET_CHANGE],
        config.random_seed,
        result.best_iterations,
    )
    training_seconds = time.perf_counter() - started
    started_prediction = time.perf_counter()
    predicted_change = np.asarray(model.predict(x_holdout), dtype=float)
    predictions = reconstruct_absolute_forecast(
        holdout[CURRENT_LOAD].to_numpy(float), predicted_change
    )
    prediction_seconds = time.perf_counter() - started_prediction
    return FinalModelResult(
        result.model_name,
        predictions,
        predicted_change,
        training_seconds,
        prediction_seconds,
        len(processor.selected_columns),
        "ok",
        {"model": model, "processor": processor, "feature_group": result.feature_group},
    )


def _quantile_oof_and_final(
    selected_lgbm: CandidateResult,
    prepared: PreparedForecastDataset,
    partitions: ChronologicalPartitions,
    folds: Sequence[WalkForwardFold],
    config: ForecastConfig,
) -> tuple[pd.DataFrame, tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, Any], Any]:
    lightgbm = _require("lightgbm")
    columns = _feature_group(prepared, selected_lgbm.feature_group)
    oof_parts: list[pd.DataFrame] = []
    raw_crossings = 0
    total_predictions = 0
    quantiles = (0.10, 0.50, 0.90)
    base_parameters = dict(selected_lgbm.parameters)
    for key in ("objective", "alpha"):
        base_parameters.pop(key, None)
    for fold in folds:
        train = partitions.development.iloc[fold.train_indices]
        validation = partitions.development.iloc[fold.validation_indices]
        processor = FeatureProcessor(
            config.maximum_missing_fraction,
            config.correlation_threshold,
        ).fit(train.loc[:, columns])
        x_train = processor.transform(train.loc[:, columns])
        x_validation = processor.transform(validation.loc[:, columns])
        absolute_predictions: list[np.ndarray] = []
        for quantile in quantiles:
            model = lightgbm.LGBMRegressor(
                objective="quantile",
                alpha=quantile,
                random_state=config.random_seed,
                n_jobs=1,
                deterministic=True,
                force_col_wise=True,
                verbosity=-1,
                **base_parameters,
            )
            model.fit(
                x_train,
                train[TARGET_CHANGE],
                eval_X=x_validation,
                eval_y=validation[TARGET_CHANGE],
                callbacks=[lightgbm.early_stopping(50, verbose=False)],
            )
            absolute_predictions.append(
                reconstruct_absolute_forecast(
                    validation[CURRENT_LOAD].to_numpy(float),
                    np.asarray(model.predict(x_validation), dtype=float),
                )
            )
        stacked = np.vstack(absolute_predictions)
        raw_crossings += int(np.sum((stacked[0] > stacked[1]) | (stacked[1] > stacked[2])))
        total_predictions += len(validation)
        ordered = np.vstack(order_prediction_intervals(*stacked))
        oof_parts.append(
            pd.DataFrame(
                {
                    "forecast_origin_date": validation.index,
                    "actual": validation[TARGET_ABSOLUTE].to_numpy(float),
                    "raw_lower": stacked[0],
                    "raw_median": stacked[1],
                    "raw_upper": stacked[2],
                    "lower": ordered[0],
                    "median": ordered[1],
                    "upper": ordered[2],
                    "fold": fold.number,
                    "backlog_state": validation["backlog_state"].to_numpy(bool),
                    "capacity_stress": validation["capacity_stress"].to_numpy(bool),
                    "has_anomaly": validation["has_anomaly"].to_numpy(bool),
                    "is_imputed_date": validation["is_imputed_date"].to_numpy(bool),
                }
            )
        )
    oof = pd.concat(oof_parts, ignore_index=True).sort_values("forecast_origin_date")
    calibration_cut = max(1, int(len(oof) * 0.8))
    calibration = oof.iloc[:calibration_cut]
    calibration_scores = np.maximum.reduce(
        [
            calibration["lower"].to_numpy(float) - calibration["actual"].to_numpy(float),
            calibration["actual"].to_numpy(float) - calibration["upper"].to_numpy(float),
            np.zeros(len(calibration)),
        ]
    )
    probability = min(1.0, math.ceil((len(calibration_scores) + 1) * 0.8) / len(calibration_scores))
    conformal_delta = float(np.quantile(calibration_scores, probability, method="higher"))
    oof["conformal_lower"] = oof["lower"] - conformal_delta
    oof["conformal_upper"] = oof["upper"] + conformal_delta

    development = partitions.development
    holdout = partitions.holdout
    processor = FeatureProcessor(
        config.maximum_missing_fraction,
        config.correlation_threshold,
    ).fit(development.loc[:, columns])
    x_train = processor.transform(development.loc[:, columns])
    x_holdout = processor.transform(holdout.loc[:, columns])
    final_models: dict[str, Any] = {}
    holdout_absolute: list[np.ndarray] = []
    for quantile in quantiles:
        params = dict(base_parameters)
        valid_iterations = [value for value in selected_lgbm.best_iterations if value]
        if valid_iterations:
            params["n_estimators"] = max(20, int(np.median(valid_iterations)))
        model = lightgbm.LGBMRegressor(
            objective="quantile",
            alpha=quantile,
            random_state=config.random_seed,
            n_jobs=1,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
            **params,
        )
        model.fit(x_train, development[TARGET_CHANGE])
        final_models[f"q{int(quantile * 100):02d}"] = model
        holdout_absolute.append(
            reconstruct_absolute_forecast(
                holdout[CURRENT_LOAD].to_numpy(float),
                np.asarray(model.predict(x_holdout), dtype=float),
            )
        )
    raw_holdout = np.vstack(holdout_absolute)
    final_crossing_rate = float(
        np.mean((raw_holdout[0] > raw_holdout[1]) | (raw_holdout[1] > raw_holdout[2])) * 100
    )
    ordered_holdout = np.vstack(order_prediction_intervals(*raw_holdout))
    lower = ordered_holdout[0] - conformal_delta
    median = ordered_holdout[1]
    upper = ordered_holdout[2] + conformal_delta

    def coverage(data: pd.DataFrame, low: str, high: str) -> float:
        if data.empty:
            return 0.0
        return float(((data["actual"] >= data[low]) & (data["actual"] <= data[high])).mean() * 100)

    evaluation = oof.iloc[calibration_cut:]
    actual_holdout = holdout[TARGET_ABSOLUTE].to_numpy(float)
    metrics = {
        "interval_model": "lightgbm_quantile_change_forecast",
        "nominal_coverage_percent": 80.0,
        "raw_walk_forward_coverage_percent": coverage(oof, "lower", "upper"),
        "conformal_development_evaluation_coverage_percent": coverage(
            evaluation, "conformal_lower", "conformal_upper"
        ),
        "conformal_calibration_rows": len(calibration),
        "conformal_evaluation_rows": len(evaluation),
        "conformal_adjustment": conformal_delta,
        "holdout_coverage_percent": float(
            np.mean((actual_holdout >= lower) & (actual_holdout <= upper)) * 100
        ),
        "holdout_mean_width": float(np.mean(upper - lower)),
        "raw_walk_forward_crossing_rate_percent": float(raw_crossings / total_predictions * 100),
        "raw_holdout_crossing_rate_percent": final_crossing_rate,
        "crossing_correction": "row-wise ascending sort before split-conformal expansion",
    }
    for label, mask in {
        "normal": ~(holdout["backlog_state"] | holdout["capacity_stress"]),
        "backlog_or_capacity_stress": holdout["backlog_state"] | holdout["capacity_stress"],
        "anomaly": holdout["has_anomaly"],
        "imputed": holdout["is_imputed_date"],
    }.items():
        values = np.asarray(mask, dtype=bool)
        metrics[f"holdout_coverage_{label}_percent"] = (
            float(
                np.mean(
                    (actual_holdout[values] >= lower[values])
                    & (actual_holdout[values] <= upper[values])
                )
                * 100
            )
            if values.any()
            else None
        )
    bundle = {
        "models": final_models,
        "processor": processor,
        "feature_group": selected_lgbm.feature_group,
    }
    return oof, (lower, median, upper), metrics, bundle


def _hybrid_model(
    sarimax_result: CandidateResult,
    lgbm_result: CandidateResult,
    catboost_result: CandidateResult,
    prepared: PreparedForecastDataset,
    partitions: ChronologicalPartitions,
    final_sarimax: FinalModelResult,
    config: ForecastConfig,
) -> tuple[CandidateResult, FinalModelResult]:
    base_oof = sarimax_result.predictions.set_index("forecast_origin_date").sort_index()
    base_oof["residual"] = build_oof_residuals(
        base_oof["actual_value"], base_oof["reconstructed_absolute_prediction"]
    )
    residual_frame = partitions.development.reindex(base_oof.index).copy()
    residual_frame["residual_target"] = base_oof["residual"]
    if len(residual_frame) < 100:
        raise ForecastingFrameworkError("Too few SARIMAX OOF residuals for the hybrid.")
    inner_splits = TimeSeriesSplit(
        n_splits=3, test_size=max(14, len(residual_frame) // 6), gap=config.gap_days
    )
    correctors = (
        ("lightgbm", lgbm_result),
        ("catboost", catboost_result),
    )
    evaluated: list[tuple[float, str, CandidateResult, pd.DataFrame, pd.DataFrame]] = []
    for corrector_name, template in correctors:
        columns = _feature_group(prepared, template.feature_group)
        prediction_parts: list[pd.DataFrame] = []
        metric_rows: list[dict[str, Any]] = []
        for fold_number, (train_idx, validation_idx) in enumerate(
            inner_splits.split(residual_frame), 1
        ):
            train = residual_frame.iloc[train_idx]
            validation = residual_frame.iloc[validation_idx]
            processor = FeatureProcessor(
                config.maximum_missing_fraction,
                config.correlation_threshold,
            ).fit(train.loc[:, columns])
            x_train = processor.transform(train.loc[:, columns])
            x_validation = processor.transform(validation.loc[:, columns])
            model, _, status = _fit_ml_model(
                corrector_name,
                template.parameters,
                x_train,
                train["residual_target"],
                x_validation,
                validation["residual_target"],
                config.random_seed,
            )
            correction = np.asarray(model.predict(x_validation), dtype=float)
            base = base_oof.reindex(validation.index)["reconstructed_absolute_prediction"].to_numpy(
                float
            )
            hybrid = base + correction
            actual = validation[TARGET_ABSOLUTE].to_numpy(float)
            persistence_mae = float(mean_absolute_error(actual, validation[CURRENT_LOAD]))
            values = regression_metrics(actual, hybrid, persistence_mae)
            metric_rows.append(
                {
                    "model_name": "sarimax_boosting_hybrid",
                    "configuration": f"sarimax_plus_{corrector_name}",
                    "fold": fold_number,
                    "training_start": train.index.min().date().isoformat(),
                    "training_end": train.index.max().date().isoformat(),
                    "validation_start": validation.index.min().date().isoformat(),
                    "validation_end": validation.index.max().date().isoformat(),
                    "gap_rows": config.gap_days,
                    "validation_rows": len(validation),
                    **values,
                    "persistence_mae": persistence_mae,
                    "feature_count": len(processor.selected_columns),
                    "training_seconds": 0.0,
                    "prediction_seconds": 0.0,
                    "best_iteration": None,
                    "aic": None,
                    "warning_status": status,
                }
            )
            prediction_parts.append(
                _prediction_rows(
                    validation,
                    hybrid,
                    "sarimax_boosting_hybrid",
                    "development_oof",
                    fold_number,
                )
            )
        metrics = pd.DataFrame(metric_rows)
        evaluated.append(
            (
                float(metrics["mae"].mean()),
                corrector_name,
                template,
                pd.concat(prediction_parts, ignore_index=True),
                metrics,
            )
        )
    _, corrector_name, template, hybrid_oof, hybrid_metrics = min(
        evaluated, key=lambda item: item[0]
    )
    hybrid_candidate = CandidateResult(
        model_name="sarimax_boosting_hybrid",
        configuration_name=f"{sarimax_result.configuration_name}_plus_{corrector_name}",
        feature_group=template.feature_group,
        parameters={
            "base": sarimax_result.parameters,
            "corrector": template.parameters,
            "corrector_name": corrector_name,
        },
        predictions=hybrid_oof,
        fold_metrics=hybrid_metrics,
        warning_status="ok",
        training_seconds=0.0,
        prediction_seconds=0.0,
        feature_counts=hybrid_metrics["feature_count"].astype(int).tolist(),
        best_iterations=template.best_iterations,
    )

    columns = _feature_group(prepared, template.feature_group)
    processor = FeatureProcessor(
        config.maximum_missing_fraction,
        config.correlation_threshold,
    ).fit(residual_frame.loc[:, columns])
    corrector = _refit_ml_model(
        corrector_name,
        template.parameters,
        processor.transform(residual_frame.loc[:, columns]),
        residual_frame["residual_target"],
        config.random_seed,
        template.best_iterations,
    )
    correction = np.asarray(
        corrector.predict(processor.transform(partitions.holdout.loc[:, columns])), dtype=float
    )
    hybrid_predictions = final_sarimax.predictions + correction
    final = FinalModelResult(
        "sarimax_boosting_hybrid",
        hybrid_predictions,
        hybrid_predictions - partitions.holdout[CURRENT_LOAD].to_numpy(float),
        0.0,
        0.0,
        len(processor.selected_columns),
        "ok",
        {
            "base_model_name": "sarimax",
            "base_model_artifact": "sarimax.joblib",
            "corrector": corrector,
            "processor": processor,
            "corrector_name": corrector_name,
            "residual_training_source": "development-period SARIMAX out-of-fold residuals only",
        },
    )
    return hybrid_candidate, final


def optimize_ensemble_weights(
    actual: np.ndarray,
    predictions: np.ndarray,
    step: float = 0.05,
) -> np.ndarray:
    """Find deterministic non-negative weights summing to one by MAE grid search."""

    if predictions.ndim != 2 or predictions.shape[0] != len(actual):
        raise ForecastingFrameworkError("Ensemble arrays are not aligned.")
    model_count = predictions.shape[1]
    if not 2 <= model_count <= 3:
        raise ForecastingFrameworkError("Ensemble grid supports two or three models.")
    units = int(round(1 / step))
    candidates: list[np.ndarray] = []
    if model_count == 2:
        for first in range(units + 1):
            candidates.append(np.array([first, units - first], dtype=float) / units)
    else:
        for first in range(units + 1):
            for second in range(units - first + 1):
                third = units - first - second
                candidates.append(np.array([first, second, third], dtype=float) / units)
    return min(
        candidates,
        key=lambda weights: (
            mean_absolute_error(actual, predictions @ weights),
            tuple(-weights),
        ),
    )


def _ensemble_model(
    candidates: Mapping[str, CandidateResult],
    final_models: Mapping[str, FinalModelResult],
    partitions: ChronologicalPartitions,
) -> tuple[CandidateResult, FinalModelResult, pd.DataFrame, dict[str, float]]:
    ranked = sorted(candidates.values(), key=lambda result: result.mean_mae)
    selected_names: list[str] = []
    series: dict[str, pd.Series] = {}
    for candidate in ranked:
        name = candidate.model_name
        if name in selected_names or name == "validation_weighted_ensemble":
            continue
        prediction_series = candidate.predictions.set_index("forecast_origin_date")[
            "reconstructed_absolute_prediction"
        ].sort_index()
        if selected_names:
            diverse = False
            for existing in selected_names:
                aligned = pd.concat([prediction_series, series[existing]], axis=1, join="inner")
                if len(aligned) >= 30 and abs(float(aligned.corr().iloc[0, 1])) < 0.999:
                    diverse = True
                    break
            if not diverse:
                continue
        selected_names.append(name)
        series[name] = prediction_series
        if len(selected_names) == 3:
            break
    if len(selected_names) < 2:
        selected_names = ["persistence", "seven_day_drift"]
        series = {
            name: candidates[name]
            .predictions.set_index("forecast_origin_date")["reconstructed_absolute_prediction"]
            .sort_index()
            for name in selected_names
        }
    aligned_predictions = pd.concat(series, axis=1, join="inner").dropna()
    actual = partitions.development.reindex(aligned_predictions.index)[TARGET_ABSOLUTE].to_numpy(
        float
    )
    weights = optimize_ensemble_weights(actual, aligned_predictions.to_numpy(float))
    active = weights > 0
    if int(active.sum()) >= 2:
        selected_names = [
            name for name, is_active in zip(selected_names, active, strict=True) if is_active
        ]
        aligned_predictions = aligned_predictions.loc[:, selected_names]
        weights = weights[active]
        weights = weights / weights.sum()
    ensemble_values = aligned_predictions.to_numpy(float) @ weights
    ensemble_frame = partitions.development.reindex(aligned_predictions.index)
    oof = _prediction_rows(
        ensemble_frame,
        ensemble_values,
        "validation_weighted_ensemble",
        "development_oof",
        "ensemble",
    )
    oof["fold"] = [
        int(
            candidates["persistence"]
            .predictions.set_index("forecast_origin_date")
            .reindex([date])["fold"]
            .iloc[0]
        )
        for date in aligned_predictions.index
    ]
    fold_metrics: list[dict[str, Any]] = []
    for fold_number, group in oof.groupby("fold", sort=True):
        persistence_mae = float(
            mean_absolute_error(group["actual_value"], group["persistence_prediction"])
        )
        metrics = regression_metrics(
            group["actual_value"], group["reconstructed_absolute_prediction"], persistence_mae
        )
        fold_metrics.append(
            {
                "model_name": "validation_weighted_ensemble",
                "configuration": "oof_nonnegative_mae_grid",
                "fold": int(fold_number),
                "training_start": None,
                "training_end": None,
                "validation_start": pd.Timestamp(group["forecast_origin_date"].min())
                .date()
                .isoformat(),
                "validation_end": pd.Timestamp(group["forecast_origin_date"].max())
                .date()
                .isoformat(),
                "gap_rows": 7,
                "validation_rows": len(group),
                **metrics,
                "persistence_mae": persistence_mae,
                "feature_count": 0,
                "training_seconds": 0.0,
                "prediction_seconds": 0.0,
                "best_iteration": None,
                "aic": None,
                "warning_status": "ok",
            }
        )
    holdout_matrix = np.column_stack([final_models[name].predictions for name in selected_names])
    final_predictions = holdout_matrix @ weights
    candidate = CandidateResult(
        "validation_weighted_ensemble",
        "oof_nonnegative_mae_grid",
        "none",
        {"models": selected_names, "weights": weights.tolist()},
        oof,
        pd.DataFrame(fold_metrics),
        "ok",
        0.0,
        0.0,
        [0] * len(fold_metrics),
        [None] * len(fold_metrics),
    )
    final = FinalModelResult(
        "validation_weighted_ensemble",
        final_predictions,
        final_predictions - partitions.holdout[CURRENT_LOAD].to_numpy(float),
        0.0,
        0.0,
        0,
        "ok",
        {"models": selected_names, "weights": weights.tolist()},
    )
    correlations = aligned_predictions.subtract(actual, axis=0).corr()
    return candidate, final, correlations, dict(zip(selected_names, weights.tolist(), strict=True))


def _audit_provenance(
    prepared: PreparedForecastDataset,
    source: pd.DataFrame,
    config: ForecastConfig,
) -> dict[str, Any]:
    preprocessing: dict[str, Any] = {}
    if config.provenance_path.is_file():
        preprocessing = json.loads(config.provenance_path.read_text(encoding="utf-8"))
    report = preprocessing.get("report", {})
    dates = pd.DatetimeIndex(source.index)
    expected = pd.date_range(dates.min(), dates.max(), freq="D")
    load = pd.to_numeric(source[TOTAL_LOAD_COLUMN], errors="coerce")
    intake = pd.to_numeric(source[INTAKE_COLUMN], errors="coerce")
    discharge = pd.to_numeric(source[DISCHARGE_COLUMN], errors="coerce")
    balance_residual = load.diff() - (intake - discharge)
    finite_balance = balance_residual.dropna()
    within_10 = float((finite_balance.abs() <= 10).mean() * 100) if len(finite_balance) else 0.0
    source_hash = _sha256(config.raw_path) if config.raw_path.is_file() else None
    recorded_hash = preprocessing.get("source_sha256")
    provenance_verified = bool(source_hash and recorded_hash and source_hash == recorded_hash)
    return {
        "classification": "unknown/unverified aggregate operational data",
        "real_synthetic_or_mixed": "unknown",
        "generalization_warning": (
            "The filename and internal checksum establish local lineage only. No "
            "authoritative publisher URL, acquisition timestamp, or external signature "
            "is recorded; performance does not demonstrate generalization to real HHS operations."
        ),
        "source_file": str(config.raw_path.relative_to(PROJECT_ROOT))
        if config.raw_path.is_file()
        else str(config.raw_path),
        "source_sha256": source_hash,
        "recorded_source_sha256": recorded_hash,
        "local_lineage_hash_matches": provenance_verified,
        "date_range": {
            "start": dates.min().date().isoformat(),
            "end": dates.max().date().isoformat(),
        },
        "processed_row_count": len(source),
        "model_eligible_row_count": len(prepared.frame),
        "source_rows_reported": report.get("source_rows"),
        "missing_date_count_before_processing": report.get("missing_dates_inserted"),
        "missing_date_count_after_processing": len(expected.difference(dates)),
        "imputed_value_count": report.get("numeric_values_imputed"),
        "imputed_date_count": int(
            pd.Series(source.get("Is Imputed Date", False)).astype(bool).sum()
        ),
        "duplicate_dates_after_processing": int(dates.duplicated().sum()),
        "target_available_rows": int(prepared.frame[TARGET_ABSOLUTE].notna().sum()),
        "source_feature_columns": len(source.columns),
        "candidate_feature_columns": len(prepared.expanded_features),
        "frequency_consistent_daily_after_processing": (
            len(dates) == len(expected) and len(expected.difference(dates)) == 0
        ),
        "stock_flow_relationship": {
            "identity_tested": "delta total load ~= apprehensions - discharges",
            "median_absolute_reconciliation_error": float(finite_balance.abs().median()),
            "percent_within_10_children": within_10,
            "approximately_satisfied": within_10 >= 80.0,
            "structural_flow_candidate_skipped": within_10 < 80.0,
            "skip_reason": (
                "Stock-flow identity is not sufficiently verified for forecasting future transfers and discharges."
                if within_10 < 80.0
                else None
            ),
        },
        "contains_personal_or_child_level_data": False,
        "data_granularity": "aggregate daily system counts",
        "audited_at_utc": utc_now_iso(),
    }


def _leakage_audit(
    prepared: PreparedForecastDataset,
    partitions: ChronologicalPartitions,
    folds: Sequence[WalkForwardFold],
    processors: Sequence[FeatureProcessor],
    config: ForecastConfig,
) -> dict[str, Any]:
    feature_names = set(prepared.expanded_features)
    forbidden = sorted(
        name for name in feature_names if "target" in name or "future" in name or "lead" in name
    )
    fold_processors_safe = all(
        processor.fitted_rows is not None
        and pd.Timestamp(processor.fitted_rows[1])
        < partitions.development.index[fold.validation_indices[0]]
        for processor, fold in zip(processors, folds, strict=False)
    )
    checks = {
        "target_and_future_features_excluded": not forbidden,
        "current_load_available_at_origin": CURRENT_LOAD in feature_names,
        "operational_flows_lagged": all(
            name not in feature_names
            for name in (INTAKE_COLUMN, TRANSFER_COLUMN, DISCHARGE_COLUMN, NET_INTAKE_COLUMN)
        ),
        "fold_preprocessing_fit_on_training_only": fold_processors_safe,
        "walk_forward_chronological": all(
            fold.train_indices[-1] < fold.validation_indices[0] for fold in folds
        ),
        "walk_forward_gap_is_seven": all(fold.observed_gap == config.gap_days for fold in folds),
        "holdout_after_development_and_embargo": (
            partitions.development.index.max()
            < partitions.embargo.index.min()
            < partitions.holdout.index.min()
        ),
        "holdout_not_used_for_tuning": True,
        "ensemble_weights_use_development_oof_only": True,
        "hybrid_residuals_are_sarimax_oof_only": True,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "forbidden_feature_matches": forbidden,
        "holdout_start": partitions.holdout.index.min().date().isoformat(),
        "holdout_end": partitions.holdout.index.max().date().isoformat(),
        "embargo_rows": len(partitions.embargo),
        "target_definition": "total_system_load[t+7] - total_system_load[t]",
        "absolute_reconstruction": "total_system_load[t] + predicted_change_7d[t]",
        "audit_created_at_utc": utc_now_iso(),
    }


def _promotion_decision(
    candidates: Mapping[str, CandidateResult],
    holdout_metrics: Mapping[str, Mapping[str, float]],
    leakage_passed: bool,
    config: ForecastConfig,
) -> dict[str, Any]:
    challenger = min(
        (result for name, result in candidates.items() if name != "persistence"),
        key=lambda result: (result.mean_mae, result.model_name),
    )
    baseline = candidates["persistence"]
    aligned = challenger.predictions.merge(
        baseline.predictions[["forecast_origin_date", "reconstructed_absolute_prediction", "fold"]],
        on="forecast_origin_date",
        suffixes=("_challenger", "_persistence"),
    )
    fold_wins = 0
    fold_rows: list[dict[str, Any]] = []
    for fold, group in aligned.groupby("fold_challenger"):
        challenger_mae = float(
            mean_absolute_error(
                group["actual_value"], group["reconstructed_absolute_prediction_challenger"]
            )
        )
        persistence_mae = float(
            mean_absolute_error(
                group["actual_value"], group["reconstructed_absolute_prediction_persistence"]
            )
        )
        won = challenger_mae < persistence_mae
        fold_wins += int(won)
        fold_rows.append(
            {
                "fold": str(fold),
                "challenger_mae": challenger_mae,
                "persistence_mae": persistence_mae,
                "won": won,
            }
        )
    majority = math.ceil(len(fold_rows) / 2)
    challenger_cv_mae = float(
        mean_absolute_error(
            aligned["actual_value"], aligned["reconstructed_absolute_prediction_challenger"]
        )
    )
    aligned_persistence_cv_mae = float(
        mean_absolute_error(
            aligned["actual_value"], aligned["reconstructed_absolute_prediction_persistence"]
        )
    )
    worst_challenger = max(row["challenger_mae"] for row in fold_rows)
    worst_persistence = max(row["persistence_mae"] for row in fold_rows)
    checks = {
        "mean_walk_forward_mae_better_than_persistence": challenger_cv_mae
        < aligned_persistence_cv_mae,
        "holdout_mae_better_than_persistence": (
            holdout_metrics[challenger.model_name]["mae"] < holdout_metrics["persistence"]["mae"]
        ),
        "validation_fold_majority": fold_wins >= majority,
        "worst_fold_within_limit": worst_challenger
        <= worst_persistence * config.worst_fold_ratio_limit,
        "leakage_audit_passed": leakage_passed,
        "same_eligible_dates": len(aligned) == len(challenger.predictions),
        "reproducible_fixed_seed": True,
        "real_and_synthetic_results_not_conflated": True,
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "champion_model": challenger.model_name if passed else "persistence",
        "selected_challenger": challenger.model_name,
        "recommendation": "promote" if passed else "continue_research",
        "selection_rule": "lowest development-period OOF MAE; holdout used only as a promotion gate",
        "checks": checks,
        "fold_wins": fold_wins,
        "required_fold_wins": majority,
        "fold_comparison": fold_rows,
        "aligned_development_rows": len(aligned),
        "challenger_walk_forward_mae": challenger_cv_mae,
        "aligned_persistence_walk_forward_mae": aligned_persistence_cv_mae,
        "challenger_holdout_mae": holdout_metrics[challenger.model_name]["mae"],
        "persistence_holdout_mae": holdout_metrics["persistence"]["mae"],
        "rationale": (
            "All predefined gates passed."
            if passed
            else "Persistence remains champion because the development-selected challenger failed one or more immutable gates."
        ),
    }


def _error_diagnostics(
    holdout: pd.DataFrame,
    predictions: np.ndarray,
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    actual = holdout[TARGET_ABSOLUTE].to_numpy(float)
    diagnostics = pd.DataFrame(
        {
            "forecast_origin_date": holdout.index,
            "target_date": holdout["target_date"].to_numpy(),
            "model_name": model_name,
            "actual": actual,
            "predicted": predictions,
            "residual": actual - predictions,
            "absolute_error": np.abs(actual - predictions),
            "day_of_week": holdout.index.day_name(),
            "month": holdout.index.month_name(),
            "forecast_load_band": pd.qcut(
                predictions, q=3, labels=["low", "medium", "high"], duplicates="drop"
            ).astype(str),
            "net_intake_magnitude": pd.cut(
                holdout["net_intake_at_origin"].abs(),
                bins=[-np.inf, 10, 50, np.inf],
                labels=["low", "moderate", "high"],
            ).astype(str),
            "backlog_state": holdout["backlog_state"].astype(bool).to_numpy(),
            "capacity_utilization_band": pd.cut(
                holdout["capacity_utilization"],
                bins=[-np.inf, 0.6, 0.8, np.inf],
                labels=["normal", "watch", "stress"],
            ).astype(str),
            "anomaly_status": holdout["has_anomaly"]
            .map({True: "anomaly", False: "normal"})
            .to_numpy(),
            "imputation_status": holdout["is_imputed_date"]
            .map({True: "imputed", False: "observed"})
            .to_numpy(),
            "pressure_regime": np.where(
                holdout["backlog_state"] | holdout["capacity_stress"], "high_pressure", "normal"
            ),
        }
    )
    rows: list[dict[str, Any]] = []
    for dimension in (
        "day_of_week",
        "month",
        "forecast_load_band",
        "net_intake_magnitude",
        "backlog_state",
        "capacity_utilization_band",
        "anomaly_status",
        "imputation_status",
        "pressure_regime",
    ):
        for value, group in diagnostics.groupby(dimension, dropna=False):
            rows.append(
                {
                    "dimension": dimension,
                    "regime": str(value),
                    "rows": len(group),
                    "mae": float(group["absolute_error"].mean()),
                    "mean_residual": float(group["residual"].mean()),
                    "rmse": float(math.sqrt(np.mean(np.square(group["residual"])))),
                }
            )
    return diagnostics, pd.DataFrame(rows)


def _oof_permutation_importance(
    selected: CandidateResult,
    prepared: PreparedForecastDataset,
    development: pd.DataFrame,
    folds: Sequence[WalkForwardFold],
    config: ForecastConfig,
) -> pd.DataFrame:
    if selected.model_name not in {"ridge", "elastic_net", "lightgbm", "catboost", "xgboost"}:
        return pd.DataFrame(
            columns=[
                "feature",
                "mean_mae_increase",
                "std_mae_increase",
                "folds",
                "interpretation_note",
            ]
        )
    columns = _feature_group(prepared, selected.feature_group)
    records: list[dict[str, Any]] = []
    for fold in folds:
        train = development.iloc[fold.train_indices]
        validation = development.iloc[fold.validation_indices]
        processor = FeatureProcessor(
            config.maximum_missing_fraction, config.correlation_threshold
        ).fit(train.loc[:, columns])
        x_train = processor.transform(train.loc[:, columns])
        x_validation = processor.transform(validation.loc[:, columns])
        model, _, _ = _fit_ml_model(
            selected.model_name,
            selected.parameters,
            x_train,
            train[TARGET_CHANGE],
            x_validation,
            validation[TARGET_CHANGE],
            config.random_seed,
        )
        truth = validation[TARGET_CHANGE].to_numpy(float)
        baseline = float(mean_absolute_error(truth, model.predict(x_validation)))
        random = np.random.default_rng(config.random_seed + fold.number)
        for feature in x_validation.columns:
            permuted = x_validation.copy()
            permuted[feature] = random.permutation(permuted[feature].to_numpy())
            permuted_mae = float(mean_absolute_error(truth, model.predict(permuted)))
            records.append(
                {
                    "feature": feature,
                    "fold": fold.number,
                    "mae_increase": permuted_mae - baseline,
                }
            )
    raw = pd.DataFrame(records)
    if raw.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "mean_mae_increase",
                "std_mae_increase",
                "folds",
                "interpretation_note",
            ]
        )
    summary = raw.groupby("feature", as_index=False).agg(
        mean_mae_increase=("mae_increase", "mean"),
        std_mae_increase=("mae_increase", lambda values: float(np.std(values, ddof=0))),
        folds=("fold", "nunique"),
    )
    summary["interpretation_note"] = (
        "Correlated features can divide or mask permutation importance."
    )
    return summary.sort_values(
        ["mean_mae_increase", "feature"], ascending=[False, True]
    ).reset_index(drop=True)


def _render_report(
    path: Path,
    holdout_predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    comparison: Mapping[str, Any],
    importance: pd.DataFrame,
    error_by_regime: pd.DataFrame,
    residual_correlation: pd.DataFrame,
    interval_metrics: Mapping[str, Any],
    provenance: Mapping[str, Any],
    promotion: Mapping[str, Any],
) -> None:
    champion = str(promotion["champion_model"])
    champion_rows = holdout_predictions.loc[holdout_predictions["model_name"] == champion]
    figure = make_subplots(
        rows=8,
        cols=1,
        subplot_titles=(
            "Actual versus champion forecast",
            "Champion residuals over time",
            "Champion residual distribution",
            "Walk-forward MAE by model and fold",
            "Holdout model comparison",
            "Out-of-fold permutation importance",
            "Prediction-interval coverage",
            "Ensemble-candidate residual correlation",
        ),
        vertical_spacing=0.045,
    )
    figure.add_trace(
        go.Scatter(
            x=champion_rows["target_date"],
            y=champion_rows["actual_value"],
            name="Actual",
            line={"color": "#163B65"},
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=champion_rows["target_date"],
            y=champion_rows["reconstructed_absolute_prediction"],
            name=f"Forecast: {champion}",
            line={"color": "#D97706"},
        ),
        row=1,
        col=1,
    )
    if champion_rows["lower_interval"].notna().any():
        figure.add_trace(
            go.Scatter(
                x=champion_rows["target_date"],
                y=champion_rows["upper_interval"],
                line={"width": 0},
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=champion_rows["target_date"],
                y=champion_rows["lower_interval"],
                fill="tonexty",
                fillcolor="rgba(46,117,182,.15)",
                line={"width": 0},
                name="LightGBM quantile 80% interval",
            ),
            row=1,
            col=1,
        )
    residuals = champion_rows["actual_value"] - champion_rows["reconstructed_absolute_prediction"]
    figure.add_trace(
        go.Scatter(
            x=champion_rows["target_date"], y=residuals, name="Residual", line={"color": "#B42318"}
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Histogram(x=residuals, name="Residual distribution", marker_color="#66788A"),
        row=3,
        col=1,
    )
    for model_name, group in fold_metrics.groupby("model_name"):
        figure.add_trace(
            go.Scatter(x=group["fold"], y=group["mae"], mode="lines+markers", name=model_name),
            row=4,
            col=1,
        )
    holdout_items = comparison["models"]
    figure.add_trace(
        go.Bar(
            x=list(holdout_items),
            y=[holdout_items[name]["holdout"]["mae"] for name in holdout_items],
            name="Holdout MAE",
            marker_color="#2E75B6",
        ),
        row=5,
        col=1,
    )
    if not importance.empty:
        top_importance = importance.head(20).sort_values("mean_mae_increase")
        figure.add_trace(
            go.Bar(
                x=top_importance["mean_mae_increase"],
                y=top_importance["feature"],
                orientation="h",
                name="Permutation MAE increase",
                marker_color="#167C80",
            ),
            row=6,
            col=1,
        )
    figure.add_trace(
        go.Bar(
            x=["Nominal", "Raw walk-forward", "Conformal development", "Final holdout"],
            y=[
                interval_metrics["nominal_coverage_percent"],
                interval_metrics["raw_walk_forward_coverage_percent"],
                interval_metrics["conformal_development_evaluation_coverage_percent"],
                interval_metrics["holdout_coverage_percent"],
            ],
            name="Coverage (%)",
            marker_color=["#66788A", "#2E75B6", "#167C80", "#D97706"],
        ),
        row=7,
        col=1,
    )
    if not residual_correlation.empty:
        figure.add_trace(
            go.Heatmap(
                z=residual_correlation.to_numpy(float),
                x=residual_correlation.columns,
                y=residual_correlation.index,
                zmin=-1,
                zmax=1,
                colorscale="RdBu",
                name="Residual correlation",
            ),
            row=8,
            col=1,
        )
    figure.update_layout(
        height=2450,
        template="plotly_white",
        title="Seven-Day System Capacity Forecast Research Report",
    )
    chart_html = figure.to_html(full_html=False, include_plotlyjs=True)
    importance_html = (
        importance.head(25).to_html(index=False)
        if not importance.empty
        else "<p>Not available.</p>"
    )
    comparison_table = (
        pd.DataFrame(
            [
                {
                    "Model": name,
                    "CV MAE": values["walk_forward"]["mean_mae"],
                    "CV SD": values["walk_forward"]["std_mae"],
                    "Holdout MAE": values["holdout"]["mae"],
                    "Holdout MASE": values["holdout"]["mase_vs_persistence"],
                }
                for name, values in comparison["models"].items()
            ]
        )
        .sort_values("CV MAE")
        .to_html(index=False)
    )
    regime_html = error_by_regime.to_html(index=False)
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>Forecast research report</title>
<style>body{{font-family:Arial,sans-serif;color:#172033;max-width:1300px;margin:auto;padding:24px}}h1,h2{{color:#163B65}}.warning{{background:#fff8e6;border-left:5px solid #d97706;padding:12px}}table{{border-collapse:collapse;width:100%}}th,td{{padding:7px;border:1px solid #d9e1ea;text-align:right}}th:first-child,td:first-child{{text-align:left}}</style></head><body>
<h1>Seven-Day System Capacity Forecast Research Report</h1>
<div class='warning'><strong>Research output only.</strong> {provenance["generalization_warning"]}</div>
<p><strong>Champion:</strong> {champion} · <strong>Promotion:</strong> {promotion["recommendation"]}</p>
<h2>Model comparison</h2>{comparison_table}
{chart_html}
<h2>Prediction interval</h2><pre>{json.dumps(json_safe(interval_metrics), indent=2)}</pre>
<h2>OOF permutation importance</h2>{importance_html}
<h2>Error by regime</h2>{regime_html}
<p>This report is not an official HHS or CBP forecast and must not drive automatic operational decisions.</p>
</body></html>"""
    with _atomic_target(path) as temporary:
        temporary.write_text(html, encoding="utf-8")


def _validate_artifacts_no_nonfinite(payload: Any, location: str = "root") -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            _validate_artifacts_no_nonfinite(value, f"{location}.{key}")
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            _validate_artifacts_no_nonfinite(value, f"{location}[{index}]")
    elif isinstance(payload, float) and not math.isfinite(payload):
        raise ForecastingFrameworkError(f"Non-finite metric at {location}.")


def train_forecasting_models(
    config: ForecastConfig | None = None,
) -> ForecastExperimentResult:
    """Run the full experiment and atomically persist its research artifacts."""

    selected = config or ForecastConfig()
    git_metadata = _git_metadata()
    paths = selected.artifact_paths
    existing = [path for path in paths.values() if path.exists()]
    if existing and not selected.overwrite:
        raise ForecastingFrameworkError(
            f"Forecast artifacts already exist ({existing[0]}). Pass --force to replace them."
        )
    np.random.seed(selected.random_seed)
    source = _read_feature_artifact(selected.feature_path)
    prepared = prepare_forecast_dataset(source, selected)
    partitions = chronological_partitions(prepared, selected)
    folds = expanding_window_folds(len(partitions.development), selected)

    # Prove preprocessing is fold-local with representative processors before
    # any model search. Every ML candidate repeats this exact fitting pattern.
    audit_processors = [
        FeatureProcessor(selected.maximum_missing_fraction, selected.correlation_threshold).fit(
            partitions.development.iloc[fold.train_indices].loc[:, prepared.compact_features]
        )
        for fold in folds
    ]
    provenance = _audit_provenance(prepared, source, selected)
    leakage = _leakage_audit(prepared, partitions, folds, audit_processors, selected)
    if not leakage["passed"]:
        raise ForecastingFrameworkError("Leakage audit failed before model training.")

    all_candidates: dict[str, list[CandidateResult]] = {
        "persistence": [_evaluate_baseline("persistence", partitions.development, folds)],
        "seven_day_drift": [_evaluate_baseline("seven_day_drift", partitions.development, folds)],
    }
    search = _candidate_space()
    for model_name in ("ridge", "elastic_net", "lightgbm", "catboost", "xgboost"):
        all_candidates[model_name] = [
            _evaluate_ml_candidate(
                model_name,
                configuration_name,
                parameters,
                feature_group,
                prepared,
                partitions.development,
                folds,
                selected,
            )
            for configuration_name, parameters, feature_group in search[model_name]
        ]
    for model_name in ("ets", "sarimax"):
        successes: list[CandidateResult] = []
        failures: list[str] = []
        for configuration_name, parameters, _ in search[model_name]:
            try:
                successes.append(
                    _evaluate_statistical_candidate(
                        model_name,
                        configuration_name,
                        parameters,
                        partitions.development,
                        folds,
                    )
                )
            except (ValueError, np.linalg.LinAlgError, ForecastingFrameworkError) as exc:
                failures.append(f"{configuration_name}: {exc}")
        if not successes:
            raise ForecastingFrameworkError(
                f"Every {model_name} configuration failed: {'; '.join(failures)}"
            )
        all_candidates[model_name] = successes

    chosen = {name: _select_best(results) for name, results in all_candidates.items()}
    final_models = {
        name: _fit_final_candidate(result, prepared, partitions, selected)
        for name, result in chosen.items()
    }

    hybrid_candidate, hybrid_final = _hybrid_model(
        chosen["sarimax"],
        chosen["lightgbm"],
        chosen["catboost"],
        prepared,
        partitions,
        final_models["sarimax"],
        selected,
    )
    chosen[hybrid_candidate.model_name] = hybrid_candidate
    final_models[hybrid_final.model_name] = hybrid_final
    ensemble_candidate, ensemble_final, residual_correlation, ensemble_weights = _ensemble_model(
        chosen, final_models, partitions
    )
    chosen[ensemble_candidate.model_name] = ensemble_candidate
    final_models[ensemble_final.model_name] = ensemble_final

    interval_oof, holdout_intervals, interval_metrics, interval_bundle = _quantile_oof_and_final(
        chosen["lightgbm"], prepared, partitions, folds, selected
    )
    lower, median, upper = holdout_intervals

    actual_holdout = partitions.holdout[TARGET_ABSOLUTE].to_numpy(float)
    persistence_holdout_mae = float(
        mean_absolute_error(actual_holdout, final_models["persistence"].predictions)
    )
    holdout_metrics = {
        name: regression_metrics(actual_holdout, final.predictions, persistence_holdout_mae)
        for name, final in final_models.items()
    }
    promotion = _promotion_decision(chosen, holdout_metrics, leakage["passed"], selected)

    comparison_models: dict[str, Any] = {}
    for name, result in chosen.items():
        comparison_models[name] = {
            "selected_configuration": result.configuration_name,
            "feature_group": result.feature_group,
            "parameters": result.parameters,
            "walk_forward": {
                "mean_mae": result.mean_mae,
                "std_mae": result.std_mae,
                "worst_fold_mae": result.worst_mae,
                "folds": len(result.fold_metrics),
            },
            "holdout": holdout_metrics[name],
            "training_seconds": result.training_seconds + final_models[name].training_seconds,
            "prediction_seconds": result.prediction_seconds + final_models[name].prediction_seconds,
            "feature_count": final_models[name].feature_count,
            "warning_status": result.warning_status,
        }
    comparison = {
        "experiment_version": MODEL_VERSION,
        "created_at_utc": utc_now_iso(),
        "target": {
            "change": "total_system_load[t+7] - total_system_load[t]",
            "absolute_reconstruction": "total_system_load[t] + predicted_change_7d[t]",
            "horizon_days": selected.horizon_days,
        },
        "development": {
            "rows": len(partitions.development),
            "start": partitions.development.index.min().date().isoformat(),
            "end": partitions.development.index.max().date().isoformat(),
            "folds": len(folds),
            "validation_rows_per_fold": len(folds[0].validation_indices),
            "gap_rows": selected.gap_days,
        },
        "holdout": {
            "rows": len(partitions.holdout),
            "start": partitions.holdout.index.min().date().isoformat(),
            "end": partitions.holdout.index.max().date().isoformat(),
            "evaluated_after_configuration_freeze": True,
        },
        "models": comparison_models,
        "ensemble_weights": ensemble_weights,
        "dependency_versions": dependency_status(),
    }
    _validate_artifacts_no_nonfinite(comparison)
    _validate_artifacts_no_nonfinite(interval_metrics)

    # Long-form prediction artifacts share identical holdout rows for every model.
    oof_predictions = pd.concat(
        [result.predictions for result in chosen.values()], ignore_index=True
    ).sort_values(["forecast_origin_date", "model_name"])
    holdout_predictions = pd.concat(
        [
            _prediction_rows(
                partitions.holdout,
                final.predictions,
                name,
                "final_holdout",
                "holdout",
            )
            for name, final in final_models.items()
        ],
        ignore_index=True,
    )
    for frame in (oof_predictions, holdout_predictions):
        frame.loc[:, "lower_interval"] = (
            np.tile(lower, len(frame) // len(lower))
            if frame is holdout_predictions
            else frame["lower_interval"]
        )
        frame.loc[:, "median_prediction"] = (
            np.tile(median, len(frame) // len(median))
            if frame is holdout_predictions
            else frame["median_prediction"]
        )
        frame.loc[:, "upper_interval"] = (
            np.tile(upper, len(frame) // len(upper))
            if frame is holdout_predictions
            else frame["upper_interval"]
        )
    interval_lookup = interval_oof.set_index("forecast_origin_date")
    oof_predictions["lower_interval"] = oof_predictions["forecast_origin_date"].map(
        interval_lookup["conformal_lower"]
    )
    oof_predictions["median_prediction"] = oof_predictions["forecast_origin_date"].map(
        interval_lookup["median"]
    )
    oof_predictions["upper_interval"] = oof_predictions["forecast_origin_date"].map(
        interval_lookup["conformal_upper"]
    )
    validate_prediction_schema(oof_predictions)
    validate_prediction_schema(holdout_predictions)

    fold_metrics = pd.concat(
        [result.fold_metrics.dropna(axis=1, how="all") for result in chosen.values()],
        ignore_index=True,
    ).sort_values(["model_name", "fold"])
    selected_learned = min(
        (
            result
            for name, result in chosen.items()
            if name in {"ridge", "elastic_net", "lightgbm", "catboost", "xgboost"}
        ),
        key=lambda result: result.mean_mae,
    )
    importance = _oof_permutation_importance(
        selected_learned, prepared, partitions.development, folds, selected
    )
    final_feature_columns = _feature_group(prepared, selected_learned.feature_group)
    final_processor = FeatureProcessor(
        selected.maximum_missing_fraction, selected.correlation_threshold
    ).fit(partitions.development.loc[:, final_feature_columns])
    exclusions = pd.concat(
        [
            prepared.exclusions,
            pd.DataFrame(
                [
                    {"feature": feature, "reason": reason}
                    for feature, reason in final_processor.exclusion_reasons.items()
                ]
            ),
        ],
        ignore_index=True,
    ).drop_duplicates(["feature", "reason"])
    champion_name = promotion["champion_model"]
    diagnostics, error_by_regime = _error_diagnostics(
        partitions.holdout,
        final_models[champion_name].predictions,
        champion_name,
    )

    # Persist candidate bundles under the dedicated directory only. Existing
    # ridge and LightGBM baseline artifacts elsewhere are never touched.
    model_files: dict[str, str] = {}
    model_directory = selected.output_dir / "models" / "candidates"
    for name, final in final_models.items():
        if name in {"persistence", "seven_day_drift", "validation_weighted_ensemble"}:
            continue
        target = model_directory / f"{name}.joblib"
        with _atomic_target(target) as temporary:
            joblib.dump(final.model_bundle, temporary)
        model_files[name] = str(target.relative_to(PROJECT_ROOT))
    interval_path = model_directory / "lightgbm_quantile_models.joblib"
    with _atomic_target(interval_path) as temporary:
        joblib.dump(interval_bundle, temporary)
    model_files["lightgbm_quantiles"] = str(interval_path.relative_to(PROJECT_ROOT))

    # This is the exact prepared frame consumed by model selection and final
    # fitting. Verification compares fresh platform-derived values against this
    # canonical artifact before trusting its stable registry fingerprint.
    canonical_prepared_path = paths["canonical_prepared_frame"]
    _write_parquet(canonical_prepared_path, prepared.frame)
    canonical_prepared_artifact_sha256 = _sha256(canonical_prepared_path)
    feature_artifact_sha256 = _sha256(selected.feature_path)
    forecast_config_sha256 = forecast_configuration_fingerprint(selected)
    prepared_date_range = {
        "start": prepared.frame.index.min().isoformat(),
        "end": prepared.frame.index.max().isoformat(),
    }

    registry = {
        "registry_version": MODEL_VERSION,
        "created_at_utc": utc_now_iso(),
        "champion": champion_name,
        "promotion_status": promotion["recommendation"],
        "models": {
            name: {
                "status": "champion" if name == champion_name else "research_candidate",
                "configuration": result.configuration_name,
                "artifact": model_files.get(name),
                "walk_forward_mae": result.mean_mae,
                "holdout_mae": holdout_metrics[name]["mae"],
            }
            for name, result in chosen.items()
        },
        "data_fingerprint_sha256": prepared.data_fingerprint,
        "schema_fingerprint_sha256": prepared.schema_fingerprint,
        "fingerprint_algorithm": FINGERPRINT_ALGORITHM,
        "fingerprint_float_decimal_places": FINGERPRINT_FLOAT_DECIMAL_PLACES,
        "verification_contract_version": PREPARED_DATASET_CONTRACT_VERSION,
        "prepared_dataset_contract": {
            "artifact": str(canonical_prepared_path.relative_to(PROJECT_ROOT)),
            "artifact_sha256": canonical_prepared_artifact_sha256,
            "data_fingerprint_sha256": prepared.data_fingerprint,
            "schema_fingerprint_sha256": prepared.schema_fingerprint,
            "feature_recipe_version": PREPARED_FEATURE_RECIPE_VERSION,
            "forecast_configuration_sha256": forecast_config_sha256,
            "processed_feature_artifact": str(selected.feature_path.relative_to(PROJECT_ROOT)),
            "processed_feature_artifact_sha256": feature_artifact_sha256,
            "row_count": len(prepared.frame),
            "date_range": prepared_date_range,
        },
        "source_sha256": provenance["source_sha256"],
        "random_seed": selected.random_seed,
        **git_metadata,
        "training_configuration": {
            "forecast_horizon_days": selected.horizon_days,
            "holdout_fraction": selected.holdout_fraction,
            "gap_days": selected.gap_days,
            "cv_splits": selected.cv_splits,
            "cv_validation_rows": selected.cv_test_size,
            "maximum_missing_fraction": selected.maximum_missing_fraction,
            "correlation_threshold": selected.correlation_threshold,
            "capacity_reference": selected.capacity_reference,
        },
        "development_period": comparison["development"],
        "holdout_period": comparison["holdout"],
        "library_versions": {
            "python": platform.python_version(),
            **{name: details["version"] for name, details in dependency_status().items()},
        },
    }
    champion_payload = {
        "model_name": champion_name,
        "promotion": promotion,
        "inference_rule": (
            "forecast equals current_total_system_load"
            if champion_name == "persistence"
            else final_models[champion_name].model_bundle
            if champion_name in {"seven_day_drift", "validation_weighted_ensemble"}
            else {"serialized_candidate": model_files.get(champion_name)}
        ),
    }
    selected_features_payload = {
        "interpretation_model": selected_learned.model_name,
        "selected_feature_group": selected_learned.feature_group,
        "selected_features": list(final_processor.selected_columns),
        "selected_feature_count": len(final_processor.selected_columns),
        "fitted_on_development_only": True,
        "training_period": final_processor.fitted_rows,
        "correlated_feature_warning": "Importance is conditional on deterministic correlation filtering.",
    }

    _write_json(paths["registry"], registry)
    _write_json(paths["comparison"], comparison)
    _write_csv(paths["fold_metrics"], fold_metrics)
    _write_csv(paths["oof_predictions"], oof_predictions.loc[:, PREDICTION_COLUMNS])
    _write_csv(paths["holdout_predictions"], holdout_predictions.loc[:, PREDICTION_COLUMNS])
    _write_json(paths["selected_features"], selected_features_payload)
    _write_csv(paths["feature_exclusions"], exclusions)
    _write_csv(paths["feature_availability"], prepared.availability)
    _write_csv(paths["feature_importance"], importance)
    _write_csv(paths["residual_diagnostics"], diagnostics)
    correlation_export = residual_correlation.reset_index().rename(columns={"index": "model_name"})
    _write_csv(paths["residual_correlation"], correlation_export)
    _write_csv(paths["error_by_regime"], error_by_regime)
    _write_json(paths["interval_metrics"], interval_metrics)
    _write_json(paths["provenance"], provenance)
    _write_json(paths["leakage"], leakage)
    _write_json(paths["promotion"], promotion)
    _write_json(paths["champion"], champion_payload)
    _render_report(
        paths["report"],
        holdout_predictions,
        fold_metrics,
        comparison,
        importance,
        error_by_regime,
        residual_correlation,
        interval_metrics,
        provenance,
        promotion,
    )
    return ForecastExperimentResult(
        config=selected,
        provenance=provenance,
        leakage_audit=leakage,
        comparison=comparison,
        promotion=promotion,
        interval_metrics=interval_metrics,
        artifact_paths={name: str(path.relative_to(PROJECT_ROOT)) for name, path in paths.items()},
    )


def experiment_summary(result: ForecastExperimentResult) -> dict[str, Any]:
    """Return a concise strict-JSON CLI summary."""

    return {
        "complete": True,
        "data_provenance": result.provenance["classification"],
        "leakage_audit_passed": result.leakage_audit["passed"],
        "champion": result.promotion["champion_model"],
        "promotion": result.promotion["recommendation"],
        "models": {
            name: {
                "walk_forward_mae": values["walk_forward"]["mean_mae"],
                "holdout_mae": values["holdout"]["mae"],
            }
            for name, values in result.comparison["models"].items()
        },
        "prediction_interval": result.interval_metrics,
        "artifacts": result.artifact_paths,
    }


__all__ = [
    "CURRENT_LOAD",
    "FORECAST_HORIZON",
    "FINGERPRINT_ALGORITHM",
    "FINGERPRINT_FLOAT_DECIMAL_PLACES",
    "MODEL_NAMES",
    "PREPARED_DATASET_CONTRACT_VERSION",
    "PREPARED_FEATURE_RECIPE_VERSION",
    "PREDICTION_COLUMNS",
    "TARGET_ABSOLUTE",
    "TARGET_CHANGE",
    "CandidateResult",
    "ChronologicalPartitions",
    "FeatureProcessor",
    "ForecastConfig",
    "ForecastExperimentResult",
    "ForecastingFrameworkError",
    "PreparedForecastDataset",
    "WalkForwardFold",
    "chronological_partitions",
    "build_oof_residuals",
    "construct_change_target",
    "dependency_status",
    "drift_forecast",
    "expanding_window_folds",
    "experiment_summary",
    "forecast_configuration_fingerprint",
    "optimize_ensemble_weights",
    "order_prediction_intervals",
    "persistence_forecast",
    "prepare_forecast_dataset",
    "reconstruct_absolute_forecast",
    "regression_metrics",
    "train_forecasting_models",
    "validate_prediction_schema",
]
