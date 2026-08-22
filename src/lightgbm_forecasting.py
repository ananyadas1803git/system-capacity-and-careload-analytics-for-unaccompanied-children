"""Leakage-safe LightGBM forecasting for seven-day HHS UAC care load.

The model predicts the change in Total System Load over the next seven days and
reconstructs an absolute forecast from the load known at the forecast origin.
Model selection uses expanding-window cross-validation with a seven-day gap;
the final chronological holdout remains untouched until selection is complete.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import lightgbm
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from app_utils import DATE_COLUMN, TOTAL_LOAD_COLUMN
from backend.utils import dataframe_fingerprint, json_safe, utc_now_iso
from src.logger import PerformanceTimer, get_logger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_PATH = PROJECT_ROOT / "data" / "processed" / "uac_capacity_ml_features.parquet"
DEFAULT_PROVENANCE_PATH = PROJECT_ROOT / "data" / "processed" / "preprocessing_report.json"
DEFAULT_RIDGE_PREDICTIONS_PATH = PROJECT_ROOT / "output" / "exports" / "model_test_predictions.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "output" / "models" / "capacity_lightgbm_baseline.txt"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "output" / "models" / "lightgbm_model_metadata.json"
DEFAULT_EVALUATION_PATH = PROJECT_ROOT / "output" / "models" / "lightgbm_evaluation_metrics.json"
DEFAULT_PREDICTIONS_PATH = PROJECT_ROOT / "output" / "exports" / "lightgbm_test_predictions.csv"
DEFAULT_IMPORTANCE_PATH = PROJECT_ROOT / "output" / "exports" / "lightgbm_feature_importance.csv"

ABSOLUTE_TARGET_COLUMN = "target_total_load_t_plus_7d"
CHANGE_TARGET_COLUMN = "target_change_7d"
CURRENT_LOAD_FEATURE = "current_total_system_load"
DRIFT_LAG_FEATURE = "lag_7d_total_system_load"
FORECAST_HORIZON_DAYS = 7

RIDGE_DATE_COLUMN = "Date"
RIDGE_ACTUAL_COLUMN = "Actual Total System Load T+7"
RIDGE_PREDICTION_COLUMN = "Ridge Prediction"

MODEL_PREDICTION_COLUMN = "lightgbm_forecast"
PERSISTENCE_PREDICTION_COLUMN = "persistence_forecast"
DRIFT_PREDICTION_COLUMN = "seven_day_drift_forecast"
RIDGE_FORECAST_COLUMN = "ridge_forecast"
LOWER_PREDICTION_COLUMN = "quantile_p10_forecast"
MEDIAN_PREDICTION_COLUMN = "quantile_p50_forecast"
UPPER_PREDICTION_COLUMN = "quantile_p90_forecast"

SAFE_FEATURE_PREFIXES = ("calendar_", "lag_", "rolling_", "ema_")
SAFE_EXACT_FEATURES = ("quality_is_imputed_date",)
TARGET_OR_FUTURE_TOKENS = ("target_", "future", "lead_", "t_plus", "next_")

REQUIRED_PARAMETER_KEYS = frozenset(
    {
        "learning_rate",
        "n_estimators",
        "num_leaves",
        "max_depth",
        "min_child_samples",
        "colsample_bytree",
        "subsample",
        "reg_alpha",
        "reg_lambda",
    }
)

PREDICTION_ARTIFACT_COLUMNS = (
    DATE_COLUMN,
    "actual_total_system_load_t_plus_7d",
    CURRENT_LOAD_FEATURE,
    CHANGE_TARGET_COLUMN,
    "lightgbm_predicted_change_7d",
    MODEL_PREDICTION_COLUMN,
    PERSISTENCE_PREDICTION_COLUMN,
    DRIFT_PREDICTION_COLUMN,
    RIDGE_FORECAST_COLUMN,
    LOWER_PREDICTION_COLUMN,
    MEDIAN_PREDICTION_COLUMN,
    UPPER_PREDICTION_COLUMN,
    "prediction_interval_covered",
    "lightgbm_residual",
)

IMPORTANCE_ARTIFACT_COLUMNS = (
    "feature",
    "gain_importance",
    "split_importance",
    "gain_importance_percent",
    "rank",
)

logger = get_logger("lightgbm_forecasting")


class ForecastingError(RuntimeError):
    """Raised when forecasting inputs, training, or artifacts are invalid."""


def default_candidate_parameters() -> tuple[dict[str, float | int], ...]:
    """Return a small, regularized, deterministic tuning search space."""

    return (
        {
            "learning_rate": 0.03,
            "n_estimators": 300,
            "num_leaves": 7,
            "max_depth": 3,
            "min_child_samples": 30,
            "colsample_bytree": 0.80,
            "subsample": 0.80,
            "reg_alpha": 1.0,
            "reg_lambda": 5.0,
        },
        {
            "learning_rate": 0.03,
            "n_estimators": 500,
            "num_leaves": 15,
            "max_depth": 4,
            "min_child_samples": 40,
            "colsample_bytree": 0.80,
            "subsample": 0.80,
            "reg_alpha": 2.0,
            "reg_lambda": 10.0,
        },
        {
            "learning_rate": 0.05,
            "n_estimators": 250,
            "num_leaves": 7,
            "max_depth": 3,
            "min_child_samples": 40,
            "colsample_bytree": 0.90,
            "subsample": 0.90,
            "reg_alpha": 2.0,
            "reg_lambda": 10.0,
        },
        {
            "learning_rate": 0.05,
            "n_estimators": 400,
            "num_leaves": 15,
            "max_depth": 4,
            "min_child_samples": 50,
            "colsample_bytree": 0.75,
            "subsample": 0.80,
            "reg_alpha": 5.0,
            "reg_lambda": 15.0,
        },
        {
            "learning_rate": 0.02,
            "n_estimators": 600,
            "num_leaves": 15,
            "max_depth": 5,
            "min_child_samples": 50,
            "colsample_bytree": 0.80,
            "subsample": 0.80,
            "reg_alpha": 2.0,
            "reg_lambda": 10.0,
        },
    )


@dataclass(frozen=True)
class LightGBMForecastConfig:
    """Configuration for leakage-safe training, evaluation, and persistence."""

    feature_path: Path = DEFAULT_FEATURE_PATH
    provenance_path: Path | None = DEFAULT_PROVENANCE_PATH
    ridge_predictions_path: Path | None = DEFAULT_RIDGE_PREDICTIONS_PATH
    model_path: Path = DEFAULT_MODEL_PATH
    metadata_path: Path = DEFAULT_METADATA_PATH
    evaluation_path: Path = DEFAULT_EVALUATION_PATH
    predictions_path: Path = DEFAULT_PREDICTIONS_PATH
    feature_importance_path: Path = DEFAULT_IMPORTANCE_PATH
    absolute_target_column: str = ABSOLUTE_TARGET_COLUMN
    current_load_column: str = TOTAL_LOAD_COLUMN
    forecast_horizon_days: int = FORECAST_HORIZON_DAYS
    holdout_fraction: float = 0.20
    gap_days: int = FORECAST_HORIZON_DAYS
    cv_splits: int = 4
    cv_validation_rows: int = 90
    minimum_training_rows: int = 180
    random_seed: int = 42
    source_label: str = "HHS_Unaccompanied_Alien_Children_Program.csv"
    synthetic_data: bool = False
    overwrite: bool = False
    require_ridge_baseline: bool = True
    candidate_parameters: tuple[Mapping[str, float | int], ...] = field(
        default_factory=default_candidate_parameters
    )

    def __post_init__(self) -> None:
        if self.forecast_horizon_days < 1:
            raise ValueError("forecast_horizon_days must be positive.")
        if not 0 < self.holdout_fraction < 1:
            raise ValueError("holdout_fraction must be between zero and one.")
        if self.gap_days < self.forecast_horizon_days:
            raise ValueError("gap_days must be at least the forecast horizon to prevent leakage.")
        if self.cv_splits < 2:
            raise ValueError("cv_splits must be at least two.")
        if self.cv_validation_rows < 1:
            raise ValueError("cv_validation_rows must be positive.")
        if self.minimum_training_rows < 20:
            raise ValueError("minimum_training_rows must be at least 20.")
        if not self.source_label.strip():
            raise ValueError("source_label must not be empty.")
        if not self.candidate_parameters:
            raise ValueError("candidate_parameters must not be empty.")
        for index, parameters in enumerate(self.candidate_parameters, start=1):
            missing = REQUIRED_PARAMETER_KEYS.difference(parameters)
            if missing:
                raise ValueError(
                    f"Candidate {index} is missing parameter(s): " + ", ".join(sorted(missing))
                )

    @property
    def artifact_paths(self) -> tuple[Path, ...]:
        """Return every output created by a training run."""

        return (
            self.model_path,
            self.metadata_path,
            self.evaluation_path,
            self.predictions_path,
            self.feature_importance_path,
        )


@dataclass(frozen=True)
class RegressionMetrics:
    """Standard regression metrics for one holdout forecast."""

    mae: float
    rmse: float
    mape_percent: float | None
    r2: float | None

    def to_dict(self, persistence_mae: float) -> dict[str, float | None]:
        """Return metrics plus MAE improvement relative to persistence."""

        improvement = (
            (persistence_mae - self.mae) / persistence_mae * 100.0 if persistence_mae > 0 else None
        )
        return {
            "mae": self.mae,
            "rmse": self.rmse,
            "mape_percent": self.mape_percent,
            "r2": self.r2,
            "mae_improvement_vs_persistence_percent": improvement,
        }


@dataclass(frozen=True)
class WalkForwardFold:
    """Integer positions for one expanding-window validation fold."""

    fold: int
    train_indices: np.ndarray
    validation_indices: np.ndarray

    @property
    def observed_gap_rows(self) -> int:
        """Return observations excluded between train and validation."""

        return int(self.validation_indices[0] - self.train_indices[-1] - 1)


@dataclass
class PreparedForecastData:
    """Validated model frame with an explicit leakage-safe feature manifest."""

    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    excluded_columns: dict[str, str]
    data_fingerprint: str
    schema_fingerprint: str
    missing_feature_values: int


@dataclass
class ChronologicalSplit:
    """Training, embargo, and untouched final holdout partitions."""

    train: pd.DataFrame
    gap: pd.DataFrame
    holdout: pd.DataFrame


@dataclass
class ForecastTrainingResult:
    """Complete in-memory and persisted output of a training run."""

    config: LightGBMForecastConfig
    model: LGBMRegressor
    quantile_models: dict[str, LGBMRegressor]
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    metadata: dict[str, Any]
    evaluation: dict[str, Any]


def construct_change_target(
    future_total_load: pd.Series,
    current_total_load: pd.Series,
) -> pd.Series:
    """Return the future-minus-current seven-day load change target."""

    if not isinstance(future_total_load, pd.Series) or not isinstance(
        current_total_load, pd.Series
    ):
        raise TypeError("future_total_load and current_total_load must be Series.")
    if not future_total_load.index.equals(current_total_load.index):
        raise ForecastingError("Future and current load indexes must match exactly.")
    future = pd.to_numeric(future_total_load, errors="coerce")
    current = pd.to_numeric(current_total_load, errors="coerce")
    change = future - current
    change.name = CHANGE_TARGET_COLUMN
    return change


def reconstruct_absolute_forecast(
    current_total_load: Sequence[float] | np.ndarray,
    predicted_change: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Reconstruct absolute load from the forecast-origin load and change."""

    current = np.asarray(current_total_load, dtype=float)
    change = np.asarray(predicted_change, dtype=float)
    if current.shape != change.shape:
        raise ForecastingError("Current loads and predicted changes must align.")
    if not np.isfinite(current).all() or not np.isfinite(change).all():
        raise ForecastingError("Forecast reconstruction received non-finite values.")
    return current + change


def order_prediction_intervals(
    lower: Sequence[float] | np.ndarray,
    median: Sequence[float] | np.ndarray,
    upper: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Order independently fitted quantiles row-wise to prevent crossing."""

    arrays = [np.asarray(values, dtype=float) for values in (lower, median, upper)]
    if arrays[0].shape != arrays[1].shape or arrays[1].shape != arrays[2].shape:
        raise ForecastingError("Quantile prediction arrays must have equal shapes.")
    stacked = np.column_stack(arrays)
    if not np.isfinite(stacked).all():
        raise ForecastingError("Quantile predictions contain non-finite values.")
    ordered = np.sort(stacked, axis=1)
    return ordered[:, 0], ordered[:, 1], ordered[:, 2]


def find_leakage_features(feature_columns: Sequence[str]) -> list[str]:
    """Return model feature names containing explicit future-target markers."""

    return sorted(
        {
            str(column)
            for column in feature_columns
            if any(token in str(column).casefold() for token in TARGET_OR_FUTURE_TOKENS)
        }
    )


def select_leakage_safe_features(
    frame: pd.DataFrame,
) -> tuple[tuple[str, ...], dict[str, str]]:
    """Select forecast-origin features and document every excluded column.

    Same-day operational ratios and flow variables are excluded. Operational
    information enters only through lagged or shifted historical aggregates.
    Current total load is added separately after selection because it is both a
    known forecast-origin value and the reconstruction anchor.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    selected: list[str] = []
    excluded: dict[str, str] = {}
    for column in frame.columns:
        name = str(column)
        normalized = name.casefold()
        if any(token in normalized for token in TARGET_OR_FUTURE_TOKENS):
            excluded[name] = "future target or future-derived field"
            continue
        safe_prefix = name.startswith(SAFE_FEATURE_PREFIXES)
        safe_load_momentum = name.startswith("momentum_total_load_")
        safe_exact = name in SAFE_EXACT_FEATURES
        if safe_prefix or safe_load_momentum or safe_exact:
            if not pd.api.types.is_numeric_dtype(frame[column]):
                excluded[name] = "non-numeric candidate feature"
            else:
                selected.append(name)
        elif name.startswith("operational_") or name.startswith("momentum_net_intake_"):
            excluded[name] = "same-day operational information; lagged form required"
        elif name.startswith("quality_"):
            excluded[name] = "same-day operational quality flag"
        else:
            excluded[name] = "source, derived display field, or unsupported feature"

    if not selected:
        raise ForecastingError("No leakage-safe numeric feature columns were found.")
    leakage = find_leakage_features(selected)
    if leakage:
        raise ForecastingError(
            "Feature selection contains future information: " + ", ".join(leakage)
        )
    return tuple(selected), excluded


def _schema_fingerprint(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    """Hash the ordered model schema separately from the full data fingerprint."""

    payload = [{"name": str(column), "dtype": str(frame[column].dtype)} for column in columns]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    """Return a bounded SHA-256 digest for an artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    """Return a repository-relative path when an artifact is inside the project."""

    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def load_feature_data(path: str | Path) -> pd.DataFrame:
    """Load and validate a chronological Parquet feature artifact."""

    feature_path = Path(path).expanduser().resolve()
    if not feature_path.is_file():
        raise ForecastingError(f"Feature artifact not found: {feature_path}")
    if feature_path.suffix.casefold() not in {".parquet", ".pq"}:
        raise ForecastingError("Feature artifact must be a Parquet file.")
    try:
        frame = pd.read_parquet(feature_path, engine="pyarrow")
    except (ImportError, OSError, ValueError) as exc:
        raise ForecastingError(f"Unable to read feature artifact: {exc}") from exc
    if frame.empty:
        raise ForecastingError("Feature artifact is empty.")
    return frame


def _normalize_date_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted, unique, daily DatetimeIndex frame."""

    normalized = frame.copy()
    if DATE_COLUMN in normalized.columns:
        dates = pd.to_datetime(normalized.pop(DATE_COLUMN), errors="coerce")
        if dates.isna().any():
            raise ForecastingError("Feature data contains invalid Date values.")
        normalized.index = pd.DatetimeIndex(dates, name=DATE_COLUMN)
    elif isinstance(normalized.index, pd.DatetimeIndex):
        normalized.index = pd.DatetimeIndex(normalized.index, name=DATE_COLUMN)
    else:
        raise ForecastingError("Feature data requires Date or a DatetimeIndex.")
    if normalized.index.has_duplicates:
        raise ForecastingError("Feature data contains duplicate dates.")
    normalized = normalized.sort_index()
    return normalized


def prepare_forecasting_data(
    frame: pd.DataFrame,
    config: LightGBMForecastConfig,
) -> PreparedForecastData:
    """Validate features, construct the change target, and enforce leakage rules."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    if frame.empty:
        raise ForecastingError("Forecasting input is empty.")
    normalized = _normalize_date_index(frame)
    required = {config.absolute_target_column, config.current_load_column}
    missing = sorted(required.difference(normalized.columns))
    if missing:
        raise ForecastingError("Missing required column(s): " + ", ".join(missing))

    selected, excluded = select_leakage_safe_features(normalized)
    numeric_columns = list(selected) + [
        config.current_load_column,
        config.absolute_target_column,
    ]
    numeric = normalized[numeric_columns].apply(pd.to_numeric, errors="coerce")
    raw_values = numeric.to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(raw_values).any():
        raise ForecastingError("Forecasting data contains infinite numeric values.")

    working = numeric.copy()
    working[CURRENT_LOAD_FEATURE] = working[config.current_load_column].astype(float)
    working[CHANGE_TARGET_COLUMN] = construct_change_target(
        working[config.absolute_target_column],
        working[config.current_load_column],
    )
    complete_target = (
        working[[config.absolute_target_column, CURRENT_LOAD_FEATURE, CHANGE_TARGET_COLUMN]]
        .notna()
        .all(axis=1)
    )
    working = working.loc[complete_target].copy()
    if working.empty:
        raise ForecastingError("No rows contain a complete current and future load.")

    model_features = (*selected, CURRENT_LOAD_FEATURE)
    all_missing = [column for column in model_features if working[column].isna().all()]
    if all_missing:
        raise ForecastingError(
            "Model feature(s) contain only missing values: " + ", ".join(all_missing)
        )
    if len(working) < config.minimum_training_rows:
        raise ForecastingError(
            f"Only {len(working):,} complete observations are available; "
            f"at least {config.minimum_training_rows:,} are required."
        )
    date_steps = working.index.to_series().diff().dropna()
    if not date_steps.eq(pd.Timedelta(days=1)).all():
        raise ForecastingError("Forecasting observations must form a complete daily time series.")
    leakage = find_leakage_features(model_features)
    if leakage:
        raise ForecastingError(
            "Future-derived model features are prohibited: " + ", ".join(leakage)
        )

    final_columns = list(model_features) + [
        config.absolute_target_column,
        CHANGE_TARGET_COLUMN,
    ]
    final_frame = working[final_columns].copy()
    final_frame.attrs.clear()
    return PreparedForecastData(
        frame=final_frame,
        feature_columns=tuple(model_features),
        excluded_columns=excluded,
        data_fingerprint=dataframe_fingerprint(final_frame),
        schema_fingerprint=_schema_fingerprint(final_frame, model_features),
        missing_feature_values=int(final_frame[list(model_features)].isna().sum().sum()),
    )


def chronological_holdout_split(
    prepared: PreparedForecastData,
    config: LightGBMForecastConfig,
) -> ChronologicalSplit:
    """Reserve the final 20% and a seven-day pre-holdout embargo."""

    frame = prepared.frame
    holdout_rows = max(1, int(np.ceil(len(frame) * config.holdout_fraction)))
    holdout_start = len(frame) - holdout_rows
    train_end = holdout_start - config.gap_days
    if train_end < config.minimum_training_rows:
        raise ForecastingError("Holdout and gap leave too few observations for model training.")
    train = frame.iloc[:train_end].copy()
    gap = frame.iloc[train_end:holdout_start].copy()
    holdout = frame.iloc[holdout_start:].copy()
    if len(gap) != config.gap_days:
        raise ForecastingError("Chronological holdout gap was constructed incorrectly.")
    return ChronologicalSplit(train=train, gap=gap, holdout=holdout)


def expanding_window_folds(
    training_rows: int,
    config: LightGBMForecastConfig,
) -> tuple[WalkForwardFold, ...]:
    """Return expanding-window folds with an explicit seven-day gap."""

    if training_rows < config.minimum_training_rows:
        raise ForecastingError("Insufficient training rows for walk-forward validation.")
    required = config.cv_splits * config.cv_validation_rows + config.gap_days + 1
    if training_rows < required:
        raise ForecastingError(
            "Training data is too short for the requested walk-forward configuration."
        )
    splitter = TimeSeriesSplit(
        n_splits=config.cv_splits,
        test_size=config.cv_validation_rows,
        gap=config.gap_days,
    )
    folds = tuple(
        WalkForwardFold(
            fold=index,
            train_indices=train_indices,
            validation_indices=validation_indices,
        )
        for index, (train_indices, validation_indices) in enumerate(
            splitter.split(np.arange(training_rows)), start=1
        )
    )
    if any(fold.observed_gap_rows != config.gap_days for fold in folds):
        raise ForecastingError("A walk-forward fold does not preserve the required gap.")
    return folds


def _lightgbm_parameters(
    candidate: Mapping[str, float | int],
    config: LightGBMForecastConfig,
    *,
    objective: str,
    alpha: float | None = None,
) -> dict[str, Any]:
    """Combine tuned values with deterministic LightGBM controls."""

    parameters: dict[str, Any] = {
        **dict(candidate),
        "objective": objective,
        "random_state": config.random_seed,
        "n_jobs": 1,
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
        "subsample_freq": 1,
        "bagging_seed": config.random_seed,
        "feature_fraction_seed": config.random_seed,
        "data_random_seed": config.random_seed,
    }
    if alpha is not None:
        parameters["alpha"] = alpha
    return parameters


def fit_deterministic_lightgbm(
    features: pd.DataFrame,
    target_change: pd.Series,
    candidate: Mapping[str, float | int],
    config: LightGBMForecastConfig,
    *,
    objective: str = "regression_l1",
    alpha: float | None = None,
) -> LGBMRegressor:
    """Fit one deterministic LightGBM model without scaling its features."""

    if features.empty or target_change.empty:
        raise ForecastingError("Model features and target must not be empty.")
    if not features.index.equals(target_change.index):
        raise ForecastingError("Model features and target indexes do not align.")
    target_values = pd.to_numeric(target_change, errors="coerce").to_numpy(float)
    if not np.isfinite(target_values).all():
        raise ForecastingError("Training target contains missing or infinite values.")
    try:
        model = LGBMRegressor(
            **_lightgbm_parameters(
                candidate,
                config,
                objective=objective,
                alpha=alpha,
            )
        )
        model.fit(features, target_values)
    except (TypeError, ValueError, lightgbm.basic.LightGBMError) as exc:
        raise ForecastingError(f"LightGBM training failed: {exc}") from exc
    return model


def calculate_regression_metrics(
    actual: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
) -> RegressionMetrics:
    """Calculate strict MAE, RMSE, MAPE, and R-squared metrics."""

    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    if actual_values.shape != predicted_values.shape or actual_values.size == 0:
        raise ForecastingError("Actual and predicted values must be non-empty and align.")
    if not np.isfinite(actual_values).all() or not np.isfinite(predicted_values).all():
        raise ForecastingError("Metric inputs contain missing or infinite values.")
    nonzero = actual_values != 0
    mape = (
        float(
            np.mean(
                np.abs(
                    (actual_values[nonzero] - predicted_values[nonzero]) / actual_values[nonzero]
                )
            )
            * 100.0
        )
        if nonzero.any()
        else None
    )
    r2 = (
        float(r2_score(actual_values, predicted_values))
        if len(actual_values) > 1 and not np.all(actual_values == actual_values[0])
        else None
    )
    return RegressionMetrics(
        mae=float(mean_absolute_error(actual_values, predicted_values)),
        rmse=float(np.sqrt(mean_squared_error(actual_values, predicted_values))),
        mape_percent=mape,
        r2=r2,
    )


def _evaluate_candidates(
    split: ChronologicalSplit,
    prepared: PreparedForecastData,
    folds: tuple[WalkForwardFold, ...],
    config: LightGBMForecastConfig,
) -> tuple[dict[str, float | int], list[dict[str, Any]], list[dict[str, Any]]]:
    """Tune candidates using only expanding-window training-period folds."""

    train = split.train
    feature_columns = list(prepared.feature_columns)
    candidate_results: list[dict[str, Any]] = []
    fold_manifest: list[dict[str, Any]] = []
    persistence_fold_mae: list[float] = []

    for fold in folds:
        validation = train.iloc[fold.validation_indices]
        actual = validation[config.absolute_target_column].to_numpy(float)
        persistence = validation[CURRENT_LOAD_FEATURE].to_numpy(float)
        persistence_fold_mae.append(float(mean_absolute_error(actual, persistence)))
        fold_manifest.append(
            {
                "fold": fold.fold,
                "training_rows": len(fold.train_indices),
                "validation_rows": len(fold.validation_indices),
                "gap_rows": fold.observed_gap_rows,
                "training_start": train.index[fold.train_indices[0]].date().isoformat(),
                "training_end": train.index[fold.train_indices[-1]].date().isoformat(),
                "validation_start": train.index[fold.validation_indices[0]].date().isoformat(),
                "validation_end": train.index[fold.validation_indices[-1]].date().isoformat(),
                "persistence_mae": persistence_fold_mae[-1],
            }
        )

    for candidate_index, candidate in enumerate(config.candidate_parameters, start=1):
        fold_mae: list[float] = []
        for fold in folds:
            fold_train = train.iloc[fold.train_indices]
            validation = train.iloc[fold.validation_indices]
            model = fit_deterministic_lightgbm(
                fold_train[feature_columns],
                fold_train[CHANGE_TARGET_COLUMN],
                candidate,
                config,
            )
            predicted_change = model.predict(validation[feature_columns])
            predicted_absolute = reconstruct_absolute_forecast(
                validation[CURRENT_LOAD_FEATURE].to_numpy(float),
                predicted_change,
            )
            fold_mae.append(
                float(
                    mean_absolute_error(
                        validation[config.absolute_target_column].to_numpy(float),
                        predicted_absolute,
                    )
                )
            )
        candidate_results.append(
            {
                "candidate": candidate_index,
                "parameters": dict(candidate),
                "fold_mae": fold_mae,
                "mean_validation_mae": float(np.mean(fold_mae)),
                "std_validation_mae": float(np.std(fold_mae, ddof=0)),
            }
        )

    best = min(
        candidate_results,
        key=lambda result: (result["mean_validation_mae"], result["candidate"]),
    )
    for manifest, persistence_mae in zip(fold_manifest, persistence_fold_mae, strict=True):
        manifest["best_lightgbm_mae"] = best["fold_mae"][manifest["fold"] - 1]
    return dict(best["parameters"]), candidate_results, fold_manifest


def _load_ridge_forecast(
    holdout: pd.DataFrame,
    config: LightGBMForecastConfig,
) -> np.ndarray | None:
    """Align the preserved ridge holdout predictions with the new holdout."""

    path = config.ridge_predictions_path
    if path is None:
        if config.require_ridge_baseline:
            raise ForecastingError("A ridge predictions path is required.")
        return None
    ridge_path = Path(path).expanduser().resolve()
    if not ridge_path.is_file():
        if config.require_ridge_baseline:
            raise ForecastingError(f"Ridge prediction artifact not found: {ridge_path}")
        return None
    try:
        ridge = pd.read_csv(ridge_path, encoding="utf-8")
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ForecastingError(f"Unable to read ridge predictions: {exc}") from exc
    required = {RIDGE_DATE_COLUMN, RIDGE_ACTUAL_COLUMN, RIDGE_PREDICTION_COLUMN}
    missing = sorted(required.difference(ridge.columns))
    if missing:
        raise ForecastingError(
            "Ridge prediction artifact is missing column(s): " + ", ".join(missing)
        )
    ridge[RIDGE_DATE_COLUMN] = pd.to_datetime(ridge[RIDGE_DATE_COLUMN], errors="coerce")
    if ridge[RIDGE_DATE_COLUMN].isna().any() or ridge[RIDGE_DATE_COLUMN].duplicated().any():
        raise ForecastingError("Ridge prediction dates are invalid or duplicated.")
    ridge = ridge.set_index(RIDGE_DATE_COLUMN).reindex(holdout.index)
    if ridge[[RIDGE_ACTUAL_COLUMN, RIDGE_PREDICTION_COLUMN]].isna().any().any():
        raise ForecastingError("Ridge predictions do not cover the final holdout dates.")
    ridge_actual = pd.to_numeric(ridge[RIDGE_ACTUAL_COLUMN], errors="coerce").to_numpy(float)
    holdout_actual = holdout[config.absolute_target_column].to_numpy(float)
    if not np.allclose(ridge_actual, holdout_actual, rtol=0, atol=1e-9):
        raise ForecastingError("Ridge artifact uses a different holdout or target.")
    predictions = pd.to_numeric(ridge[RIDGE_PREDICTION_COLUMN], errors="coerce").to_numpy(float)
    if not np.isfinite(predictions).all():
        raise ForecastingError("Ridge predictions contain non-finite values.")
    return predictions


def _feature_importance(
    model: LGBMRegressor,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """Return deterministic gain and split feature importance rankings."""

    if model.booster_ is None:
        raise ForecastingError("The fitted model does not expose a booster.")
    gain = model.booster_.feature_importance(importance_type="gain").astype(float)
    split = model.booster_.feature_importance(importance_type="split").astype(float)
    total_gain = float(gain.sum())
    importance = pd.DataFrame(
        {
            "feature": list(feature_columns),
            "gain_importance": gain,
            "split_importance": split,
            "gain_importance_percent": (
                gain / total_gain * 100.0 if total_gain > 0 else np.zeros_like(gain)
            ),
        }
    ).sort_values(
        ["gain_importance", "split_importance", "feature"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    importance["rank"] = np.arange(1, len(importance) + 1)
    return importance[list(IMPORTANCE_ARTIFACT_COLUMNS)].reset_index(drop=True)


def _infer_provenance(
    config: LightGBMForecastConfig,
    feature_frame: pd.DataFrame,
) -> dict[str, Any]:
    """Build explicit source and feature-artifact provenance metadata."""

    feature_path = Path(config.feature_path).expanduser().resolve()
    provenance: dict[str, Any] = {
        "source_label": config.source_label,
        "synthetic_data": config.synthetic_data,
        "source_type": "synthetic" if config.synthetic_data else "real_aggregate",
        "feature_artifact": _portable_path(feature_path),
        "feature_artifact_sha256": (_file_sha256(feature_path) if feature_path.is_file() else None),
        "feature_rows": len(feature_frame),
        "feature_columns": len(feature_frame.columns),
    }
    if config.provenance_path is not None:
        path = Path(config.provenance_path).expanduser().resolve()
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise ForecastingError(f"Unable to read provenance metadata: {exc}") from exc
            provenance["preprocessing_report"] = _portable_path(path)
            provenance["source_sha256"] = payload.get("source_sha256")
            provenance["preprocessing_source"] = payload.get("source")
    return provenance


def _promotion_decision(
    lightgbm_cv_mae: float,
    persistence_cv_mae: float,
    lightgbm_holdout_mae: float,
    persistence_holdout_mae: float,
) -> dict[str, Any]:
    """Apply the immutable two-part promotion rule."""

    cv_pass = lightgbm_cv_mae < persistence_cv_mae
    holdout_pass = lightgbm_holdout_mae < persistence_holdout_mae
    passed = cv_pass and holdout_pass
    reasons: list[str] = []
    if not cv_pass:
        reasons.append(
            "LightGBM did not beat persistence on mean walk-forward MAE; "
            "regime changes and limited historical samples reduce generalization."
        )
    if not holdout_pass:
        reasons.append(
            "LightGBM did not beat persistence on the untouched holdout; "
            "seven-day care load was highly persistent in this period."
        )
    return {
        "passed": passed,
        "cross_validation_passed": cv_pass,
        "holdout_passed": holdout_pass,
        "champion_model": "lightgbm_change_forecast" if passed else "persistence",
        "recommendation": "promote" if passed else "continue_research",
        "rationale": reasons or ["LightGBM beat persistence in both required evaluation stages."],
    }


def _prediction_frame(
    split: ChronologicalSplit,
    config: LightGBMForecastConfig,
    model_change: np.ndarray,
    model_absolute: np.ndarray,
    drift: np.ndarray,
    ridge: np.ndarray | None,
    lower: np.ndarray,
    median: np.ndarray,
    upper: np.ndarray,
) -> pd.DataFrame:
    """Build the documented holdout prediction artifact."""

    holdout = split.holdout
    actual = holdout[config.absolute_target_column].to_numpy(float)
    current = holdout[CURRENT_LOAD_FEATURE].to_numpy(float)
    ridge_values = ridge if ridge is not None else np.full(len(holdout), np.nan)
    predictions = pd.DataFrame(
        {
            DATE_COLUMN: holdout.index,
            "actual_total_system_load_t_plus_7d": actual,
            CURRENT_LOAD_FEATURE: current,
            CHANGE_TARGET_COLUMN: holdout[CHANGE_TARGET_COLUMN].to_numpy(float),
            "lightgbm_predicted_change_7d": model_change,
            MODEL_PREDICTION_COLUMN: model_absolute,
            PERSISTENCE_PREDICTION_COLUMN: current,
            DRIFT_PREDICTION_COLUMN: drift,
            RIDGE_FORECAST_COLUMN: ridge_values,
            LOWER_PREDICTION_COLUMN: lower,
            MEDIAN_PREDICTION_COLUMN: median,
            UPPER_PREDICTION_COLUMN: upper,
            "prediction_interval_covered": (actual >= lower) & (actual <= upper),
            "lightgbm_residual": actual - model_absolute,
        }
    )
    validate_artifact_schemas(predictions, None)
    return predictions


def validate_artifact_schemas(
    predictions: pd.DataFrame,
    feature_importance: pd.DataFrame | None,
) -> None:
    """Validate stable CSV schemas before artifacts are persisted."""

    if tuple(predictions.columns) != PREDICTION_ARTIFACT_COLUMNS:
        raise ForecastingError("Prediction artifact schema is invalid.")
    if predictions.empty:
        raise ForecastingError("Prediction artifact must not be empty.")
    if (
        not (predictions[LOWER_PREDICTION_COLUMN] <= predictions[MEDIAN_PREDICTION_COLUMN]).all()
        or not (predictions[MEDIAN_PREDICTION_COLUMN] <= predictions[UPPER_PREDICTION_COLUMN]).all()
    ):
        raise ForecastingError("Prediction interval columns are not ordered.")
    if feature_importance is not None:
        if tuple(feature_importance.columns) != IMPORTANCE_ARTIFACT_COLUMNS:
            raise ForecastingError("Feature-importance artifact schema is invalid.")
        if feature_importance.empty:
            raise ForecastingError("Feature-importance artifact must not be empty.")


@contextmanager
def _atomic_target(path: Path) -> Iterator[Path]:
    """Yield a sibling temporary path and replace the target after success."""

    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        temporary.unlink(missing_ok=True)
        yield temporary
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _preflight_artifacts(config: LightGBMForecastConfig) -> None:
    """Protect ridge outputs and existing LightGBM artifacts by default."""

    resolved = [path.expanduser().resolve() for path in config.artifact_paths]
    if len(resolved) != len(set(resolved)):
        raise ForecastingError("LightGBM artifact paths must be unique.")
    protected_names = {
        "capacity_ridge_baseline.npz",
        "evaluation_metrics.json",
        "model_test_predictions.csv",
    }
    conflicts = [str(path) for path in resolved if path.name in protected_names]
    if conflicts:
        raise ForecastingError(
            "LightGBM outputs cannot overwrite ridge artifacts: " + ", ".join(conflicts)
        )
    existing = [str(path) for path in resolved if path.exists()]
    if existing and not config.overwrite:
        raise ForecastingError(
            "LightGBM artifact(s) already exist; use overwrite/--force to replace: "
            + ", ".join(existing)
        )


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    """Write strict JSON atomically using the shared scientific serializer."""

    with _atomic_target(path) as temporary:
        temporary.write_text(
            json.dumps(
                json_safe(payload),
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a deterministic UTF-8 CSV artifact atomically."""

    export = frame.copy()
    export.attrs.clear()
    with _atomic_target(path) as temporary:
        export.to_csv(
            temporary,
            index=False,
            encoding="utf-8",
            date_format="%Y-%m-%d",
            lineterminator="\n",
        )


def _persist_artifacts(result: ForecastTrainingResult) -> None:
    """Persist the model, metadata, metrics, predictions, and importance."""

    config = result.config
    if result.model.booster_ is None:
        raise ForecastingError("The final LightGBM model has no fitted booster.")
    with _atomic_target(config.model_path) as temporary:
        result.model.booster_.save_model(str(temporary))
    _write_json(result.metadata, config.metadata_path)
    _write_json(result.evaluation, config.evaluation_path)
    _write_csv(result.predictions, config.predictions_path)
    _write_csv(result.feature_importance, config.feature_importance_path)


def train_lightgbm_forecast(
    config: LightGBMForecastConfig | None = None,
) -> ForecastTrainingResult:
    """Train, evaluate, persist, and return the full LightGBM candidate run."""

    selected = config or LightGBMForecastConfig()
    _preflight_artifacts(selected)
    with PerformanceTimer(logger, "train_lightgbm_forecast"):
        feature_frame = load_feature_data(selected.feature_path)
        provenance = _infer_provenance(selected, feature_frame)
        prepared = prepare_forecasting_data(feature_frame, selected)
        split = chronological_holdout_split(prepared, selected)
        folds = expanding_window_folds(len(split.train), selected)
        best_parameters, candidate_results, fold_manifest = _evaluate_candidates(
            split, prepared, folds, selected
        )

        feature_columns = list(prepared.feature_columns)
        final_model = fit_deterministic_lightgbm(
            split.train[feature_columns],
            split.train[CHANGE_TARGET_COLUMN],
            best_parameters,
            selected,
        )
        model_change = final_model.predict(split.holdout[feature_columns])
        current = split.holdout[CURRENT_LOAD_FEATURE].to_numpy(float)
        model_absolute = reconstruct_absolute_forecast(current, model_change)
        persistence = current.copy()
        if DRIFT_LAG_FEATURE not in split.holdout.columns:
            raise ForecastingError(f"Seven-day drift requires feature '{DRIFT_LAG_FEATURE}'.")
        drift = current + (current - split.holdout[DRIFT_LAG_FEATURE].to_numpy(float))
        if not np.isfinite(drift).all():
            raise ForecastingError("Seven-day drift forecast contains missing values.")
        ridge = _load_ridge_forecast(split.holdout, selected)

        quantile_models: dict[str, LGBMRegressor] = {}
        quantile_absolute: list[np.ndarray] = []
        for label, alpha in (("p10", 0.10), ("p50", 0.50), ("p90", 0.90)):
            quantile_model = fit_deterministic_lightgbm(
                split.train[feature_columns],
                split.train[CHANGE_TARGET_COLUMN],
                best_parameters,
                selected,
                objective="quantile",
                alpha=alpha,
            )
            quantile_models[label] = quantile_model
            quantile_absolute.append(
                reconstruct_absolute_forecast(
                    current,
                    quantile_model.predict(split.holdout[feature_columns]),
                )
            )
        lower, median, upper = order_prediction_intervals(*quantile_absolute)

        predictions = _prediction_frame(
            split,
            selected,
            np.asarray(model_change, dtype=float),
            model_absolute,
            drift,
            ridge,
            lower,
            median,
            upper,
        )
        importance = _feature_importance(final_model, feature_columns)
        validate_artifact_schemas(predictions, importance)

        actual = split.holdout[selected.absolute_target_column].to_numpy(float)
        persistence_metrics = calculate_regression_metrics(actual, persistence)
        models: dict[str, RegressionMetrics] = {
            "persistence": persistence_metrics,
            "seven_day_drift": calculate_regression_metrics(actual, drift),
            "lightgbm_change_forecast": calculate_regression_metrics(actual, model_absolute),
            "lightgbm_quantile_median": calculate_regression_metrics(actual, median),
        }
        if ridge is not None:
            models["ridge"] = calculate_regression_metrics(actual, ridge)

        best_cv = min(
            candidate_results,
            key=lambda result: (result["mean_validation_mae"], result["candidate"]),
        )
        persistence_cv_mae = float(np.mean([fold["persistence_mae"] for fold in fold_manifest]))
        lightgbm_cv_mae = float(best_cv["mean_validation_mae"])
        promotion = _promotion_decision(
            lightgbm_cv_mae,
            persistence_cv_mae,
            models["lightgbm_change_forecast"].mae,
            persistence_metrics.mae,
        )

        model_metrics = {
            name: metric.to_dict(persistence_metrics.mae) for name, metric in models.items()
        }
        interval_coverage = float(predictions["prediction_interval_covered"].mean())
        evaluation: dict[str, Any] = {
            "model": "LightGBM seven-day change forecast candidate",
            "target": {
                "absolute": selected.absolute_target_column,
                "modeled": CHANGE_TARGET_COLUMN,
                "definition": (
                    f"{selected.absolute_target_column} - {selected.current_load_column}"
                ),
                "reconstruction": (
                    f"{selected.current_load_column} + predicted {CHANGE_TARGET_COLUMN}"
                ),
            },
            "forecast_horizon_days": selected.forecast_horizon_days,
            "cross_validation": {
                "strategy": "expanding_window",
                "optimization_metric": "mean_validation_mae",
                "fold_count": selected.cv_splits,
                "validation_rows_per_fold": selected.cv_validation_rows,
                "gap_rows": selected.gap_days,
                "best_lightgbm_mean_mae": lightgbm_cv_mae,
                "persistence_mean_mae": persistence_cv_mae,
                "candidate_results": candidate_results,
                "folds": fold_manifest,
            },
            "holdout": {
                "rows": len(split.holdout),
                "start": split.holdout.index.min().date().isoformat(),
                "end": split.holdout.index.max().date().isoformat(),
                "models": model_metrics,
                "prediction_interval": {
                    "nominal_lower_quantile": 0.10,
                    "median_quantile": 0.50,
                    "nominal_upper_quantile": 0.90,
                    "nominal_coverage_percent": 80.0,
                    "empirical_coverage_percent": interval_coverage * 100.0,
                    "mean_width": float(np.mean(upper - lower)),
                    "ordering_enforced": True,
                },
            },
            "promotion": promotion,
        }

        metadata: dict[str, Any] = {
            "model": "LightGBM seven-day change forecast candidate",
            "created_at_utc": utc_now_iso(),
            "training_start": split.train.index.min().date().isoformat(),
            "training_end": split.train.index.max().date().isoformat(),
            "training_rows": len(split.train),
            "pre_holdout_gap_start": split.gap.index.min().date().isoformat(),
            "pre_holdout_gap_end": split.gap.index.max().date().isoformat(),
            "gap_rows": len(split.gap),
            "holdout_start": split.holdout.index.min().date().isoformat(),
            "holdout_end": split.holdout.index.max().date().isoformat(),
            "holdout_rows": len(split.holdout),
            "forecast_horizon_days": selected.forecast_horizon_days,
            "feature_count": len(feature_columns),
            "feature_names": feature_columns,
            "excluded_columns": prepared.excluded_columns,
            "target_definition": evaluation["target"],
            "best_hyperparameters": best_parameters,
            "cross_validation_scores": {
                "lightgbm_fold_mae": best_cv["fold_mae"],
                "lightgbm_mean_mae": lightgbm_cv_mae,
                "persistence_fold_mae": [fold["persistence_mae"] for fold in fold_manifest],
                "persistence_mean_mae": persistence_cv_mae,
            },
            "dataset_provenance": provenance,
            "random_seed": selected.random_seed,
            "library_versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "lightgbm": lightgbm.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "data_fingerprint_sha256": prepared.data_fingerprint,
            "feature_schema_fingerprint_sha256": prepared.schema_fingerprint,
            "missing_feature_values_handled_by_lightgbm": (prepared.missing_feature_values),
            "leakage_controls": {
                "chronological_holdout": True,
                "walk_forward_gap_days": selected.gap_days,
                "targets_excluded_from_features": True,
                "same_day_operational_features_excluded": True,
                "operational_inputs_use_lags_or_shifted_history": True,
                "feature_scaling_applied": False,
            },
            "quantile_models": [0.10, 0.50, 0.90],
            "promotion_recommendation": promotion,
            "artifacts": {
                "model": _portable_path(selected.model_path),
                "metadata": _portable_path(selected.metadata_path),
                "evaluation": _portable_path(selected.evaluation_path),
                "predictions": _portable_path(selected.predictions_path),
                "feature_importance": _portable_path(selected.feature_importance_path),
            },
        }

        result = ForecastTrainingResult(
            config=selected,
            model=final_model,
            quantile_models=quantile_models,
            predictions=predictions,
            feature_importance=importance,
            metadata=metadata,
            evaluation=evaluation,
        )
        _persist_artifacts(result)
        logger.info(
            "LightGBM training completed; recommendation=%s",
            promotion["recommendation"],
            extra={
                "event": "lightgbm_training_complete",
                "promotion_passed": promotion["passed"],
                "holdout_mae": models["lightgbm_change_forecast"].mae,
                "persistence_mae": persistence_metrics.mae,
            },
        )
        return result


def training_summary(result: ForecastTrainingResult) -> dict[str, Any]:
    """Return a compact JSON-safe CLI summary for a completed run."""

    if not isinstance(result, ForecastTrainingResult):
        raise TypeError("result must be a ForecastTrainingResult.")
    holdout = result.evaluation["holdout"]
    cross_validation = result.evaluation["cross_validation"]
    return {
        "model": result.evaluation["model"],
        "training_rows": result.metadata["training_rows"],
        "holdout_rows": holdout["rows"],
        "feature_count": result.metadata["feature_count"],
        "cross_validation": {
            "lightgbm_mean_mae": cross_validation["best_lightgbm_mean_mae"],
            "persistence_mean_mae": cross_validation["persistence_mean_mae"],
        },
        "holdout_metrics": holdout["models"],
        "prediction_interval": holdout["prediction_interval"],
        "promotion": result.evaluation["promotion"],
        "artifacts": result.metadata["artifacts"],
    }


__all__ = [
    "ABSOLUTE_TARGET_COLUMN",
    "CHANGE_TARGET_COLUMN",
    "CURRENT_LOAD_FEATURE",
    "DRIFT_LAG_FEATURE",
    "FORECAST_HORIZON_DAYS",
    "IMPORTANCE_ARTIFACT_COLUMNS",
    "PREDICTION_ARTIFACT_COLUMNS",
    "ChronologicalSplit",
    "ForecastTrainingResult",
    "ForecastingError",
    "LightGBMForecastConfig",
    "PreparedForecastData",
    "RegressionMetrics",
    "WalkForwardFold",
    "calculate_regression_metrics",
    "chronological_holdout_split",
    "construct_change_target",
    "default_candidate_parameters",
    "expanding_window_folds",
    "find_leakage_features",
    "fit_deterministic_lightgbm",
    "load_feature_data",
    "order_prediction_intervals",
    "prepare_forecasting_data",
    "reconstruct_absolute_forecast",
    "select_leakage_safe_features",
    "train_lightgbm_forecast",
    "training_summary",
    "validate_artifact_schemas",
]
