"""Reproducible orchestration for the complete analytics and forecast workflow.

The orchestrator is intentionally separate from serving code. Importing this
module never preprocesses data or trains a model; callers explicitly select a
quick fixture smoke test, approved-artifact verification, or full retraining.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app_utils import compute_capacity_metrics, generate_synthetic_data
from backend.utils import json_safe, utc_now_iso
from src.feature_engineering import build_feature_matrix
from src.forecasting import (
    CURRENT_LOAD,
    FINGERPRINT_ALGORITHM,
    FINGERPRINT_FLOAT_DECIMAL_PLACES,
    PREPARED_DATASET_CONTRACT_VERSION,
    PREPARED_FEATURE_RECIPE_VERSION,
    TARGET_ABSOLUTE,
    TARGET_CHANGE,
    FeatureProcessor,
    ForecastConfig,
    ForecastingFrameworkError,
    chronological_partitions,
    expanding_window_folds,
    experiment_summary,
    forecast_configuration_fingerprint,
    prepare_forecast_dataset,
    reconstruct_absolute_forecast,
    regression_metrics,
    train_forecasting_models,
    validate_prediction_schema,
    _frame_fingerprint,
    _schema_fingerprint,
    _semantic_dtype,
)
from src.preprocessor import preprocess_data
from src.validation import validate_capacity_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "HHS_Unaccompanied_Alien_Children_Program.csv"
DEFAULT_FEATURE_PATH = PROJECT_ROOT / "data" / "processed" / "uac_capacity_ml_features.parquet"
DEFAULT_PROVENANCE_PATH = PROJECT_ROOT / "data" / "processed" / "preprocessing_report.json"
DEFAULT_FORECAST_OUTPUT = PROJECT_ROOT / "output" / "forecasting"
DEFAULT_QUICK_OUTPUT = PROJECT_ROOT / "output" / "quick_pipeline"

# Regenerated rolling, polynomial-fit, and trigonometric features can vary by
# platform below any operationally meaningful resolution. These comparison
# tolerances are deliberately separate from the fingerprint precision: targets,
# dates, missingness, schema, and non-numeric values remain exact.
PREPARED_FRAME_RELATIVE_TOLERANCE = 1e-12
PREPARED_FRAME_ABSOLUTE_TOLERANCE = 1e-9
EXACT_PREPARED_COLUMNS = frozenset({TARGET_ABSOLUTE, TARGET_CHANGE, "target_date"})


class PipelineError(RuntimeError):
    """Raised when an orchestration stage cannot complete safely."""


@dataclass(frozen=True)
class PipelineConfig:
    """Paths and execution mode for the research pipeline."""

    raw_path: Path = DEFAULT_RAW_PATH
    feature_path: Path = DEFAULT_FEATURE_PATH
    provenance_path: Path = DEFAULT_PROVENANCE_PATH
    output_dir: Path = DEFAULT_FORECAST_OUTPUT
    quick: bool = False
    train_models: bool = False
    overwrite: bool = False
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.quick and self.train_models:
            raise ValueError("Quick mode and full model training are mutually exclusive.")


@dataclass(frozen=True)
class PipelineResult:
    """JSON-safe result of all completed workflow stages."""

    mode: str
    stages: Mapping[str, Any]
    artifacts: Mapping[str, str]
    completed_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "status": "complete",
                "mode": self.mode,
                "stages": self.stages,
                "artifacts": self.artifacts,
                "completed_at_utc": self.completed_at_utc,
            }
        )


@contextmanager
def _atomic_target(path: Path) -> Iterator[Path]:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.unlink(missing_ok=True)
        yield temporary
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with _atomic_target(path) as temporary:
        temporary.write_text(
            json.dumps(json_safe(payload), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    with _atomic_target(path) as temporary:
        frame.to_csv(temporary, index=False, date_format="%Y-%m-%d", lineterminator="\n")


def _file_sha256(path: Path) -> str:
    """Return an exact byte-level SHA-256 for an immutable input artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file_sha256(path: Path, expected: str | None, label: str) -> str:
    """Verify an exact file hash and report both hashes on failure."""

    actual = _file_sha256(path)
    if not expected or actual != expected:
        raise PipelineError(
            f"{label} SHA-256 mismatch: expected={expected!r}; actual={actual!r}; path={path}."
        )
    return actual


def _normalize_prepared_datetimes(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize datetime storage units to nanoseconds without rounding values."""

    normalized = frame.copy()
    index = pd.to_datetime(normalized.index, errors="coerce")
    if index.isna().any():
        raise PipelineError("Prepared forecasting frame contains invalid index dates.")
    normalized.index = pd.DatetimeIndex(index).as_unit("ns")
    for column in normalized.columns:
        if pd.api.types.is_datetime64_any_dtype(normalized[column].dtype):
            values = pd.to_datetime(normalized[column], errors="coerce")
            if values.isna().sum() != normalized[column].isna().sum():
                raise PipelineError(f"Prepared datetime column {column!r} is invalid.")
            normalized[column] = values.astype("datetime64[ns]")
    return normalized


def _prepared_hash_context(canonical: pd.DataFrame, regenerated: pd.DataFrame) -> str:
    """Return expected/actual diagnostic hashes for comparison errors."""

    try:
        expected = _frame_fingerprint(canonical)
    except (TypeError, ValueError):
        expected = "unavailable"
    try:
        actual = _frame_fingerprint(regenerated)
    except (TypeError, ValueError):
        actual = "unavailable"
    return f"expected_hash={expected}; actual_hash={actual}"


def _assert_prepared_frames_equivalent(
    regenerated: pd.DataFrame,
    canonical: pd.DataFrame,
) -> None:
    """Verify a fresh prepared frame against the exact training-time frame.

    Structure, date positions, missingness, targets, strings, and booleans are
    exact. Only non-target numeric derived features receive the documented
    1e-12 relative / 1e-9 absolute cross-platform tolerance.
    """

    actual = _normalize_prepared_datetimes(regenerated)
    expected = _normalize_prepared_datetimes(canonical)
    hashes = _prepared_hash_context(expected, actual)

    if actual.index.name != expected.index.name:
        raise PipelineError(
            "Prepared index schema changed: "
            f"expected_name={expected.index.name!r}; actual_name={actual.index.name!r}; {hashes}."
        )
    if list(actual.columns) != list(expected.columns):
        raise PipelineError(f"Prepared column order or names changed; {hashes}.")
    if actual.shape != expected.shape:
        raise PipelineError(
            "Prepared row or column count changed: "
            f"expected_shape={expected.shape}; actual_shape={actual.shape}; {hashes}."
        )

    expected_schema = _schema_fingerprint(expected)
    actual_schema = _schema_fingerprint(actual)
    if actual_schema != expected_schema:
        raise PipelineError(
            "Prepared schema fingerprint changed: "
            f"expected={expected_schema}; actual={actual_schema}; {hashes}."
        )
    if not actual.index.equals(expected.index):
        raise PipelineError(f"Prepared row dates or order changed; {hashes}.")

    expected_missing = expected.isna()
    actual_missing = actual.isna()
    if not actual_missing.equals(expected_missing):
        raise PipelineError(f"Prepared missing-value locations changed; {hashes}.")

    for column in expected.columns:
        semantic = _semantic_dtype(expected[column])
        expected_values = expected[column]
        actual_values = actual[column]
        if column in EXACT_PREPARED_COLUMNS:
            equal = expected_values.equals(actual_values)
        elif semantic == "numeric":
            equal = bool(
                np.isclose(
                    pd.to_numeric(actual_values, errors="raise").to_numpy(float),
                    pd.to_numeric(expected_values, errors="raise").to_numpy(float),
                    rtol=PREPARED_FRAME_RELATIVE_TOLERANCE,
                    atol=PREPARED_FRAME_ABSOLUTE_TOLERANCE,
                    equal_nan=True,
                ).all()
            )
        else:
            # Booleans, strings, and normalized nanosecond datetimes remain exact.
            equal = expected_values.equals(actual_values)
        if not equal:
            raise PipelineError(f"Prepared values changed in column {column!r}; {hashes}.")


def _read_canonical_prepared_frame(path: Path) -> pd.DataFrame:
    """Load the immutable training-time prepared frame without recomputation."""

    try:
        frame = pd.read_parquet(path, engine="pyarrow")
    except (ImportError, OSError, ValueError) as exc:
        raise PipelineError(f"Unable to read canonical prepared frame: {exc}") from exc
    if frame.empty:
        raise PipelineError("Canonical prepared forecasting frame is empty.")
    return frame


def _assert_finite(payload: Any, location: str = "root") -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            _assert_finite(value, f"{location}.{key}")
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            _assert_finite(value, f"{location}[{index}]")
    elif isinstance(payload, float) and not math.isfinite(payload):
        raise PipelineError(f"Non-finite value generated at {location}.")


def _verified_canonical_features(
    regenerated: pd.DataFrame,
    feature_path: Path,
) -> pd.DataFrame:
    """Load the canonical feature artifact after verifying regenerated values.

    Parquet may restore a DatetimeIndex at microsecond resolution while an
    in-memory pandas transform uses nanoseconds. That representational detail
    changes a byte-level fingerprint even when dates and feature values are
    identical, so the comparison normalizes the index and checks values before
    returning the exact artifact that was used for model training.
    """

    try:
        canonical = pd.read_parquet(feature_path, engine="pyarrow")
    except (ImportError, OSError, ValueError) as exc:
        raise PipelineError(f"Unable to read canonical feature artifact: {exc}") from exc
    regenerated_check = regenerated.copy()
    canonical_check = canonical.copy()
    regenerated_check.index = pd.DatetimeIndex(regenerated_check.index).as_unit("ns")
    canonical_check.index = pd.DatetimeIndex(canonical_check.index).as_unit("ns")
    try:
        pd.testing.assert_frame_equal(
            regenerated_check,
            canonical_check,
            check_dtype=False,
            check_freq=False,
            check_exact=False,
            # pandas rolling variance can differ by a few trillionths across
            # supported pandas/Arrow builds; this tolerance is many orders of
            # magnitude below one child and cannot mask a material data change.
            rtol=1e-9,
            atol=1e-9,
        )
    except AssertionError as exc:
        expected_hash = _frame_fingerprint(canonical_check)
        actual_hash = _frame_fingerprint(regenerated_check)
        raise PipelineError(
            "Canonical feature artifact differs from regenerated features. "
            f"expected_hash={expected_hash}; actual_hash={actual_hash}. "
            "Run generate-data and review its audit before continuing."
        ) from exc
    return canonical


def _quick_pipeline(config: PipelineConfig) -> PipelineResult:
    """Run every pipeline class of operation on a small deterministic fixture."""

    raw = generate_synthetic_data("2023-01-01", "2024-06-30", seed=config.random_seed)
    preprocessed = preprocess_data(raw)
    validation = validate_capacity_data(preprocessed.data)
    metrics = compute_capacity_metrics(preprocessed.data)
    engineered = build_feature_matrix(metrics).frame
    forecast_config = ForecastConfig(
        output_dir=config.output_dir,
        cv_splits=4,
        cv_test_size=28,
        random_seed=config.random_seed,
        overwrite=True,
    )
    prepared = prepare_forecast_dataset(engineered, forecast_config)
    partitions = chronological_partitions(prepared, forecast_config)
    folds = expanding_window_folds(len(partitions.development), forecast_config)
    fold = folds[-1]
    train = partitions.development.iloc[fold.train_indices]
    validation_frame = partitions.development.iloc[fold.validation_indices]
    processor = FeatureProcessor(
        forecast_config.maximum_missing_fraction,
        forecast_config.correlation_threshold,
    ).fit(train.loc[:, prepared.compact_features])
    x_train = processor.transform(train.loc[:, prepared.compact_features])
    x_validation = processor.transform(validation_frame.loc[:, prepared.compact_features])
    model = Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=100.0))]).fit(
        x_train, train[TARGET_CHANGE]
    )
    predicted_change = np.asarray(model.predict(x_validation), dtype=float)
    predicted = reconstruct_absolute_forecast(
        validation_frame[CURRENT_LOAD].to_numpy(float), predicted_change
    )
    actual = validation_frame[TARGET_ABSOLUTE].to_numpy(float)
    persistence = validation_frame[CURRENT_LOAD].to_numpy(float)
    persistence_mae = float(mean_absolute_error(actual, persistence))
    model_metrics = regression_metrics(actual, predicted, persistence_mae)
    predictions = pd.DataFrame(
        {
            "forecast_origin_date": validation_frame.index,
            "target_date": validation_frame["target_date"].to_numpy(),
            "actual_value": actual,
            "current_load": validation_frame[CURRENT_LOAD].to_numpy(float),
            "model_name": "quick_ridge",
            "predicted_change_7d": predicted_change,
            "reconstructed_absolute_prediction": predicted,
            "persistence_prediction": persistence,
            "lower_interval": predicted - persistence_mae,
            "median_prediction": predicted,
            "upper_interval": predicted + persistence_mae,
            "evaluation_label": "quick_fixture",
            "fold": fold.number,
            "is_imputed_date": validation_frame["is_imputed_date"].to_numpy(bool),
            "has_anomaly": validation_frame["has_anomaly"].to_numpy(bool),
            "backlog_state": validation_frame["backlog_state"].to_numpy(bool),
            "capacity_stress": validation_frame["capacity_stress"].to_numpy(bool),
        }
    )
    validate_prediction_schema(predictions)
    summary = {
        "fixture": "deterministic synthetic data",
        "source_rows": len(raw),
        "processed_rows": len(preprocessed.data),
        "validation_status": validation.report.status,
        "engineered_columns": len(engineered.columns),
        "eligible_rows": len(prepared.frame),
        "walk_forward_folds": len(folds),
        "gap_rows": fold.observed_gap,
        "selected_feature_count": len(processor.selected_columns),
        "quick_model_metrics": model_metrics,
        "leakage_checks": {
            "no_target_features": not any(
                "target" in name or "future" in name for name in prepared.expanded_features
            ),
            "chronological": train.index.max() < validation_frame.index.min(),
            "seven_day_gap": fold.observed_gap == 7,
            "preprocessing_fit_ends_before_validation": (
                pd.Timestamp(processor.fitted_rows[1]) < validation_frame.index.min()
            ),
        },
        "random_seed": config.random_seed,
    }
    if not all(summary["leakage_checks"].values()):
        raise PipelineError("Quick-pipeline leakage invariant failed.")
    _assert_finite(summary)
    output = config.output_dir.expanduser().resolve()
    summary_path = output / "quick_pipeline_summary.json"
    predictions_path = output / "quick_predictions.csv"
    model_path = output / "quick_ridge.joblib"
    _write_json(summary_path, summary)
    _write_csv(predictions_path, predictions)
    with _atomic_target(model_path) as temporary:
        joblib.dump({"model": model, "processor": processor}, temporary)
    return PipelineResult(
        mode="quick",
        stages=summary,
        artifacts={
            "summary": str(summary_path),
            "predictions": str(predictions_path),
            "model": str(model_path),
        },
        completed_at_utc=utc_now_iso(),
    )


def _load_approved_artifacts(config: PipelineConfig) -> PipelineResult:
    """Verify exact inputs and a canonical training-time prepared dataset."""

    if not config.raw_path.is_file():
        raise PipelineError(f"Input discovery failed; raw CSV not found: {config.raw_path}")
    if not config.feature_path.is_file():
        raise PipelineError(
            f"Feature artifact not found: {config.feature_path}. Run generate-data first."
        )
    required = {
        "registry": config.output_dir / "models" / "model_registry.json",
        "canonical_prepared_frame": (
            config.output_dir / "audits" / "canonical_prepared_forecast_frame.parquet"
        ),
        "comparison": config.output_dir / "metrics" / "model_comparison_metrics.json",
        "promotion": config.output_dir / "metrics" / "promotion_decision.json",
        "leakage": config.output_dir / "audits" / "leakage_audit.json",
        "predictions": config.output_dir / "predictions" / "final_holdout_predictions.csv",
        "report": config.output_dir / "forecast_model_report.html",
    }
    missing = [path for path in required.values() if not path.is_file()]
    if missing:
        raise PipelineError(
            "Approved forecasting artifacts are missing: "
            + ", ".join(str(path) for path in missing)
            + ". Run: python main.py pipeline --train --force"
        )
    registry = json.loads(required["registry"].read_text(encoding="utf-8"))
    contract = registry.get("prepared_dataset_contract")
    if not isinstance(contract, Mapping):
        raise PipelineError("Approved model registry has no prepared-dataset contract.")
    if registry.get("verification_contract_version") != PREPARED_DATASET_CONTRACT_VERSION:
        raise PipelineError(
            "Approved verification contract is unsupported: "
            f"expected={PREPARED_DATASET_CONTRACT_VERSION!r}; "
            f"actual={registry.get('verification_contract_version')!r}. Retrain explicitly."
        )
    if registry.get("fingerprint_algorithm") != FINGERPRINT_ALGORITHM:
        raise PipelineError(
            "Approved model fingerprint algorithm is unsupported: "
            f"expected={FINGERPRINT_ALGORITHM!r}; "
            f"actual={registry.get('fingerprint_algorithm')!r}. Retrain explicitly."
        )
    if registry.get("fingerprint_float_decimal_places") != FINGERPRINT_FLOAT_DECIMAL_PLACES:
        raise PipelineError(
            "Approved model fingerprint precision does not match the runtime: "
            f"expected={FINGERPRINT_FLOAT_DECIMAL_PLACES!r}; "
            f"actual={registry.get('fingerprint_float_decimal_places')!r}. Retrain explicitly."
        )

    _verify_file_sha256(config.raw_path, registry.get("source_sha256"), "Raw source")
    _verify_file_sha256(
        config.feature_path,
        contract.get("processed_feature_artifact_sha256"),
        "Processed feature Parquet",
    )
    _verify_file_sha256(
        required["canonical_prepared_frame"],
        contract.get("artifact_sha256"),
        "Canonical prepared Parquet",
    )

    forecast_config = ForecastConfig(
        feature_path=config.feature_path,
        raw_path=config.raw_path,
        provenance_path=config.provenance_path,
        output_dir=config.output_dir,
        random_seed=config.random_seed,
    )
    actual_config_hash = forecast_configuration_fingerprint(forecast_config)
    expected_config_hash = contract.get("forecast_configuration_sha256")
    if actual_config_hash != expected_config_hash:
        raise PipelineError(
            "Forecast configuration fingerprint mismatch: "
            f"expected={expected_config_hash!r}; actual={actual_config_hash!r}."
        )
    if contract.get("feature_recipe_version") != PREPARED_FEATURE_RECIPE_VERSION:
        raise PipelineError(
            "Prepared feature recipe is unsupported: "
            f"expected={PREPARED_FEATURE_RECIPE_VERSION!r}; "
            f"actual={contract.get('feature_recipe_version')!r}."
        )

    canonical_prepared = _read_canonical_prepared_frame(required["canonical_prepared_frame"])
    canonical_normalized = _normalize_prepared_datetimes(canonical_prepared)
    actual_row_count = len(canonical_normalized)
    expected_row_count = contract.get("row_count")
    if actual_row_count != expected_row_count:
        raise PipelineError(
            "Canonical prepared row count mismatch: "
            f"expected={expected_row_count!r}; actual={actual_row_count!r}."
        )
    actual_date_range = {
        "start": canonical_normalized.index.min().isoformat(),
        "end": canonical_normalized.index.max().isoformat(),
    }
    expected_date_range = contract.get("date_range")
    if actual_date_range != expected_date_range:
        raise PipelineError(
            "Canonical prepared date range mismatch: "
            f"expected={expected_date_range!r}; actual={actual_date_range!r}."
        )

    # Rebuild from raw data on this platform. The processed Parquet remains an
    # independently byte-verified lineage checkpoint, while the fresh prepared
    # frame exercises the complete transformations that previously destabilized CI.
    preprocessed = preprocess_data(config.raw_path)
    validation = validate_capacity_data(preprocessed.data)
    metrics = compute_capacity_metrics(preprocessed.data)
    engineered = build_feature_matrix(metrics).frame
    _verified_canonical_features(engineered, config.feature_path)
    regenerated_prepared = prepare_forecast_dataset(engineered, forecast_config)
    _assert_prepared_frames_equivalent(regenerated_prepared.frame, canonical_normalized)

    # Only after the tolerant, structurally strict comparison succeeds do we
    # validate the registry fingerprint against stable canonical artifact values.
    canonical_data_hash = _frame_fingerprint(canonical_normalized)
    expected_data_hash = contract.get("data_fingerprint_sha256")
    if canonical_data_hash != expected_data_hash:
        raise PipelineError(
            "Canonical prepared data fingerprint mismatch: "
            f"expected={expected_data_hash!r}; actual={canonical_data_hash!r}."
        )
    if registry.get("data_fingerprint_sha256") != canonical_data_hash:
        raise PipelineError(
            "Top-level registry data fingerprint mismatch: "
            f"expected={canonical_data_hash!r}; "
            f"actual={registry.get('data_fingerprint_sha256')!r}."
        )
    canonical_schema_hash = _schema_fingerprint(canonical_normalized)
    expected_schema_hash = contract.get("schema_fingerprint_sha256")
    if canonical_schema_hash != expected_schema_hash:
        raise PipelineError(
            "Canonical prepared schema fingerprint mismatch: "
            f"expected={expected_schema_hash!r}; actual={canonical_schema_hash!r}."
        )
    if registry.get("schema_fingerprint_sha256") != canonical_schema_hash:
        raise PipelineError(
            "Top-level registry schema fingerprint mismatch: "
            f"expected={canonical_schema_hash!r}; "
            f"actual={registry.get('schema_fingerprint_sha256')!r}."
        )

    comparison = json.loads(required["comparison"].read_text(encoding="utf-8"))
    promotion = json.loads(required["promotion"].read_text(encoding="utf-8"))
    leakage = json.loads(required["leakage"].read_text(encoding="utf-8"))
    predictions = pd.read_csv(required["predictions"])
    validate_prediction_schema(predictions)
    if registry.get("champion") != promotion.get("champion_model"):
        raise PipelineError("Champion registry and promotion decision disagree.")
    if not leakage.get("passed"):
        raise PipelineError("Stored leakage audit is not approved.")
    stage_summary = {
        "input_discovery": str(config.raw_path),
        "preprocessed_rows": len(preprocessed.data),
        "validation_status": validation.report.status,
        "engineered_columns": len(engineered.columns),
        "model_action": "loaded_and_verified_approved_artifacts",
        "fingerprint_algorithm": FINGERPRINT_ALGORITHM,
        "verification_contract_version": PREPARED_DATASET_CONTRACT_VERSION,
        "canonical_data_fingerprint_sha256": canonical_data_hash,
        "regenerated_data_fingerprint_sha256": regenerated_prepared.data_fingerprint,
        "champion": registry["champion"],
        "promotion_status": registry["promotion_status"],
        "models_compared": len(comparison["models"]),
        "holdout_prediction_rows": len(predictions),
        "leakage_audit_passed": True,
        "report_verified": True,
    }
    return PipelineResult(
        mode="approved-artifact-verification",
        stages=stage_summary,
        artifacts={name: str(path) for name, path in required.items()},
        completed_at_utc=utc_now_iso(),
    )


def run_pipeline(config: PipelineConfig | None = None) -> PipelineResult:
    """Run quick validation, approved-artifact verification, or full training."""

    selected = config or PipelineConfig()
    try:
        if selected.quick:
            return _quick_pipeline(selected)
        if selected.train_models:
            if not selected.raw_path.is_file():
                raise PipelineError(
                    f"Input discovery failed; raw CSV not found: {selected.raw_path}"
                )
            preprocessed = preprocess_data(selected.raw_path)
            validation = validate_capacity_data(preprocessed.data)
            metrics = compute_capacity_metrics(preprocessed.data)
            engineered = build_feature_matrix(metrics).frame
            # Use the canonical feature artifact only after proving that fresh
            # preprocessing and feature engineering reproduce its values.
            _verified_canonical_features(engineered, selected.feature_path)
            result = train_forecasting_models(
                ForecastConfig(
                    feature_path=selected.feature_path,
                    raw_path=selected.raw_path,
                    provenance_path=selected.provenance_path,
                    output_dir=selected.output_dir,
                    random_seed=selected.random_seed,
                    overwrite=selected.overwrite,
                )
            )
            summary = experiment_summary(result)
            summary["pre_training_validation_status"] = validation.report.status
            return PipelineResult(
                mode="full-training",
                stages=summary,
                artifacts=result.artifact_paths,
                completed_at_utc=utc_now_iso(),
            )
        return _load_approved_artifacts(selected)
    except (
        ForecastingFrameworkError,
        ImportError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        if isinstance(exc, PipelineError):
            raise
        raise PipelineError(str(exc)) from exc


__all__ = [
    "DEFAULT_FORECAST_OUTPUT",
    "DEFAULT_QUICK_OUTPUT",
    "PipelineConfig",
    "PipelineError",
    "PipelineResult",
    "run_pipeline",
]
