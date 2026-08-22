"""Reproducible orchestration for the complete analytics and forecast workflow.

The orchestrator is intentionally separate from serving code. Importing this
module never preprocesses data or trains a model; callers explicitly select a
quick fixture smoke test, approved-artifact verification, or full retraining.
"""

from __future__ import annotations

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
    TARGET_ABSOLUTE,
    TARGET_CHANGE,
    FeatureProcessor,
    ForecastConfig,
    ForecastingFrameworkError,
    chronological_partitions,
    expanding_window_folds,
    experiment_summary,
    prepare_forecast_dataset,
    reconstruct_absolute_forecast,
    regression_metrics,
    train_forecasting_models,
    validate_prediction_schema,
)
from src.preprocessor import preprocess_data
from src.validation import validate_capacity_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "HHS_Unaccompanied_Alien_Children_Program.csv"
DEFAULT_FEATURE_PATH = PROJECT_ROOT / "data" / "processed" / "uac_capacity_ml_features.parquet"
DEFAULT_PROVENANCE_PATH = PROJECT_ROOT / "data" / "processed" / "preprocessing_report.json"
DEFAULT_FORECAST_OUTPUT = PROJECT_ROOT / "output" / "forecasting"
DEFAULT_QUICK_OUTPUT = PROJECT_ROOT / "output" / "quick_pipeline"


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
        raise PipelineError(
            "Canonical feature artifact differs from regenerated features. "
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
    """Re-run data stages and verify approved model artifacts without training."""

    if not config.raw_path.is_file():
        raise PipelineError(f"Input discovery failed; raw CSV not found: {config.raw_path}")
    if not config.feature_path.is_file():
        raise PipelineError(
            f"Feature artifact not found: {config.feature_path}. Run generate-data first."
        )
    preprocessed = preprocess_data(config.raw_path)
    validation = validate_capacity_data(preprocessed.data)
    metrics = compute_capacity_metrics(preprocessed.data)
    engineered = build_feature_matrix(metrics).frame
    canonical = _verified_canonical_features(engineered, config.feature_path)
    prepared = prepare_forecast_dataset(canonical, ForecastConfig())
    required = {
        "registry": config.output_dir / "models" / "model_registry.json",
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
    comparison = json.loads(required["comparison"].read_text(encoding="utf-8"))
    promotion = json.loads(required["promotion"].read_text(encoding="utf-8"))
    leakage = json.loads(required["leakage"].read_text(encoding="utf-8"))
    predictions = pd.read_csv(required["predictions"])
    validate_prediction_schema(predictions)
    if registry.get("fingerprint_algorithm") != FINGERPRINT_ALGORITHM:
        raise PipelineError(
            "Approved model fingerprint algorithm is unsupported: "
            f"expected {FINGERPRINT_ALGORITHM!r}, got "
            f"{registry.get('fingerprint_algorithm')!r}. Retrain explicitly."
        )
    if registry.get("fingerprint_float_decimal_places") != FINGERPRINT_FLOAT_DECIMAL_PLACES:
        raise PipelineError(
            "Approved model fingerprint precision does not match the runtime. Retrain explicitly."
        )
    if registry.get("data_fingerprint_sha256") != prepared.data_fingerprint:
        raise PipelineError(
            "Approved model data fingerprint does not match regenerated features. "
            "Retrain explicitly after reviewing the data change."
        )
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
