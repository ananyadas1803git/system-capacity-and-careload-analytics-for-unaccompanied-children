"""Offline model monitoring and explicit fallback decisions for research use."""

from __future__ import annotations

import json
import math
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import pandas as pd

from backend.utils import json_safe, utc_now_iso


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "output" / "forecasting"


class MonitoringError(RuntimeError):
    """Raised when approved local artifacts cannot be monitored safely."""


@dataclass(frozen=True)
class MonitoringConfig:
    """Thresholds for transparent research-model degradation decisions."""

    artifact_root: Path = DEFAULT_ARTIFACT_ROOT
    rolling_window: int = 30
    degradation_ratio: float = 1.25
    input_drift_threshold: float = 0.25
    residual_drift_threshold: float = 1.0
    write_artifacts: bool = True

    def __post_init__(self) -> None:
        if self.rolling_window < 7:
            raise ValueError("rolling_window must be at least seven observations.")
        if self.degradation_ratio <= 1:
            raise ValueError("degradation_ratio must exceed one.")


@dataclass(frozen=True)
class MonitoringResult:
    """Current metrics, model status, and any explicit fallback event."""

    model_version: str
    configured_champion: str
    active_model: str
    model_status: str
    reason: str
    evaluated_at_utc: str
    metrics: Mapping[str, Any]
    latest_forecast: Mapping[str, Any]
    event: Mapping[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise MonitoringError(f"Required monitoring artifact not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MonitoringError(f"Unable to read monitoring artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise MonitoringError(f"Monitoring artifact must contain a JSON object: {path}")
    return payload


def population_stability_index(
    reference: pd.Series,
    recent: pd.Series,
    bins: int = 10,
) -> float:
    """Calculate a bounded-bin population stability index."""

    reference_values = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(float)
    recent_values = pd.to_numeric(recent, errors="coerce").dropna().to_numpy(float)
    if len(reference_values) < bins * 2 or len(recent_values) < bins:
        return 0.0
    boundaries = np.unique(np.quantile(reference_values, np.linspace(0, 1, bins + 1)))
    if len(boundaries) < 3:
        return 0.0
    boundaries[0], boundaries[-1] = -np.inf, np.inf
    reference_hist = np.histogram(reference_values, bins=boundaries)[0].astype(float)
    recent_hist = np.histogram(recent_values, bins=boundaries)[0].astype(float)
    reference_rate = np.maximum(reference_hist / reference_hist.sum(), 1e-6)
    recent_rate = np.maximum(recent_hist / recent_hist.sum(), 1e-6)
    return float(np.sum((recent_rate - reference_rate) * np.log(recent_rate / reference_rate)))


def _component_artifacts_available(
    artifact_root: Path,
    champion: str,
    champion_spec: Mapping[str, Any],
) -> tuple[bool, str]:
    if champion in {"persistence", "seven_day_drift"}:
        return True, "rule-based model requires no binary artifact"
    candidates = artifact_root / "models" / "candidates"
    if champion == "validation_weighted_ensemble":
        models = champion_spec.get("inference_rule", {}).get("models", [])
        for model in models:
            if model in {"persistence", "seven_day_drift"}:
                continue
            if not (candidates / f"{model}.joblib").is_file():
                return False, f"ensemble component artifact is missing: {model}.joblib"
        return True, "all ensemble component artifacts are present"
    path = candidates / f"{champion}.joblib"
    return (
        path.is_file(),
        "candidate artifact is present"
        if path.is_file()
        else f"candidate artifact is missing: {path.name}",
    )


def evaluate_monitoring(config: MonitoringConfig | None = None) -> MonitoringResult:
    """Evaluate saved holdout outcomes and choose a transparent active fallback."""

    selected = config or MonitoringConfig()
    root = selected.artifact_root.expanduser().resolve()
    registry = _read_json(root / "models" / "model_registry.json")
    promotion = _read_json(root / "metrics" / "promotion_decision.json")
    champion_spec = _read_json(root / "models" / "champion_model.json")
    predictions_path = root / "predictions" / "final_holdout_predictions.csv"
    if not predictions_path.is_file():
        raise MonitoringError(f"Prediction artifact not found: {predictions_path}")
    predictions = pd.read_csv(
        predictions_path,
        parse_dates=["forecast_origin_date", "target_date"],
    )
    configured_champion = str(registry.get("champion", "persistence"))
    champion_rows = predictions.loc[predictions["model_name"] == configured_champion].sort_values(
        "forecast_origin_date"
    )
    persistence_rows = predictions.loc[predictions["model_name"] == "persistence"].sort_values(
        "forecast_origin_date"
    )
    if champion_rows.empty or persistence_rows.empty:
        raise MonitoringError("Champion and persistence predictions must both be present.")
    aligned = champion_rows.merge(
        persistence_rows[["forecast_origin_date", "reconstructed_absolute_prediction"]],
        on="forecast_origin_date",
        suffixes=("_champion", "_persistence"),
        validate="one_to_one",
    )
    window = min(selected.rolling_window, len(aligned))
    recent = aligned.tail(window)
    actual = recent["actual_value"].to_numpy(float)
    champion_forecast = recent["reconstructed_absolute_prediction_champion"].to_numpy(float)
    persistence_forecast = recent["reconstructed_absolute_prediction_persistence"].to_numpy(float)
    champion_mae = float(np.mean(np.abs(actual - champion_forecast)))
    persistence_mae = float(np.mean(np.abs(actual - persistence_forecast)))
    rolling_mase = champion_mae / persistence_mae if persistence_mae > 0 else 1.0
    coverage = float(
        np.mean(
            (actual >= recent["lower_interval"].to_numpy(float))
            & (actual <= recent["upper_interval"].to_numpy(float))
        )
        * 100
    )
    midpoint = max(1, len(aligned) // 2)
    input_psi = population_stability_index(
        aligned.iloc[:midpoint]["current_load"],
        aligned.iloc[midpoint:]["current_load"],
    )
    residuals = aligned["actual_value"] - aligned["reconstructed_absolute_prediction_champion"]
    reference_residual = residuals.iloc[:midpoint]
    recent_residual = residuals.iloc[midpoint:]
    reference_scale = float(reference_residual.std(ddof=0))
    residual_shift = (
        float(abs(recent_residual.mean() - reference_residual.mean()) / reference_scale)
        if reference_scale > 0
        else 0.0
    )
    missing_feature_rate = 0.0
    imputation_rate = float(recent["is_imputed_date"].astype(bool).mean() * 100)
    promoted = bool(promotion.get("passed"))
    registry_matches = configured_champion == promotion.get("champion_model")
    artifacts_available, artifact_reason = _component_artifacts_available(
        root, configured_champion, champion_spec
    )
    degraded = champion_mae > persistence_mae * selected.degradation_ratio

    active_model = configured_champion
    status = "approved"
    reason = "Promoted model is available and within the rolling performance threshold."
    event_type: str | None = None
    if not promoted:
        active_model = "persistence"
        status = "fallback"
        reason = "No learned or ensemble candidate passed the promotion rules."
        event_type = "fallback"
    elif not registry_matches:
        active_model = "persistence"
        status = "fallback"
        reason = "Registry champion and promotion decision disagree."
        event_type = "fallback"
    elif not artifacts_available:
        active_model = "persistence"
        status = "fallback"
        reason = artifact_reason
        event_type = "fallback"
    elif degraded:
        active_model = "persistence"
        status = "degraded"
        reason = (
            f"Rolling champion MAE is more than {selected.degradation_ratio:.2f}x "
            "the persistence MAE."
        )
        event_type = "degradation_and_fallback"

    latest = champion_rows.iloc[-1]
    timestamp = utc_now_iso()
    event = (
        {
            "event_type": event_type,
            "timestamp_utc": timestamp,
            "configured_champion": configured_champion,
            "active_model": active_model,
            "reason": reason,
        }
        if event_type
        else None
    )
    metrics = {
        "rolling_window": window,
        "rolling_mae": champion_mae,
        "rolling_persistence_mae": persistence_mae,
        "rolling_mase": rolling_mase,
        "interval_coverage_percent": coverage,
        "missing_feature_rate_percent": missing_feature_rate,
        "imputation_rate_percent": imputation_rate,
        "input_distribution_psi": input_psi,
        "residual_standardized_mean_shift": residual_shift,
        "input_drift_flag": input_psi > selected.input_drift_threshold,
        "residual_drift_flag": residual_shift > selected.residual_drift_threshold,
        "performance_degraded": degraded,
    }
    for key, value in metrics.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise MonitoringError(f"Monitoring metric is not finite: {key}")
    result = MonitoringResult(
        model_version=str(registry.get("registry_version", "unknown")),
        configured_champion=configured_champion,
        active_model=active_model,
        model_status=status,
        reason=reason,
        evaluated_at_utc=timestamp,
        metrics=metrics,
        latest_forecast={
            "forecast_origin": latest["forecast_origin_date"],
            "target_date": latest["target_date"],
            "prediction": latest["reconstructed_absolute_prediction"],
            "lower_interval": latest["lower_interval"],
            "median_prediction": latest["median_prediction"],
            "upper_interval": latest["upper_interval"],
            "actual": latest["actual_value"],
        },
        event=event,
    )
    if selected.write_artifacts:
        monitoring_dir = root / "monitoring"
        payload = result.to_dict()
        with _atomic_target(monitoring_dir / "monitoring_summary.json") as temporary:
            temporary.write_text(
                json.dumps(payload, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        if event is not None:
            event_path = monitoring_dir / "model_events.jsonl"
            existing = event_path.read_text(encoding="utf-8") if event_path.is_file() else ""
            with _atomic_target(event_path) as temporary:
                temporary.write_text(
                    existing + json.dumps(json_safe(event), allow_nan=False) + "\n",
                    encoding="utf-8",
                )
    return result


__all__ = [
    "DEFAULT_ARTIFACT_ROOT",
    "MonitoringConfig",
    "MonitoringError",
    "MonitoringResult",
    "evaluate_monitoring",
    "population_stability_index",
]
