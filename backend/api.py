"""ASGI API for the HHS UAC System Capacity Analytics service.

The service uses Starlette and can be started with either command::

    uvicorn backend.api:app --host 0.0.0.0 --port 8000
    python -m backend.api

CSV analysis accepts a raw ``text/csv`` request body, avoiding a multipart-form
dependency. JSON analysis accepts a list of row objects matching the six-column
HHS input schema.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app_utils import (  # noqa: E402
    DATE_COLUMN,
    DISCHARGE_ANOMALY_COLUMN,
    GROWTH_RATE_COLUMN,
    NET_INTAKE_COLUMN,
    OFFSET_RATIO_COLUMN,
    REQUIRED_COLUMNS,
    ROLLING_14_COLUMN,
    ROLLING_7_COLUMN,
    TOTAL_LOAD_COLUMN,
    TRANSFER_ANOMALY_COLUMN,
    DataValidationError,
    generate_mock_data,
)
from backend.analytics import (  # noqa: E402
    AnalysisConfig,
    AnalysisResult,
    AnalyticsError,
    CapacityAnalyticsEngine,
    TimeGranularity,
    calculate_capacity_scenario,
)
from backend.utils import (  # noqa: E402
    dataframe_records as serialize_dataframe_records,
    get_backend_settings,
    json_safe as serialize_json_safe,
    utc_now_iso,
)
from src.forecasting import validate_prediction_schema  # noqa: E402
from src.monitoring import MonitoringConfig, evaluate_monitoring  # noqa: E402


API_TITLE = "HHS UAC Capacity Analytics API"
API_VERSION = "1.1.0"
API_PREFIX = "/api/v1"
DEFAULT_MAX_BODY_BYTES = 20 * 1024 * 1024
DEFAULT_FORECAST_ARTIFACT_ROOT = PROJECT_ROOT / "output" / "forecasting"

logger = logging.getLogger("hhs_uac.api")
if not logger.handlers:
    logging.basicConfig(
        level=os.getenv("HHS_API_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


class APIError(Exception):
    """A safe, structured error intended for an API client."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details


def _max_body_bytes() -> int:
    """Return the configured request-body limit with a safe fallback."""
    try:
        return get_backend_settings().max_request_bytes
    except ValueError as exc:
        logger.warning("Invalid backend settings; using request-size default: %s", exc)
        return DEFAULT_MAX_BODY_BYTES


def _cors_origins() -> list[str]:
    """Return configured browser origins for dashboard/API development."""
    try:
        return list(get_backend_settings().cors_origins)
    except ValueError as exc:
        logger.warning("Invalid backend settings; using CORS default: %s", exc)
        return ["http://localhost:8501", "http://127.0.0.1:8501"]


def _json_safe(value: Any) -> Any:
    """Recursively convert pandas/numpy objects into strict JSON values."""
    return serialize_json_safe(value)


def _dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Serialize a DataFrame into JSON-safe row records."""
    return serialize_dataframe_records(frame)


def _forecast_artifact_root() -> Path:
    """Return the trusted, operator-configured artifact directory."""

    configured = os.getenv("HHS_FORECAST_ARTIFACT_ROOT", "").strip()
    return Path(configured).expanduser().resolve() if configured else DEFAULT_FORECAST_ARTIFACT_ROOT


def _artifact_json(relative_path: str) -> dict[str, Any]:
    """Read a required JSON artifact without deserializing executable models."""

    path = _forecast_artifact_root() / relative_path
    if not path.is_file():
        raise APIError(
            503,
            "MODEL_ARTIFACTS_UNAVAILABLE",
            f"Required approved artifact is unavailable: {relative_path}.",
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise APIError(
            503,
            "MODEL_ARTIFACTS_INVALID",
            f"Required approved artifact cannot be read: {relative_path}.",
        ) from exc
    if not isinstance(value, dict):
        raise APIError(
            503,
            "MODEL_ARTIFACTS_INVALID",
            f"Required approved artifact is not a JSON object: {relative_path}.",
        )
    return value


def _holdout_predictions() -> pd.DataFrame:
    """Load and schema-check immutable holdout predictions for read-only serving."""

    relative = "predictions/final_holdout_predictions.csv"
    path = _forecast_artifact_root() / relative
    if not path.is_file():
        raise APIError(
            503,
            "MODEL_ARTIFACTS_UNAVAILABLE",
            f"Required approved artifact is unavailable: {relative}.",
        )
    try:
        frame = pd.read_csv(path)
        validate_prediction_schema(frame)
        frame["forecast_origin_date"] = pd.to_datetime(
            frame["forecast_origin_date"], errors="raise"
        )
        frame["target_date"] = pd.to_datetime(frame["target_date"], errors="raise")
    except (OSError, TypeError, ValueError) as exc:
        raise APIError(
            503,
            "MODEL_ARTIFACTS_INVALID",
            "Stored holdout predictions failed schema validation.",
        ) from exc
    return frame


def _request_id(request: Request) -> str:
    """Return the middleware-assigned request identifier."""
    return str(getattr(request.state, "request_id", "unavailable"))


def _response(
    request: Request,
    payload: Mapping[str, Any],
    status_code: int = 200,
) -> JSONResponse:
    """Build a standard JSON response containing the request identifier."""
    body = {"request_id": _request_id(request), **dict(payload)}
    return JSONResponse(_json_safe(body), status_code=status_code)


async def _read_limited_body(request: Request) -> bytes:
    """Read a request body after enforcing the configured byte limit."""
    maximum = _max_body_bytes()
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            declared_length = int(content_length)
        except ValueError:
            raise APIError(400, "INVALID_CONTENT_LENGTH", "Invalid Content-Length header.")
        if declared_length > maximum:
            raise APIError(
                413,
                "REQUEST_TOO_LARGE",
                f"Request body exceeds the {maximum:,}-byte limit.",
            )

    body = await request.body()
    if len(body) > maximum:
        raise APIError(
            413,
            "REQUEST_TOO_LARGE",
            f"Request body exceeds the {maximum:,}-byte limit.",
        )
    return body


async def _read_json_object(request: Request) -> dict[str, Any]:
    """Read and validate a JSON object request body."""
    body = await _read_limited_body(request)
    if not body:
        raise APIError(400, "EMPTY_BODY", "A JSON request body is required.")
    try:
        payload = await request.json()
    except Exception as exc:
        raise APIError(400, "INVALID_JSON", "Request body is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise APIError(422, "INVALID_PAYLOAD", "The JSON body must be an object.")
    return payload


def _parse_bool(value: Any, field_name: str, default: bool = False) -> bool:
    """Parse a strict boolean from JSON or query-string input."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise APIError(422, "INVALID_BOOLEAN", f"{field_name} must be true or false.")


def _parse_int(
    value: Any,
    field_name: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    """Parse and range-check an integer input."""
    if value is None:
        return default
    if isinstance(value, bool):
        raise APIError(422, "INVALID_INTEGER", f"{field_name} must be an integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise APIError(422, "INVALID_INTEGER", f"{field_name} must be an integer.") from exc
    if not minimum <= parsed <= maximum:
        raise APIError(
            422,
            "INTEGER_OUT_OF_RANGE",
            f"{field_name} must be between {minimum} and {maximum}.",
        )
    return parsed


def _parse_float(
    value: Any,
    field_name: str,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    """Parse and range-check a finite numeric input."""
    if value is None:
        return default
    if isinstance(value, bool):
        raise APIError(422, "INVALID_NUMBER", f"{field_name} must be numeric.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise APIError(422, "INVALID_NUMBER", f"{field_name} must be numeric.") from exc
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise APIError(
            422,
            "NUMBER_OUT_OF_RANGE",
            f"{field_name} must be between {minimum} and {maximum}.",
        )
    return parsed


def _analysis_config(values: Mapping[str, Any]) -> AnalysisConfig:
    """Construct a validated backend analysis configuration."""
    granularity = values.get("granularity", TimeGranularity.DAILY.value)
    threshold = _parse_int(
        values.get("backlog_threshold_days"),
        "backlog_threshold_days",
        default=3,
        minimum=1,
        maximum=365,
    )
    try:
        return AnalysisConfig(
            start_date=values.get("start_date"),
            end_date=values.get("end_date"),
            granularity=TimeGranularity.parse(granularity),
            backlog_threshold_days=threshold,
        )
    except (TypeError, ValueError) as exc:
        raise APIError(422, "INVALID_ANALYSIS_CONFIG", str(exc)) from exc


def _engine_from_json(payload: Mapping[str, Any]) -> CapacityAnalyticsEngine:
    """Create an analytics engine from JSON records or the mock-data flag."""
    use_mock = _parse_bool(payload.get("use_mock_data"), "use_mock_data", False)
    records = payload.get("records")
    if use_mock:
        return CapacityAnalyticsEngine(generate_mock_data())
    if not isinstance(records, list) or not records:
        raise APIError(
            422,
            "MISSING_RECORDS",
            "Provide a non-empty 'records' list or set 'use_mock_data' to true.",
        )
    if not all(isinstance(row, dict) for row in records):
        raise APIError(422, "INVALID_RECORDS", "Every records item must be an object.")
    try:
        return CapacityAnalyticsEngine(pd.DataFrame.from_records(records))
    except (AnalyticsError, DataValidationError, TypeError, ValueError) as exc:
        raise APIError(422, "DATA_VALIDATION_FAILED", str(exc)) from exc


def _validation_payload(result: AnalysisResult) -> dict[str, Any]:
    """Serialize validation findings from an analysis result."""
    report = result.validation_report
    if report is None:
        return {"errors": 0, "warnings": 0, "issues": []}
    return {
        "errors": report.error_count,
        "warnings": report.warning_count,
        "issues": _dataframe_records(report.to_frame()),
    }


def _analysis_payload(
    result: AnalysisResult,
    options: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a configurable API representation of an analysis result."""
    include_daily = _parse_bool(
        options.get("include_daily_metrics"), "include_daily_metrics", False
    )
    include_chart = _parse_bool(options.get("include_chart_metrics"), "include_chart_metrics", True)
    include_episodes = _parse_bool(
        options.get("include_backlog_episodes"), "include_backlog_episodes", True
    )
    include_anomalies = _parse_bool(options.get("include_anomalies"), "include_anomalies", True)

    response: dict[str, Any] = {
        "api_version": API_VERSION,
        "config": {
            "start_date": result.config.start_date,
            "end_date": result.config.end_date,
            "granularity": TimeGranularity.parse(result.config.granularity).value,
            "backlog_threshold_days": result.config.backlog_threshold_days,
        },
        "kpis": result.kpis,
        "operational_summary": result.operational_summary,
        "validation": _validation_payload(result),
    }
    data: dict[str, Any] = {}
    if include_daily:
        data["daily_metrics"] = _dataframe_records(result.daily_metrics)
    if include_chart:
        data["chart_metrics"] = _dataframe_records(result.chart_metrics)
    if include_episodes:
        data["backlog_episodes"] = _dataframe_records(result.backlog_episodes)
    if include_anomalies:
        data["anomalies"] = _dataframe_records(result.anomaly_rows)
    response["data"] = data
    return response


def _run_analysis(
    engine: CapacityAnalyticsEngine,
    values: Mapping[str, Any],
) -> AnalysisResult:
    """Run the backend engine and translate domain failures to API errors."""
    config = _analysis_config(values)
    try:
        return engine.run(config)
    except (AnalyticsError, DataValidationError, TypeError, ValueError) as exc:
        raise APIError(422, "ANALYSIS_FAILED", str(exc)) from exc


async def root_endpoint(request: Request) -> Response:
    """Return API discovery metadata."""
    return _response(
        request,
        {
            "service": API_TITLE,
            "version": API_VERSION,
            "status": "available",
            "endpoints": {
                "health": "/health",
                "schema": f"{API_PREFIX}/schema",
                "mock_data": f"{API_PREFIX}/mock-data",
                "mock_analysis": f"{API_PREFIX}/mock-analysis",
                "json_analysis": f"{API_PREFIX}/analyze",
                "csv_analysis": f"{API_PREFIX}/analyze/csv",
                "capacity_scenario": f"{API_PREFIX}/capacity-scenario",
                "model": f"{API_PREFIX}/model",
                "model_metrics": f"{API_PREFIX}/model/metrics",
                "model_provenance": f"{API_PREFIX}/model/provenance",
                "model_monitoring": f"{API_PREFIX}/model/monitoring",
                "forecast": f"{API_PREFIX}/forecast",
            },
        },
    )


async def health_endpoint(request: Request) -> Response:
    """Return a lightweight liveness response."""
    return _response(
        request,
        {
            "status": "healthy",
            "service": API_TITLE,
            "version": API_VERSION,
            "timestamp_utc": utc_now_iso(),
        },
    )


async def schema_endpoint(request: Request) -> Response:
    """Describe the request schema, derived fields, and supported options."""
    return _response(
        request,
        {
            "input_schema": {
                "required_columns": REQUIRED_COLUMNS,
                "date_format": "YYYY-MM-DD or another pandas-parseable date",
                "numeric_columns": [column for column in REQUIRED_COLUMNS if column != DATE_COLUMN],
            },
            "derived_fields": [
                TOTAL_LOAD_COLUMN,
                NET_INTAKE_COLUMN,
                GROWTH_RATE_COLUMN,
                ROLLING_7_COLUMN,
                ROLLING_14_COLUMN,
                OFFSET_RATIO_COLUMN,
                TRANSFER_ANOMALY_COLUMN,
                DISCHARGE_ANOMALY_COLUMN,
            ],
            "analysis_options": {
                "start_date": "optional inclusive date",
                "end_date": "optional inclusive date",
                "granularity": [member.value for member in TimeGranularity],
                "backlog_threshold_days": {"minimum": 1, "maximum": 365},
                "include_daily_metrics": False,
                "include_chart_metrics": True,
                "include_backlog_episodes": True,
                "include_anomalies": True,
            },
            "json_source": {
                "records": "non-empty list of input row objects",
                "use_mock_data": "boolean alternative to records",
            },
            "csv_source": "POST raw CSV bytes with Content-Type: text/csv",
        },
    )


async def mock_data_endpoint(request: Request) -> Response:
    """Return a paginated slice of the deterministic mock dataset."""
    offset = _parse_int(
        request.query_params.get("offset"),
        "offset",
        default=0,
        minimum=0,
        maximum=10_000,
    )
    limit = _parse_int(
        request.query_params.get("limit"),
        "limit",
        default=100,
        minimum=1,
        maximum=5_000,
    )
    frame = generate_mock_data()
    page = frame.iloc[offset : offset + limit]
    return _response(
        request,
        {
            "total_records": int(len(frame)),
            "offset": offset,
            "limit": limit,
            "returned_records": int(len(page)),
            "records": _dataframe_records(page),
        },
    )


async def mock_analysis_endpoint(request: Request) -> Response:
    """Run an analysis against deterministic mock data using query options."""
    values: dict[str, Any] = dict(request.query_params)
    engine = CapacityAnalyticsEngine(generate_mock_data())
    result = _run_analysis(engine, values)
    return _response(request, _analysis_payload(result, values))


async def json_analysis_endpoint(request: Request) -> Response:
    """Run an analysis from JSON records or generated mock data."""
    payload = await _read_json_object(request)
    engine = _engine_from_json(payload)
    result = _run_analysis(engine, payload)
    return _response(request, _analysis_payload(result, payload))


async def csv_analysis_endpoint(request: Request) -> Response:
    """Run an analysis from a raw CSV request body."""
    body = await _read_limited_body(request)
    if not body.strip():
        raise APIError(400, "EMPTY_CSV", "A non-empty CSV request body is required.")
    try:
        engine = CapacityAnalyticsEngine.from_csv(body)
    except (AnalyticsError, DataValidationError, TypeError, ValueError) as exc:
        raise APIError(422, "CSV_VALIDATION_FAILED", str(exc)) from exc

    values: dict[str, Any] = dict(request.query_params)
    result = _run_analysis(engine, values)
    return _response(request, _analysis_payload(result, values))


async def capacity_scenario_endpoint(request: Request) -> Response:
    """Run an analysis and evaluate user-provided planning capacities."""
    payload = await _read_json_object(request)
    cbp_capacity = _parse_int(
        payload.get("cbp_capacity"),
        "cbp_capacity",
        default=0,
        minimum=1,
        maximum=100_000_000,
    )
    hhs_capacity = _parse_int(
        payload.get("hhs_capacity"),
        "hhs_capacity",
        default=0,
        minimum=1,
        maximum=100_000_000,
    )
    warning_threshold = _parse_float(
        payload.get("warning_threshold"),
        "warning_threshold",
        default=80.0,
        minimum=0.1,
        maximum=499.9,
    )
    critical_threshold = _parse_float(
        payload.get("critical_threshold"),
        "critical_threshold",
        default=95.0,
        minimum=0.2,
        maximum=500.0,
    )
    if warning_threshold >= critical_threshold:
        raise APIError(
            422,
            "INVALID_THRESHOLDS",
            "warning_threshold must be lower than critical_threshold.",
        )

    engine = _engine_from_json(payload)
    analysis = _run_analysis(engine, payload)
    try:
        scenario = calculate_capacity_scenario(
            analysis.daily_metrics,
            cbp_capacity=cbp_capacity,
            hhs_capacity=hhs_capacity,
            warning_threshold=warning_threshold,
            critical_threshold=critical_threshold,
        )
    except (AnalyticsError, TypeError, ValueError) as exc:
        raise APIError(422, "SCENARIO_FAILED", str(exc)) from exc

    include_metrics = _parse_bool(
        payload.get("include_scenario_metrics"),
        "include_scenario_metrics",
        False,
    )
    response: dict[str, Any] = {
        "api_version": API_VERSION,
        "analysis": {
            "kpis": analysis.kpis,
            "operational_summary": analysis.operational_summary,
        },
        "scenario": {
            "summary": scenario.summary,
            "stress_episodes": _dataframe_records(scenario.stress_episodes),
        },
    }
    if include_metrics:
        response["scenario"]["metrics"] = _dataframe_records(scenario.metrics)
    return _response(request, response)


async def model_endpoint(request: Request) -> Response:
    """Expose the approved registry and promotion state as inert metadata."""

    registry = _artifact_json("models/model_registry.json")
    promotion = _artifact_json("metrics/promotion_decision.json")
    champion = _artifact_json("models/champion_model.json")
    return _response(
        request,
        {
            "api_version": API_VERSION,
            "model_version": registry.get("registry_version"),
            "champion": registry.get("champion"),
            "promotion_status": registry.get("promotion_status"),
            "promotion": promotion,
            "champion_specification": champion,
            "registry": registry,
        },
    )


async def model_metrics_endpoint(request: Request) -> Response:
    """Return frozen walk-forward, holdout, interval, and regime metrics."""

    comparison = _artifact_json("metrics/model_comparison_metrics.json")
    interval = _artifact_json("metrics/prediction_interval_metrics.json")
    regime_path = _forecast_artifact_root() / "diagnostics" / "error_by_regime.csv"
    if not regime_path.is_file():
        raise APIError(
            503,
            "MODEL_ARTIFACTS_UNAVAILABLE",
            "Required approved artifact is unavailable: diagnostics/error_by_regime.csv.",
        )
    try:
        regimes = pd.read_csv(regime_path)
    except (OSError, ValueError) as exc:
        raise APIError(
            503,
            "MODEL_ARTIFACTS_INVALID",
            "Stored regime metrics cannot be read.",
        ) from exc
    return _response(
        request,
        {
            "api_version": API_VERSION,
            "comparison": comparison,
            "prediction_intervals": interval,
            "error_by_regime": _dataframe_records(regimes),
        },
    )


async def model_provenance_endpoint(request: Request) -> Response:
    """Return dataset provenance, leakage audit, and artifact fingerprints."""

    registry = _artifact_json("models/model_registry.json")
    return _response(
        request,
        {
            "api_version": API_VERSION,
            "dataset": _artifact_json("audits/dataset_provenance.json"),
            "leakage_audit": _artifact_json("audits/leakage_audit.json"),
            "fingerprints": {
                "source_sha256": registry.get("source_sha256"),
                "data_fingerprint_sha256": registry.get("data_fingerprint_sha256"),
                "schema_fingerprint_sha256": registry.get("schema_fingerprint_sha256"),
            },
        },
    )


async def model_monitoring_endpoint(request: Request) -> Response:
    """Evaluate current artifact health without changing model artifacts."""

    try:
        result = evaluate_monitoring(
            MonitoringConfig(
                artifact_root=_forecast_artifact_root(),
                write_artifacts=False,
            )
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise APIError(
            503,
            "MODEL_MONITORING_UNAVAILABLE",
            "Monitoring could not evaluate the approved artifacts.",
        ) from exc
    return _response(request, {"api_version": API_VERSION, **result.to_dict()})


async def forecast_endpoint(request: Request) -> Response:
    """Return a stored, reproducible seven-day forecast for an origin date."""

    registry = _artifact_json("models/model_registry.json")
    promotion = _artifact_json("metrics/promotion_decision.json")
    model_name = str(registry.get("champion", ""))
    predictions = _holdout_predictions()
    champion = predictions.loc[predictions["model_name"].eq(model_name)].copy()
    if champion.empty:
        raise APIError(
            503,
            "CHAMPION_PREDICTIONS_UNAVAILABLE",
            "No stored predictions match the approved champion.",
        )
    requested = request.query_params.get("as_of")
    if requested:
        try:
            as_of = pd.Timestamp(requested)
        except (TypeError, ValueError) as exc:
            raise APIError(
                422,
                "INVALID_FORECAST_DATE",
                "as_of must be a valid ISO date such as 2025-12-14.",
            ) from exc
        if as_of.tzinfo is not None:
            as_of = as_of.tz_localize(None)
        champion = champion.loc[champion["forecast_origin_date"].eq(as_of.normalize())]
        if champion.empty:
            raise APIError(
                404,
                "FORECAST_NOT_FOUND",
                "No stored champion forecast exists for the requested origin date.",
            )
    row = champion.sort_values("forecast_origin_date").iloc[-1]
    monitoring = evaluate_monitoring(
        MonitoringConfig(artifact_root=_forecast_artifact_root(), write_artifacts=False)
    )
    forecast = {
        "forecast_origin_date": row["forecast_origin_date"],
        "target_date": row["target_date"],
        "horizon_days": int((row["target_date"] - row["forecast_origin_date"]).days),
        "prediction": row["reconstructed_absolute_prediction"],
        "lower_interval": row["lower_interval"],
        "median_prediction": row["median_prediction"],
        "upper_interval": row["upper_interval"],
        "current_load": row["current_load"],
        "actual_value": row["actual_value"],
        "evaluation_label": row["evaluation_label"],
    }
    return _response(
        request,
        {
            "api_version": API_VERSION,
            "model_version": registry.get("registry_version"),
            "configured_model": model_name,
            "active_model": monitoring.active_model,
            "model_status": monitoring.model_status,
            "promotion_status": promotion.get("recommendation"),
            "forecast": forecast,
            "scope_note": (
                "This endpoint serves frozen holdout-evaluation artifacts; it does not "
                "claim to be a live operational forecast."
            ),
        },
    )


async def api_error_handler(request: Request, exc: Exception) -> Response:
    """Return a client-safe response for explicit API errors."""
    if not isinstance(exc, APIError):
        raise exc
    payload: dict[str, Any] = {
        "request_id": _request_id(request),
        "error": {
            "code": exc.code,
            "message": exc.message,
        },
    }
    if exc.details is not None:
        payload["error"]["details"] = _json_safe(exc.details)
    return JSONResponse(payload, status_code=exc.status_code)


async def unexpected_error_handler(request: Request, exc: Exception) -> Response:
    """Log unexpected failures without exposing internal details to clients."""
    logger.exception("Unhandled API error request_id=%s", _request_id(request), exc_info=exc)
    return JSONResponse(
        {
            "request_id": _request_id(request),
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected server error occurred.",
            },
        },
        status_code=500,
    )


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach request IDs and basic security headers to every response."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        supplied_request_id = request.headers.get("X-Request-ID", "").strip()
        request.state.request_id = supplied_request_id[:128] or str(uuid.uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        return response


routes = [
    Route("/", root_endpoint, methods=["GET"]),
    Route("/health", health_endpoint, methods=["GET"]),
    Route(f"{API_PREFIX}/schema", schema_endpoint, methods=["GET"]),
    Route(f"{API_PREFIX}/mock-data", mock_data_endpoint, methods=["GET"]),
    Route(f"{API_PREFIX}/mock-analysis", mock_analysis_endpoint, methods=["GET"]),
    Route(f"{API_PREFIX}/analyze", json_analysis_endpoint, methods=["POST"]),
    Route(f"{API_PREFIX}/analyze/csv", csv_analysis_endpoint, methods=["POST"]),
    Route(
        f"{API_PREFIX}/capacity-scenario",
        capacity_scenario_endpoint,
        methods=["POST"],
    ),
    Route(f"{API_PREFIX}/model", model_endpoint, methods=["GET"]),
    Route(f"{API_PREFIX}/model/metrics", model_metrics_endpoint, methods=["GET"]),
    Route(
        f"{API_PREFIX}/model/provenance",
        model_provenance_endpoint,
        methods=["GET"],
    ),
    Route(
        f"{API_PREFIX}/model/monitoring",
        model_monitoring_endpoint,
        methods=["GET"],
    ),
    Route(f"{API_PREFIX}/forecast", forecast_endpoint, methods=["GET"]),
]

middleware = [
    Middleware(RequestContextMiddleware),
    Middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Accept", "Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
        allow_credentials=False,
    ),
]

app = Starlette(
    debug=False,
    routes=routes,
    middleware=middleware,
    exception_handlers={
        APIError: api_error_handler,
        Exception: unexpected_error_handler,
    },
)


if __name__ == "__main__":
    import uvicorn

    settings = get_backend_settings()

    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )
