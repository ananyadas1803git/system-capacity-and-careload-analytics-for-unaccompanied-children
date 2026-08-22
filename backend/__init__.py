"""Public backend API for capacity analytics and service integration.

Exports are loaded lazily so importing :mod:`backend` does not instantiate the
ASGI application, read environment settings, or import the analytics stack.
This keeps CLI commands, tests, and worker startup predictable.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "1.0.0"

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # Analytics orchestration.
    "AnalyticsError": ("backend.analytics", "AnalyticsError"),
    "TimeGranularity": ("backend.analytics", "TimeGranularity"),
    "AnalysisConfig": ("backend.analytics", "AnalysisConfig"),
    "AnalysisResult": ("backend.analytics", "AnalysisResult"),
    "CapacityScenarioResult": ("backend.analytics", "CapacityScenarioResult"),
    "filter_reporting_period": ("backend.analytics", "filter_reporting_period"),
    "resample_metrics": ("backend.analytics", "resample_metrics"),
    "calculate_backlog_episodes": (
        "backend.analytics",
        "calculate_backlog_episodes",
    ),
    "extract_anomaly_rows": ("backend.analytics", "extract_anomaly_rows"),
    "build_operational_summary": (
        "backend.analytics",
        "build_operational_summary",
    ),
    "calculate_capacity_scenario": (
        "backend.analytics",
        "calculate_capacity_scenario",
    ),
    "CapacityAnalyticsEngine": (
        "backend.analytics",
        "CapacityAnalyticsEngine",
    ),
    "run_capacity_analysis": ("backend.analytics", "run_capacity_analysis"),
    # Backend utilities.
    "DEFAULT_MAX_REQUEST_BYTES": ("backend.utils", "DEFAULT_MAX_REQUEST_BYTES"),
    "DEFAULT_MAX_DATAFRAME_ROWS": (
        "backend.utils",
        "DEFAULT_MAX_DATAFRAME_ROWS",
    ),
    "DEFAULT_CORS_ORIGINS": ("backend.utils", "DEFAULT_CORS_ORIGINS"),
    "BackendUtilityError": ("backend.utils", "BackendUtilityError"),
    "BackendSettings": ("backend.utils", "BackendSettings"),
    "PaginationMetadata": ("backend.utils", "PaginationMetadata"),
    "load_backend_settings": ("backend.utils", "load_backend_settings"),
    "get_backend_settings": ("backend.utils", "get_backend_settings"),
    "reset_backend_settings_cache": (
        "backend.utils",
        "reset_backend_settings_cache",
    ),
    "configure_logging": ("backend.utils", "configure_logging"),
    "RequestIDAdapter": ("backend.utils", "RequestIDAdapter"),
    "logger_with_request_id": ("backend.utils", "logger_with_request_id"),
    "utc_now_iso": ("backend.utils", "utc_now_iso"),
    "json_safe": ("backend.utils", "json_safe"),
    "dataframe_records": ("backend.utils", "dataframe_records"),
    "dataframe_to_csv_bytes": ("backend.utils", "dataframe_to_csv_bytes"),
    "read_csv_bytes": ("backend.utils", "read_csv_bytes"),
    "paginate_dataframe": ("backend.utils", "paginate_dataframe"),
    "dataframe_fingerprint": ("backend.utils", "dataframe_fingerprint"),
    "dataset_metadata": ("backend.utils", "dataset_metadata"),
    "require_columns": ("backend.utils", "require_columns"),
    # ASGI service. Accessing any of these explicitly imports backend.api.
    "app": ("backend.api", "app"),
    "API_TITLE": ("backend.api", "API_TITLE"),
    "API_VERSION": ("backend.api", "API_VERSION"),
    "API_PREFIX": ("backend.api", "API_PREFIX"),
    "APIError": ("backend.api", "APIError"),
}


def __getattr__(name: str) -> Any:
    """Load and cache a declared public export on first access."""

    export = _LAZY_EXPORTS.get(name)
    if export is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = export
    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return package attributes, including lazily exposed names."""

    return sorted(set(globals()) | set(_LAZY_EXPORTS))


def get_asgi_app() -> Any:
    """Return the configured ASGI application, importing it only on demand."""

    return __getattr__("app")


__all__ = ["__version__", "get_asgi_app", *_LAZY_EXPORTS]
