"""Structured, context-aware logging for the HHS UAC analytics system.

The module avoids logging source rows or sensitive values by default. It supports
human-readable local output, JSON production logs, rotating files, request/run
context, audit events, timing helpers, and synchronous/asynchronous decorators.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import inspect
import json
import logging
import logging.handlers
import re
import sys
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, ParamSpec, TypeVar, cast

import numpy as np
import pandas as pd


P = ParamSpec("P")
R = TypeVar("R")

ROOT_LOGGER_NAME = "hhs_uac"
REDACTED_VALUE = "***REDACTED***"
VALID_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}

_log_context: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "hhs_uac_log_context",
    default={},
)

_SENSITIVE_KEY_PATTERN = re.compile(
    r"(?:authorization|api[_-]?key|access[_-]?token|refresh[_-]?token|"
    r"token|password|passwd|secret|cookie|session[_-]?id)",
    flags=re.IGNORECASE,
)
_SENSITIVE_TEXT_PATTERN = re.compile(
    r"(?i)\b(authorization|api[_-]?key|access[_-]?token|refresh[_-]?token|"
    r"token|password|passwd|secret|cookie|session[_-]?id)"
    r"(\s*[:=]\s*)([^\s,;]+)"
)

_STANDARD_RECORD_FIELDS = set(
    logging.LogRecord(
        name="",
        level=0,
        pathname="",
        lineno=0,
        msg="",
        args=(),
        exc_info=None,
    ).__dict__
) | {"message", "asctime"}


class LoggerConfigurationError(ValueError):
    """Raised when logger configuration is invalid."""


@dataclass(frozen=True)
class LoggingConfig:
    """Configuration for project logging handlers and formatting."""

    level: str = "INFO"
    json_output: bool = False
    console_output: bool = True
    file_path: str | Path | None = None
    max_file_bytes: int = 10 * 1024 * 1024
    backup_count: int = 5
    service_name: str = "hhs-uac-capacity-analytics"
    environment: str = "development"
    propagate: bool = False

    def __post_init__(self) -> None:
        normalized_level = self.level.strip().upper()
        if normalized_level not in VALID_LEVELS:
            raise ValueError("level must be one of: " + ", ".join(sorted(VALID_LEVELS)))
        if not self.console_output and self.file_path is None:
            raise ValueError("At least one logging output must be configured.")
        if self.max_file_bytes < 1:
            raise ValueError("max_file_bytes must be positive.")
        if self.backup_count < 0:
            raise ValueError("backup_count must be non-negative.")
        if not self.service_name.strip():
            raise ValueError("service_name must not be empty.")
        if not self.environment.strip():
            raise ValueError("environment must not be empty.")


def _utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp for structured logs."""
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    """Convert common scientific Python values into JSON-compatible data."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def redact_text(value: str) -> str:
    """Redact inline key/value secrets from a text string."""
    return _SENSITIVE_TEXT_PATTERN.sub(
        lambda match: f"{match.group(1)}{match.group(2)}{REDACTED_VALUE}",
        value,
    )


def sanitize_log_value(value: Any) -> Any:
    """Recursively redact sensitive keys and normalize log values."""
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            sanitized[key_text] = (
                REDACTED_VALUE
                if _SENSITIVE_KEY_PATTERN.search(key_text)
                else sanitize_log_value(item)
            )
        return sanitized
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        return [sanitize_log_value(item) for item in value]
    return _json_safe(value)


class ContextFilter(logging.Filter):
    """Attach context-variable fields and static service metadata."""

    def __init__(self, config: LoggingConfig) -> None:
        super().__init__()
        self.config = config

    def filter(self, record: logging.LogRecord) -> bool:
        context = _log_context.get()
        defaults = {
            "service": self.config.service_name,
            "environment": self.config.environment,
            "request_id": "system",
            "correlation_id": "unavailable",
            "run_id": "unavailable",
            "component": record.name,
            "operation": "unspecified",
        }
        for key, default in defaults.items():
            if not hasattr(record, key):
                setattr(record, key, context.get(key, default))
        for key, value in context.items():
            if key not in record.__dict__:
                setattr(record, key, value)
        return True


class SensitiveDataFilter(logging.Filter):
    """Redact common credential fields before records reach a formatter."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_text(record.msg)
        else:
            record.msg = sanitize_log_value(record.msg)

        if isinstance(record.args, Mapping):
            record.args = sanitize_log_value(record.args)
        elif isinstance(record.args, tuple):
            record.args = tuple(sanitize_log_value(item) for item in record.args)

        for key, value in list(record.__dict__.items()):
            if key in _STANDARD_RECORD_FIELDS:
                continue
            if _SENSITIVE_KEY_PATTERN.search(key):
                setattr(record, key, REDACTED_VALUE)
            else:
                setattr(record, key, sanitize_log_value(value))
        return True


class JSONLogFormatter(logging.Formatter):
    """Render each log record as one strict JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()
        payload: dict[str, Any] = {
            "timestamp": _utc_timestamp(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.message,
            "service": getattr(record, "service", None),
            "environment": getattr(record, "environment", None),
            "request_id": getattr(record, "request_id", None),
            "correlation_id": getattr(record, "correlation_id", None),
            "run_id": getattr(record, "run_id", None),
            "component": getattr(record, "component", None),
            "operation": getattr(record, "operation", None),
            "source": {
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
            },
            "process_id": record.process,
            "thread_name": record.threadName,
        }

        custom_fields = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD_RECORD_FIELDS
            and key
            not in {
                "service",
                "environment",
                "request_id",
                "correlation_id",
                "run_id",
                "component",
                "operation",
            }
        }
        if custom_fields:
            payload["details"] = sanitize_log_value(custom_fields)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        return json.dumps(_json_safe(payload), ensure_ascii=False, separators=(",", ":"))


class HumanReadableFormatter(logging.Formatter):
    """Render concise local-development logs with useful context."""

    def __init__(self) -> None:
        super().__init__(
            fmt=(
                "%(asctime)s %(levelname)s %(name)s "
                "request_id=%(request_id)s operation=%(operation)s %(message)s"
            ),
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )


def _build_handler(
    handler: logging.Handler,
    config: LoggingConfig,
) -> logging.Handler:
    """Apply level, formatters, and safety filters to a handler."""
    handler.setLevel(config.level.strip().upper())
    handler.setFormatter(JSONLogFormatter() if config.json_output else HumanReadableFormatter())
    handler.addFilter(ContextFilter(config))
    handler.addFilter(SensitiveDataFilter())
    handler._hhs_uac_managed = True  # type: ignore[attr-defined]
    return handler


def configure_logging(
    config: LoggingConfig | None = None,
    *,
    logger_name: str = ROOT_LOGGER_NAME,
    force: bool = False,
) -> logging.Logger:
    """Configure and return the project root logger.

    Repeated calls are idempotent unless ``force=True``. Only handlers created by
    this module are replaced, preserving unrelated host-application handlers.
    """
    selected = config or LoggingConfig()
    logger = logging.getLogger(logger_name)
    logger.setLevel(selected.level.strip().upper())
    logger.propagate = selected.propagate

    managed_handlers = [
        handler for handler in logger.handlers if getattr(handler, "_hhs_uac_managed", False)
    ]
    if managed_handlers and not force:
        return logger
    for handler in managed_handlers:
        logger.removeHandler(handler)
        handler.close()

    if selected.console_output:
        logger.addHandler(_build_handler(logging.StreamHandler(sys.stderr), selected))

    if selected.file_path is not None:
        file_path = Path(selected.file_path).expanduser().resolve()
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                file_path,
                maxBytes=selected.max_file_bytes,
                backupCount=selected.backup_count,
                encoding="utf-8",
            )
        except OSError as exc:
            raise LoggerConfigurationError(
                f"Unable to configure log file '{file_path}': {exc}"
            ) from exc
        logger.addHandler(_build_handler(file_handler, selected))

    return logger


class ContextLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that merges component-bound and call-specific fields."""

    def process(
        self,
        msg: object,
        kwargs: dict[str, Any],
    ) -> tuple[object, dict[str, Any]]:
        call_extra = dict(kwargs.get("extra") or {})
        kwargs["extra"] = {**self.extra, **call_extra}
        return msg, kwargs


def get_logger(
    component: str | None = None,
    *,
    config: LoggingConfig | None = None,
) -> ContextLoggerAdapter:
    """Return a configured project logger optionally bound to a component."""
    root_logger = logging.getLogger(ROOT_LOGGER_NAME)
    if not any(getattr(handler, "_hhs_uac_managed", False) for handler in root_logger.handlers):
        configure_logging(config)

    component_name = component.strip() if component else "application"
    logger_name = (
        ROOT_LOGGER_NAME
        if component_name == "application"
        else f"{ROOT_LOGGER_NAME}.{component_name}"
    )
    logger = logging.getLogger(logger_name)
    logger.setLevel(root_logger.level)
    return ContextLoggerAdapter(logger, {"component": component_name})


def new_context_id(prefix: str = "run") -> str:
    """Generate a compact unique identifier for a request, job, or model run."""
    normalized_prefix = re.sub(r"[^a-zA-Z0-9_-]", "-", prefix.strip())[:24] or "run"
    return f"{normalized_prefix}-{uuid.uuid4()}"


@contextmanager
def log_context(**fields: Any) -> Iterator[dict[str, Any]]:
    """Temporarily bind structured fields using task/thread-safe contextvars."""
    current = dict(_log_context.get())
    sanitized = {
        str(key): sanitize_log_value(value) for key, value in fields.items() if value is not None
    }
    merged = {**current, **sanitized}
    token = _log_context.set(merged)
    try:
        yield merged
    finally:
        _log_context.reset(token)


def current_log_context() -> dict[str, Any]:
    """Return a defensive copy of the active structured context."""
    return dict(_log_context.get())


class PerformanceTimer:
    """Context manager that logs operation duration and completion status."""

    def __init__(
        self,
        logger: logging.Logger | logging.LoggerAdapter,
        operation: str,
        *,
        level: int = logging.INFO,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        if not operation.strip():
            raise ValueError("operation must not be empty.")
        self.logger = logger
        self.operation = operation.strip()
        self.level = level
        self.extra = dict(extra or {})
        self._started_at: float | None = None
        self.elapsed_ms: float | None = None

    def __enter__(self) -> PerformanceTimer:
        self._started_at = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> bool:
        del traceback
        if self._started_at is None:
            return False
        self.elapsed_ms = (time.perf_counter() - self._started_at) * 1000
        details = {
            **self.extra,
            "event": "operation_completed" if exc_type is None else "operation_failed",
            "operation": self.operation,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "outcome": "success" if exc_type is None else "failure",
        }
        if exc_type is None:
            self.logger.log(
                self.level,
                "%s completed in %.3f ms",
                self.operation,
                self.elapsed_ms,
                extra=details,
            )
        else:
            details["exception_type"] = exc_type.__name__
            details["exception_message"] = str(exc_value)
            self.logger.error(
                "%s failed after %.3f ms",
                self.operation,
                self.elapsed_ms,
                extra=details,
            )
        return False


def log_execution(
    operation: str | None = None,
    *,
    logger: logging.Logger | logging.LoggerAdapter | None = None,
    level: int = logging.INFO,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorate a synchronous or asynchronous function with duration logging."""

    def decorator(function: Callable[P, R]) -> Callable[P, R]:
        selected_logger = logger or get_logger(function.__module__)
        operation_name = operation or function.__qualname__

        if inspect.iscoroutinefunction(function):

            @functools.wraps(function)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                with PerformanceTimer(selected_logger, operation_name, level=level):
                    return await function(*args, **kwargs)

            return cast(Callable[P, R], async_wrapper)

        @functools.wraps(function)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with PerformanceTimer(selected_logger, operation_name, level=level):
                return function(*args, **kwargs)

        return sync_wrapper

    return decorator


def log_audit_event(
    logger: logging.Logger | logging.LoggerAdapter,
    event: str,
    *,
    outcome: str,
    actor: str | None = None,
    resource: str | None = None,
    details: Mapping[str, Any] | None = None,
    level: int = logging.INFO,
) -> None:
    """Write a structured audit event without exposing source row contents."""
    if not event.strip() or not outcome.strip():
        raise ValueError("event and outcome must not be empty.")
    extra = {
        "event": event.strip(),
        "event_type": "audit",
        "outcome": outcome.strip(),
        "actor": actor or "system",
        "resource": resource or "unspecified",
        "audit_details": sanitize_log_value(dict(details or {})),
    }
    logger.log(level, "Audit event: %s outcome=%s", event, outcome, extra=extra)


def log_dataframe_profile(
    logger: logging.Logger | logging.LoggerAdapter,
    event: str,
    frame: pd.DataFrame,
    *,
    date_column: str = "Date",
    level: int = logging.INFO,
) -> dict[str, Any]:
    """Log safe DataFrame structure/quality metadata, never row values."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")

    if date_column in frame.columns:
        dates = pd.to_datetime(frame[date_column], errors="coerce")
    elif isinstance(frame.index, pd.DatetimeIndex):
        dates = pd.Series(frame.index)
    else:
        dates = pd.Series(dtype="datetime64[ns]")
    valid_dates = dates.dropna()
    profile: dict[str, Any] = {
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "columns": [str(column) for column in frame.columns],
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_index_values": int(frame.index.duplicated().sum()),
        "memory_bytes": int(frame.memory_usage(index=True, deep=True).sum()),
        "date_min": valid_dates.min().date().isoformat() if not valid_dates.empty else None,
        "date_max": valid_dates.max().date().isoformat() if not valid_dates.empty else None,
    }
    logger.log(
        level,
        "DataFrame profile: %s rows=%s columns=%s",
        event,
        profile["row_count"],
        profile["column_count"],
        extra={"event": event, "event_type": "dataframe_profile", **profile},
    )
    return profile


async def flush_logging() -> None:
    """Flush all managed handlers without blocking an async event loop heavily."""

    def flush_handlers() -> None:
        logger = logging.getLogger(ROOT_LOGGER_NAME)
        for handler in logger.handlers:
            if getattr(handler, "_hhs_uac_managed", False):
                handler.flush()

    await asyncio.to_thread(flush_handlers)
