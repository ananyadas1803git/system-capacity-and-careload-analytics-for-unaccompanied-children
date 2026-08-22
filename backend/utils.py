"""Shared backend infrastructure utilities for the HHS UAC analytics system.

This module intentionally contains no Streamlit or endpoint-specific logic. It
provides reusable helpers for the API, batch jobs, tests, and future persistence
layers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from functools import lru_cache
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_MAX_REQUEST_BYTES = 20 * 1024 * 1024
DEFAULT_MAX_DATAFRAME_ROWS = 1_000_000
DEFAULT_CORS_ORIGINS = (
    "http://localhost:8501",
    "http://127.0.0.1:8501",
)
VALID_LOG_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}


class BackendUtilityError(ValueError):
    """Raised when shared backend input or configuration is invalid."""


@dataclass(frozen=True)
class BackendSettings:
    """Validated runtime configuration loaded from environment variables."""

    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    environment: str = "development"
    cors_origins: tuple[str, ...] = DEFAULT_CORS_ORIGINS
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES
    max_dataframe_rows: int = DEFAULT_MAX_DATAFRAME_ROWS


@dataclass(frozen=True)
class PaginationMetadata:
    """JSON-friendly metadata for a paginated DataFrame slice."""

    total_records: int
    offset: int
    limit: int
    returned_records: int
    has_previous: bool
    has_next: bool


def _environment_int(
    environment: Mapping[str, str],
    name: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    """Read and validate a bounded integer environment variable."""
    raw_value = environment.get(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        parsed = int(raw_value)
    except ValueError as exc:
        raise BackendUtilityError(f"{name} must be an integer.") from exc
    if not minimum <= parsed <= maximum:
        raise BackendUtilityError(
            f"{name} must be between {minimum} and {maximum}."
        )
    return parsed


def load_backend_settings(
    environment: Mapping[str, str] | None = None,
) -> BackendSettings:
    """Load and validate backend settings.

    Args:
        environment: Optional mapping used instead of ``os.environ``. Supplying a
            mapping makes configuration behavior deterministic in tests.

    Returns:
        An immutable :class:`BackendSettings` instance.

    Raises:
        BackendUtilityError: If an environment value is invalid.
    """
    env = os.environ if environment is None else environment
    host = env.get("HHS_API_HOST", "127.0.0.1").strip()
    if not host:
        raise BackendUtilityError("HHS_API_HOST must not be empty.")

    port = _environment_int(env, "HHS_API_PORT", 8000, 1, 65_535)
    max_request_bytes = _environment_int(
        env,
        "HHS_API_MAX_BODY_BYTES",
        DEFAULT_MAX_REQUEST_BYTES,
        1,
        2_147_483_647,
    )
    max_dataframe_rows = _environment_int(
        env,
        "HHS_API_MAX_DATAFRAME_ROWS",
        DEFAULT_MAX_DATAFRAME_ROWS,
        1,
        100_000_000,
    )

    log_level = env.get("HHS_API_LOG_LEVEL", "INFO").strip().upper()
    if log_level not in VALID_LOG_LEVELS:
        raise BackendUtilityError(
            "HHS_API_LOG_LEVEL must be one of: "
            + ", ".join(sorted(VALID_LOG_LEVELS))
            + "."
        )

    runtime_environment = env.get("HHS_API_ENVIRONMENT", "development").strip()
    if not runtime_environment:
        raise BackendUtilityError("HHS_API_ENVIRONMENT must not be empty.")

    cors_value = env.get("HHS_API_CORS_ORIGINS")
    if cors_value is None:
        cors_origins = DEFAULT_CORS_ORIGINS
    else:
        cors_origins = tuple(
            origin.strip() for origin in cors_value.split(",") if origin.strip()
        )
        if not cors_origins:
            raise BackendUtilityError(
                "HHS_API_CORS_ORIGINS must contain at least one origin."
            )

    return BackendSettings(
        host=host,
        port=port,
        log_level=log_level,
        environment=runtime_environment,
        cors_origins=cors_origins,
        max_request_bytes=max_request_bytes,
        max_dataframe_rows=max_dataframe_rows,
    )


@lru_cache(maxsize=1)
def get_backend_settings() -> BackendSettings:
    """Return process-wide cached backend settings."""
    return load_backend_settings()


def reset_backend_settings_cache() -> None:
    """Clear cached settings, primarily for tests and controlled reloads."""
    get_backend_settings.cache_clear()


class _RequestIDFilter(logging.Filter):
    """Ensure formatters always receive a request_id field."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = "system"
        return True


def configure_logging(
    logger_name: str = "hhs_uac",
    level: str | None = None,
) -> logging.Logger:
    """Configure and return a consistently formatted application logger."""
    selected_level = (level or get_backend_settings().log_level).strip().upper()
    if selected_level not in VALID_LOG_LEVELS:
        raise BackendUtilityError(
            "Logging level must be one of: " + ", ".join(sorted(VALID_LOG_LEVELS))
        )

    logger = logging.getLogger(logger_name)
    logger.setLevel(selected_level)
    if not any(getattr(handler, "_hhs_uac_handler", False) for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler.setLevel(selected_level)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s "
                "request_id=%(request_id)s %(message)s"
            )
        )
        handler.addFilter(_RequestIDFilter())
        handler._hhs_uac_handler = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    return logger


class RequestIDAdapter(logging.LoggerAdapter):
    """Logging adapter that safely injects a request ID into every record."""

    def process(
        self,
        msg: object,
        kwargs: dict[str, Any],
    ) -> tuple[object, dict[str, Any]]:
        extra = dict(kwargs.get("extra") or {})
        extra.setdefault("request_id", self.extra.get("request_id", "system"))
        kwargs["extra"] = extra
        return msg, kwargs


def logger_with_request_id(
    logger: logging.Logger,
    request_id: str | None,
) -> RequestIDAdapter:
    """Return a logger adapter associated with a request or job ID."""
    normalized = str(request_id or "system").strip()[:128] or "system"
    return RequestIDAdapter(logger, {"request_id": normalized})


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    """Recursively convert Python, pandas, and numpy values to strict JSON.

    Non-finite numbers and pandas missing values become ``None``. Date-like
    values are represented in ISO-8601 format.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Decimal):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return json_safe(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def dataframe_records(
    frame: pd.DataFrame,
    *,
    include_index: bool | None = None,
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    """Convert a DataFrame into strict JSON-safe row records.

    Args:
        frame: Source DataFrame.
        include_index: ``None`` includes meaningful indexes but omits an unnamed
            RangeIndex. ``True`` always attempts to include the index.
        max_rows: Optional hard row limit. Exceeding it raises an error rather
            than silently truncating the response.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    if max_rows is not None:
        if max_rows < 0:
            raise BackendUtilityError("max_rows must be non-negative.")
        if len(frame) > max_rows:
            raise BackendUtilityError(
                f"DataFrame contains {len(frame):,} rows; limit is {max_rows:,}."
            )
    if frame.empty:
        return []

    serializable = frame.copy()
    should_include_index = include_index
    if should_include_index is None:
        should_include_index = not (
            isinstance(serializable.index, pd.RangeIndex)
            and serializable.index.name is None
        )
    if should_include_index:
        index_name = serializable.index.name or "index"
        if index_name in serializable.columns:
            serializable = serializable.reset_index(drop=True)
        else:
            serializable = serializable.reset_index()

    serializable.columns = [str(column) for column in serializable.columns]
    serializable.attrs.clear()
    return json_safe(serializable.to_dict(orient="records"))


def dataframe_to_csv_bytes(
    frame: pd.DataFrame,
    *,
    include_index: bool = False,
) -> bytes:
    """Serialize a DataFrame as UTF-8 CSV bytes."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    buffer = StringIO()
    frame.to_csv(
        buffer,
        index=include_index,
        date_format="%Y-%m-%d",
        lineterminator="\n",
    )
    return buffer.getvalue().encode("utf-8")


def read_csv_bytes(
    data: bytes,
    *,
    max_bytes: int | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Read CSV bytes with size, encoding, parser, and row-count validation."""
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes.")
    if not data.strip():
        raise BackendUtilityError("CSV data is empty.")

    settings = get_backend_settings()
    byte_limit = max_bytes if max_bytes is not None else settings.max_request_bytes
    row_limit = max_rows if max_rows is not None else settings.max_dataframe_rows
    if byte_limit <= 0 or row_limit <= 0:
        raise BackendUtilityError("CSV byte and row limits must be positive.")
    if len(data) > byte_limit:
        raise BackendUtilityError(
            f"CSV contains {len(data):,} bytes; limit is {byte_limit:,}."
        )
    if b"\x00" in data:
        raise BackendUtilityError("CSV contains unsupported null bytes.")

    try:
        frame = pd.read_csv(
            BytesIO(data),
            dtype=str,
            encoding="utf-8-sig",
        )
    except (
        UnicodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as exc:
        raise BackendUtilityError(f"CSV could not be parsed: {exc}") from exc
    if len(frame) > row_limit:
        raise BackendUtilityError(
            f"CSV contains {len(frame):,} rows; limit is {row_limit:,}."
        )
    return frame


def paginate_dataframe(
    frame: pd.DataFrame,
    *,
    offset: int = 0,
    limit: int = 100,
    maximum_limit: int = 5_000,
) -> tuple[pd.DataFrame, PaginationMetadata]:
    """Return a defensive DataFrame page and pagination metadata."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    if offset < 0:
        raise BackendUtilityError("offset must be non-negative.")
    if maximum_limit < 1:
        raise BackendUtilityError("maximum_limit must be positive.")
    if not 1 <= limit <= maximum_limit:
        raise BackendUtilityError(
            f"limit must be between 1 and {maximum_limit:,}."
        )

    total = len(frame)
    page = frame.iloc[offset : offset + limit].copy()
    metadata = PaginationMetadata(
        total_records=total,
        offset=offset,
        limit=limit,
        returned_records=len(page),
        has_previous=offset > 0 and total > 0,
        has_next=offset + len(page) < total,
    )
    return page, metadata


def dataframe_fingerprint(frame: pd.DataFrame) -> str:
    """Return a stable SHA-256 fingerprint of values, index, schema, and order."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")

    digest = hashlib.sha256()
    schema = {
        "columns": [str(column) for column in frame.columns],
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "index_name": str(frame.index.name),
        "index_dtype": str(frame.index.dtype),
        "shape": frame.shape,
    }
    digest.update(json.dumps(schema, sort_keys=True).encode("utf-8"))
    try:
        hashes = pd.util.hash_pandas_object(frame, index=True, categorize=True)
        digest.update(hashes.to_numpy(dtype="uint64").tobytes())
    except (TypeError, ValueError):
        # Fallback for unusual object columns containing unhashable values.
        records = dataframe_records(frame, include_index=True)
        digest.update(
            json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
    return digest.hexdigest()


def dataset_metadata(
    frame: pd.DataFrame,
    *,
    date_column: str = "Date",
) -> dict[str, Any]:
    """Build JSON-friendly audit metadata for a DataFrame."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")

    if date_column in frame.columns:
        dates = pd.to_datetime(frame[date_column], errors="coerce")
    elif frame.index.name == date_column or isinstance(frame.index, pd.DatetimeIndex):
        dates = pd.Series(pd.to_datetime(frame.index, errors="coerce"))
    else:
        dates = pd.Series(dtype="datetime64[ns]")
    valid_dates = dates.dropna()

    return {
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "columns": [str(column) for column in frame.columns],
        "date_min": valid_dates.min().date().isoformat() if not valid_dates.empty else None,
        "date_max": valid_dates.max().date().isoformat() if not valid_dates.empty else None,
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_index_values": int(frame.index.duplicated().sum()),
        "memory_bytes": int(frame.memory_usage(index=True, deep=True).sum()),
        "fingerprint_sha256": dataframe_fingerprint(frame),
        "generated_at_utc": utc_now_iso(),
    }


def require_columns(
    frame: pd.DataFrame,
    required_columns: Sequence[str],
) -> None:
    """Raise a clear error when required DataFrame columns are missing."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    required = [str(column) for column in required_columns]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise BackendUtilityError(
            "Missing required column(s): " + ", ".join(missing)
        )
