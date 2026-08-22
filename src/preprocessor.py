"""Configurable preprocessing for HHS UAC capacity source data.

The preprocessor accepts DataFrames, CSV paths, bytes, and file-like objects. It
normalizes the official footnoted schema, produces a complete daily time series,
and records every repair or anomaly in a structured report and row-level flags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from io import BytesIO, StringIO
from pathlib import Path
from typing import BinaryIO, TextIO

import numpy as np
import pandas as pd

from app_utils import (
    CBP_COLUMN,
    DATE_COLUMN,
    DISCHARGE_ANOMALY_COLUMN,
    DISCHARGE_COLUMN,
    FLOW_COLUMNS,
    HHS_COLUMN,
    NUMERIC_COLUMNS,
    QUALITY_FLAG_COLUMN,
    REQUIRED_COLUMNS,
    STOCK_COLUMNS,
    TRANSFER_ANOMALY_COLUMN,
    TRANSFER_COLUMN,
)


IS_IMPUTED_DATE_COLUMN = "Is Imputed Date"
HAS_IMPUTED_VALUES_COLUMN = "Has Imputed Values"
IMPUTED_VALUE_COUNT_COLUMN = "Imputed Value Count"
NEGATIVE_COUNT_ANOMALY_COLUMN = "Anomaly_Negative_Count"
ANY_ANOMALY_COLUMN = "Anomaly_Any"


class PreprocessingError(ValueError):
    """Raised when source data cannot be preprocessed safely."""


class IssueSeverity(str, Enum):
    """Severity assigned to a preprocessing finding."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class PreprocessingIssue:
    """One structured source-data or repair finding."""

    severity: IssueSeverity
    code: str
    message: str
    affected_rows: int = 0
    column: str | None = None

    def to_dict(self) -> dict[str, str | int | None]:
        """Return a JSON-friendly issue dictionary."""
        return {
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
            "affected_rows": self.affected_rows,
            "column": self.column,
        }


@dataclass
class PreprocessingReport:
    """Audit report for one preprocessing run."""

    source_name: str
    source_rows: int
    output_rows: int = 0
    source_columns: int = 0
    output_columns: int = 0
    reporting_start: str | None = None
    reporting_end: str | None = None
    empty_rows_removed: int = 0
    invalid_date_rows_removed: int = 0
    duplicate_rows_removed: int = 0
    missing_dates_inserted: int = 0
    numeric_values_imputed: int = 0
    negative_values_found: int = 0
    logical_anomaly_rows: int = 0
    issues: list[PreprocessingIssue] = field(default_factory=list)

    def add(
        self,
        severity: IssueSeverity,
        code: str,
        message: str,
        *,
        affected_rows: int = 0,
        column: str | None = None,
    ) -> None:
        """Append one issue to the report."""
        self.issues.append(
            PreprocessingIssue(
                severity=severity,
                code=code,
                message=message,
                affected_rows=int(affected_rows),
                column=column,
            )
        )

    @property
    def error_count(self) -> int:
        return sum(issue.severity is IssueSeverity.ERROR for issue in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(issue.severity is IssueSeverity.WARNING for issue in self.issues)

    @property
    def has_errors(self) -> bool:
        return self.error_count > 0

    def to_frame(self) -> pd.DataFrame:
        """Return issue details as a presentation-ready DataFrame."""
        if not self.issues:
            return pd.DataFrame(
                [
                    {
                        "Severity": "info",
                        "Code": "OK",
                        "Message": "No preprocessing issues found.",
                        "Affected Rows": 0,
                        "Column": None,
                    }
                ]
            )
        return pd.DataFrame(
            [
                {
                    "Severity": issue.severity.value,
                    "Code": issue.code,
                    "Message": issue.message,
                    "Affected Rows": issue.affected_rows,
                    "Column": issue.column,
                }
                for issue in self.issues
            ]
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly report summary and issue list."""
        return {
            "source_name": self.source_name,
            "source_rows": self.source_rows,
            "output_rows": self.output_rows,
            "source_columns": self.source_columns,
            "output_columns": self.output_columns,
            "reporting_start": self.reporting_start,
            "reporting_end": self.reporting_end,
            "empty_rows_removed": self.empty_rows_removed,
            "invalid_date_rows_removed": self.invalid_date_rows_removed,
            "duplicate_rows_removed": self.duplicate_rows_removed,
            "missing_dates_inserted": self.missing_dates_inserted,
            "numeric_values_imputed": self.numeric_values_imputed,
            "negative_values_found": self.negative_values_found,
            "logical_anomaly_rows": self.logical_anomaly_rows,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class PreprocessingConfig:
    """Controls repair policies for the preprocessing pipeline."""

    fill_missing_dates: bool = True
    duplicate_policy: str = "last"
    stock_imputation: str = "interpolate"
    flow_imputation: str = "zero"
    round_fractional_counts: bool = True
    clip_negative_counts: bool = False
    strict: bool = False
    strict_logical_constraints: bool = False
    maximum_rows: int = 1_000_000
    maximum_csv_bytes: int = 50 * 1024 * 1024

    def __post_init__(self) -> None:
        if self.duplicate_policy not in {"first", "last", "error"}:
            raise ValueError("duplicate_policy must be 'first', 'last', or 'error'.")
        if self.stock_imputation not in {
            "interpolate",
            "forward_fill",
            "zero",
            "none",
        }:
            raise ValueError(
                "stock_imputation must be interpolate, forward_fill, zero, or none."
            )
        if self.flow_imputation not in {"zero", "interpolate", "forward_fill", "none"}:
            raise ValueError(
                "flow_imputation must be zero, interpolate, forward_fill, or none."
            )
        if self.maximum_rows < 1:
            raise ValueError("maximum_rows must be positive.")
        if self.maximum_csv_bytes < 1:
            raise ValueError("maximum_csv_bytes must be positive.")


@dataclass
class PreprocessedDataset:
    """Cleaned daily dataset and its full preprocessing audit report."""

    data: pd.DataFrame
    report: PreprocessingReport
    config: PreprocessingConfig

    def copy(self) -> PreprocessedDataset:
        """Return a defensive copy for downstream transformation."""
        return PreprocessedDataset(
            data=self.data.copy(),
            report=self.report,
            config=self.config,
        )


DataSource = pd.DataFrame | str | Path | bytes | BinaryIO | TextIO


def _canonical_column_name(name: object) -> str:
    """Normalize source header whitespace, BOMs, and footnote markers."""
    normalized = " ".join(str(name).replace("\ufeff", "").strip().split())
    return normalized.rstrip("*").strip()


def _parse_dates(values: pd.Series) -> pd.Series:
    """Parse mixed ISO and human-readable reporting dates."""
    try:
        return pd.to_datetime(values, format="mixed", errors="coerce")
    except (TypeError, ValueError):
        return values.apply(lambda value: pd.to_datetime(value, errors="coerce"))


def _read_source(
    source: DataSource,
    config: PreprocessingConfig,
) -> tuple[pd.DataFrame, str]:
    """Read a supported source into a string-first DataFrame."""
    if isinstance(source, pd.DataFrame):
        if len(source) > config.maximum_rows:
            raise PreprocessingError(
                f"Source contains {len(source):,} rows; limit is {config.maximum_rows:,}."
            )
        return source.copy(), "dataframe"

    source_name = "uploaded.csv"
    read_target: str | Path | BinaryIO | TextIO
    if isinstance(source, bytes):
        if len(source) > config.maximum_csv_bytes:
            raise PreprocessingError(
                f"CSV contains {len(source):,} bytes; limit is "
                f"{config.maximum_csv_bytes:,}."
            )
        if not source.strip():
            raise PreprocessingError("CSV source is empty.")
        if b"\x00" in source:
            raise PreprocessingError("CSV source contains unsupported null bytes.")
        read_target = BytesIO(source)
    else:
        read_target = source
        if isinstance(source, (str, Path)):
            path = Path(source).expanduser()
            source_name = path.name
            try:
                size = path.stat().st_size
            except OSError as exc:
                raise PreprocessingError(f"Unable to access source '{path}': {exc}") from exc
            if size > config.maximum_csv_bytes:
                raise PreprocessingError(
                    f"CSV contains {size:,} bytes; limit is {config.maximum_csv_bytes:,}."
                )

    try:
        frame = pd.read_csv(
            read_target,
            dtype=str,
            encoding="utf-8-sig",
        )
    except (
        OSError,
        UnicodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as exc:
        raise PreprocessingError(f"Unable to read CSV source: {exc}") from exc
    if len(frame) > config.maximum_rows:
        raise PreprocessingError(
            f"Source contains {len(frame):,} rows; limit is {config.maximum_rows:,}."
        )
    return frame, source_name


def _normalize_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize and validate the six required source columns."""
    canonical_to_original: dict[str, object] = {}
    duplicate_canonical: list[str] = []
    for original in frame.columns:
        canonical = _canonical_column_name(original)
        if canonical in canonical_to_original:
            duplicate_canonical.append(canonical)
        canonical_to_original[canonical] = original
    if duplicate_canonical:
        raise PreprocessingError(
            "Multiple columns normalize to the same name: "
            + ", ".join(sorted(set(duplicate_canonical)))
        )

    missing = [column for column in REQUIRED_COLUMNS if column not in canonical_to_original]
    if missing:
        raise PreprocessingError("Missing required column(s): " + ", ".join(missing))

    renamed = frame.rename(
        columns={
            original: canonical
            for canonical, original in canonical_to_original.items()
        }
    )
    return renamed[REQUIRED_COLUMNS].copy()


def _numeric_series(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Convert formatted numeric counts and return invalid-value flags."""
    text = values.astype("string").str.strip()
    text = text.mask(text.eq(""), pd.NA)
    text = text.str.replace(",", "", regex=False)
    text = text.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    converted = pd.to_numeric(text, errors="coerce")
    invalid = text.notna() & converted.isna()
    return converted, invalid


def _impute_columns(
    frame: pd.DataFrame,
    columns: list[str],
    policy: str,
) -> pd.DataFrame:
    """Apply one configured missing-value policy to numeric columns."""
    result = frame.copy()
    if policy == "interpolate":
        result[columns] = result[columns].interpolate(
            method="time",
            limit_direction="both",
        )
    elif policy == "forward_fill":
        result[columns] = result[columns].ffill().bfill()
    elif policy == "zero":
        result[columns] = result[columns].fillna(0)
    elif policy != "none":
        raise PreprocessingError(f"Unsupported imputation policy: {policy}")
    return result


class HHSDataPreprocessor:
    """Configurable source-to-daily-data preprocessing pipeline."""

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        self.config = config or PreprocessingConfig()

    def transform(self, source: DataSource) -> PreprocessedDataset:
        """Read, validate, repair, flag, and report an HHS-style dataset."""
        raw_frame, source_name = _read_source(source, self.config)
        if raw_frame.empty:
            raise PreprocessingError("Source data is empty.")

        report = PreprocessingReport(
            source_name=source_name,
            source_rows=len(raw_frame),
            source_columns=len(raw_frame.columns),
        )
        frame = _normalize_schema(raw_frame)

        empty_mask = frame.apply(
            lambda column: column.isna() | column.astype("string").str.strip().eq("")
        ).all(axis=1)
        if empty_mask.any():
            count = int(empty_mask.sum())
            report.empty_rows_removed = count
            report.add(
                IssueSeverity.WARNING,
                "EMPTY_ROWS_REMOVED",
                "Completely empty export-padding rows were removed.",
                affected_rows=count,
            )
            frame = frame.loc[~empty_mask].copy()
        if frame.empty:
            raise PreprocessingError("Source contains no populated rows.")

        frame[DATE_COLUMN] = _parse_dates(frame[DATE_COLUMN])
        invalid_dates = frame[DATE_COLUMN].isna()
        if invalid_dates.any():
            count = int(invalid_dates.sum())
            report.invalid_date_rows_removed = count
            report.add(
                IssueSeverity.ERROR,
                "INVALID_DATES_REMOVED",
                "Rows with invalid reporting dates were removed.",
                affected_rows=count,
                column=DATE_COLUMN,
            )
            frame = frame.loc[~invalid_dates].copy()
        if frame.empty:
            raise PreprocessingError("No valid reporting dates remain after parsing.")

        if not frame[DATE_COLUMN].is_monotonic_increasing:
            report.add(
                IssueSeverity.WARNING,
                "CHRONOLOGY_SORTED",
                "Rows were sorted into chronological order.",
            )

        for column in NUMERIC_COLUMNS:
            converted, invalid = _numeric_series(frame[column])
            if invalid.any():
                count = int(invalid.sum())
                report.add(
                    IssueSeverity.ERROR,
                    "INVALID_NUMERIC_VALUES",
                    "Invalid numeric values were converted to missing values.",
                    affected_rows=count,
                    column=column,
                )
            frame[column] = converted

        frame = frame.sort_values(DATE_COLUMN, kind="stable")
        duplicate_mask = frame.duplicated(DATE_COLUMN, keep=False)
        duplicate_count = int(duplicate_mask.sum())
        if duplicate_count:
            if self.config.duplicate_policy == "error":
                raise PreprocessingError(
                    f"Duplicate reporting dates affect {duplicate_count} rows."
                )
            keep = self.config.duplicate_policy
            removed = int(frame.duplicated(DATE_COLUMN, keep=keep).sum())
            report.duplicate_rows_removed = removed
            report.add(
                IssueSeverity.ERROR,
                "DUPLICATE_DATES_RESOLVED",
                f"Duplicate dates were resolved by retaining the {keep} row.",
                affected_rows=removed,
                column=DATE_COLUMN,
            )
            frame = frame.loc[~frame.duplicated(DATE_COLUMN, keep=keep)].copy()

        frame = frame.set_index(DATE_COLUMN)
        frame.index = pd.DatetimeIndex(frame.index, name=DATE_COLUMN)
        if self.config.fill_missing_dates:
            complete_index = pd.date_range(
                frame.index.min(),
                frame.index.max(),
                freq="D",
                name=DATE_COLUMN,
            )
            inserted_dates = ~complete_index.isin(frame.index)
            frame = frame.reindex(complete_index)
        else:
            inserted_dates = np.zeros(len(frame), dtype=bool)
        frame[IS_IMPUTED_DATE_COLUMN] = inserted_dates
        inserted_count = int(np.asarray(inserted_dates).sum())
        if inserted_count:
            report.missing_dates_inserted = inserted_count
            report.add(
                IssueSeverity.WARNING,
                "MISSING_DATES_INSERTED",
                "Missing calendar dates were inserted into the daily series.",
                affected_rows=inserted_count,
            )

        missing_before = frame[NUMERIC_COLUMNS].isna()
        frame = _impute_columns(frame, STOCK_COLUMNS, self.config.stock_imputation)
        frame = _impute_columns(frame, FLOW_COLUMNS, self.config.flow_imputation)
        missing_after = frame[NUMERIC_COLUMNS].isna()
        imputed_cells = missing_before & ~missing_after
        frame[IMPUTED_VALUE_COUNT_COLUMN] = imputed_cells.sum(axis=1).astype("int16")
        frame[HAS_IMPUTED_VALUES_COLUMN] = frame[IMPUTED_VALUE_COUNT_COLUMN].gt(0)
        report.numeric_values_imputed = int(imputed_cells.sum().sum())
        if report.numeric_values_imputed:
            report.add(
                IssueSeverity.WARNING,
                "NUMERIC_VALUES_IMPUTED",
                "Missing numeric values were filled using configured policies.",
                affected_rows=int(frame[HAS_IMPUTED_VALUES_COLUMN].sum()),
            )

        unresolved_missing = int(frame[NUMERIC_COLUMNS].isna().sum().sum())
        if unresolved_missing:
            report.add(
                IssueSeverity.ERROR,
                "UNRESOLVED_MISSING_VALUES",
                "Numeric missing values remain because an imputation policy is disabled.",
                affected_rows=unresolved_missing,
            )

        fractional_mask = (
            frame[NUMERIC_COLUMNS].notna()
            & frame[NUMERIC_COLUMNS].mod(1).abs().gt(1e-9)
        )
        fractional_count = int(fractional_mask.sum().sum())
        if fractional_count:
            severity = (
                IssueSeverity.WARNING
                if self.config.round_fractional_counts
                else IssueSeverity.ERROR
            )
            report.add(
                severity,
                "FRACTIONAL_COUNTS",
                (
                    "Fractional child counts were rounded to integers."
                    if self.config.round_fractional_counts
                    else "Fractional child counts remain in the output."
                ),
                affected_rows=fractional_count,
            )
            if self.config.round_fractional_counts:
                frame[NUMERIC_COLUMNS] = frame[NUMERIC_COLUMNS].round()

        negative_cells = frame[NUMERIC_COLUMNS].lt(0)
        negative_rows = negative_cells.any(axis=1)
        negative_count = int(negative_cells.sum().sum())
        report.negative_values_found = negative_count
        if negative_count:
            report.add(
                (
                    IssueSeverity.WARNING
                    if self.config.clip_negative_counts
                    else IssueSeverity.ERROR
                ),
                "NEGATIVE_COUNTS",
                (
                    "Negative counts were clipped to zero."
                    if self.config.clip_negative_counts
                    else "Negative counts remain flagged in the output."
                ),
                affected_rows=int(negative_rows.sum()),
            )
            if self.config.clip_negative_counts:
                frame[NUMERIC_COLUMNS] = frame[NUMERIC_COLUMNS].clip(lower=0)

        # Use nullable integers whenever all repaired count values are integral.
        if self.config.round_fractional_counts:
            for column in NUMERIC_COLUMNS:
                frame[column] = frame[column].astype("Int64")

        frame[TRANSFER_ANOMALY_COLUMN] = (
            frame[TRANSFER_COLUMN] > frame[CBP_COLUMN]
        ).fillna(False)
        frame[DISCHARGE_ANOMALY_COLUMN] = (
            frame[DISCHARGE_COLUMN] > frame[HHS_COLUMN]
        ).fillna(False)
        frame[NEGATIVE_COUNT_ANOMALY_COLUMN] = negative_rows.fillna(False)
        frame[ANY_ANOMALY_COLUMN] = frame[
            [
                TRANSFER_ANOMALY_COLUMN,
                DISCHARGE_ANOMALY_COLUMN,
                NEGATIVE_COUNT_ANOMALY_COLUMN,
            ]
        ].any(axis=1)

        transfer_anomalies = int(frame[TRANSFER_ANOMALY_COLUMN].sum())
        discharge_anomalies = int(frame[DISCHARGE_ANOMALY_COLUMN].sum())
        report.logical_anomaly_rows = int(
            frame[
                [TRANSFER_ANOMALY_COLUMN, DISCHARGE_ANOMALY_COLUMN]
            ].any(axis=1).sum()
        )
        if transfer_anomalies:
            report.add(
                IssueSeverity.ERROR,
                "TRANSFER_EXCEEDS_CBP",
                "Transfers exceed active CBP custody.",
                affected_rows=transfer_anomalies,
                column=TRANSFER_COLUMN,
            )
        if discharge_anomalies:
            report.add(
                IssueSeverity.ERROR,
                "DISCHARGE_EXCEEDS_HHS",
                "Discharges exceed active HHS care.",
                affected_rows=discharge_anomalies,
                column=DISCHARGE_COLUMN,
            )

        def quality_label(row: pd.Series) -> str:
            labels: list[str] = []
            if bool(row[IS_IMPUTED_DATE_COLUMN]):
                labels.append("Imputed date")
            if bool(row[HAS_IMPUTED_VALUES_COLUMN]):
                labels.append("Imputed values")
            if bool(row[TRANSFER_ANOMALY_COLUMN]):
                labels.append("Transfer > CBP custody")
            if bool(row[DISCHARGE_ANOMALY_COLUMN]):
                labels.append("Discharge > HHS care")
            if bool(row[NEGATIVE_COUNT_ANOMALY_COLUMN]):
                labels.append("Negative count")
            return "; ".join(labels) if labels else "OK"

        frame[QUALITY_FLAG_COLUMN] = frame.apply(quality_label, axis=1)
        report.output_rows = len(frame)
        report.output_columns = len(frame.columns)
        report.reporting_start = frame.index.min().date().isoformat()
        report.reporting_end = frame.index.max().date().isoformat()

        strict_errors = report.has_errors
        if self.config.strict and strict_errors:
            raise PreprocessingError(
                f"Strict preprocessing rejected {report.error_count} error type(s)."
            )
        if self.config.strict_logical_constraints and report.logical_anomaly_rows:
            raise PreprocessingError(
                "Strict logical-constraint validation rejected "
                f"{report.logical_anomaly_rows} anomalous row(s)."
            )

        frame.attrs = {
            "preprocessing_report": report,
            "preprocessing_config": self.config,
        }
        return PreprocessedDataset(data=frame, report=report, config=self.config)


def preprocess_data(
    source: DataSource,
    config: PreprocessingConfig | None = None,
) -> PreprocessedDataset:
    """Functional entry point for the complete preprocessing pipeline."""
    return HHSDataPreprocessor(config).transform(source)


def validate_preprocessed_data(frame: pd.DataFrame) -> None:
    """Validate core invariants expected by analytics and feature engineering."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    if frame.empty:
        raise PreprocessingError("Preprocessed data is empty.")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise PreprocessingError("Preprocessed data must use a DatetimeIndex.")
    if frame.index.name != DATE_COLUMN:
        raise PreprocessingError(f"Preprocessed index must be named '{DATE_COLUMN}'.")
    if not frame.index.is_monotonic_increasing:
        raise PreprocessingError("Preprocessed dates are not chronologically sorted.")
    if frame.index.has_duplicates:
        raise PreprocessingError("Preprocessed data contains duplicate dates.")
    missing = [column for column in NUMERIC_COLUMNS if column not in frame.columns]
    if missing:
        raise PreprocessingError(
            "Preprocessed data is missing count column(s): " + ", ".join(missing)
        )
    non_numeric = [
        column
        for column in NUMERIC_COLUMNS
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        raise PreprocessingError(
            "Preprocessed count columns must be numeric: " + ", ".join(non_numeric)
        )


def preprocessed_to_csv_bytes(
    dataset: PreprocessedDataset | pd.DataFrame,
) -> bytes:
    """Serialize cleaned daily data as a UTF-8 CSV with an explicit Date column."""
    frame = dataset.data if isinstance(dataset, PreprocessedDataset) else dataset
    validate_preprocessed_data(frame)
    export_frame = frame.reset_index().copy()
    export_frame.attrs.clear()
    buffer = StringIO()
    export_frame.to_csv(
        buffer,
        index=False,
        date_format="%Y-%m-%d",
        lineterminator="\n",
    )
    return buffer.getvalue().encode("utf-8")
