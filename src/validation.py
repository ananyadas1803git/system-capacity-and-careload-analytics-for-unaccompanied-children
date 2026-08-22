"""Non-mutating data-quality validation for HHS UAC capacity datasets.

Unlike :mod:`src.preprocessor`, this module never repairs source data.  It audits
raw, cleaned, or metric-enriched frames and returns machine-readable findings,
row-level issue codes, and a concise quality score suitable for APIs and reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from app_utils import (
    BACKLOG_STREAK_COLUMN,
    CBP_COLUMN,
    DATE_COLUMN,
    DISCHARGE_ANOMALY_COLUMN,
    DISCHARGE_COLUMN,
    GROWTH_RATE_COLUMN,
    HHS_COLUMN,
    INTAKE_COLUMN,
    NET_INTAKE_COLUMN,
    NUMERIC_COLUMNS,
    OFFSET_RATIO_COLUMN,
    REQUIRED_COLUMNS,
    ROLLING_14_COLUMN,
    ROLLING_7_COLUMN,
    TOTAL_LOAD_COLUMN,
    TRANSFER_ANOMALY_COLUMN,
    TRANSFER_COLUMN,
)


ROW_NUMBER_COLUMN = "Source Row"
ISSUE_CODES_COLUMN = "Validation Issue Codes"


class DatasetValidationError(ValueError):
    """Raised when validation cannot run or strict validation fails."""


class ValidationSeverity(str, Enum):
    """Supported validation severity levels, ordered from least to most severe."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


_SEVERITY_RANK = {
    ValidationSeverity.INFO: 0,
    ValidationSeverity.WARNING: 1,
    ValidationSeverity.ERROR: 2,
    ValidationSeverity.CRITICAL: 3,
}


@dataclass(frozen=True)
class ValidationConfig:
    """Configuration for dataset audit rules.

    ``check_stock_flow_balance`` is disabled by default because published stock
    and flow fields may use different daily cut-off times.  When enabled, large
    accounting residuals are reported as warnings rather than hard errors.
    """

    require_complete_daily_series: bool = True
    allow_fractional_counts: bool = False
    check_derived_metrics: bool = True
    check_stock_flow_balance: bool = False
    stock_flow_tolerance: float = 1.0
    metric_absolute_tolerance: float = 1e-6
    metric_relative_tolerance: float = 1e-6
    outlier_robust_z_threshold: float | None = 8.0
    strict_warnings: bool = False
    maximum_rows: int = 1_000_000
    maximum_flagged_rows: int = 10_000

    def __post_init__(self) -> None:
        if self.stock_flow_tolerance < 0:
            raise ValueError("stock_flow_tolerance must be non-negative.")
        if self.metric_absolute_tolerance < 0:
            raise ValueError("metric_absolute_tolerance must be non-negative.")
        if self.metric_relative_tolerance < 0:
            raise ValueError("metric_relative_tolerance must be non-negative.")
        if self.outlier_robust_z_threshold is not None and self.outlier_robust_z_threshold <= 0:
            raise ValueError("outlier_robust_z_threshold must be positive or None.")
        if self.maximum_rows < 1 or self.maximum_flagged_rows < 1:
            raise ValueError("Row limits must be positive integers.")


@dataclass(frozen=True)
class ValidationFinding:
    """One aggregate validation finding."""

    severity: ValidationSeverity
    category: str
    code: str
    message: str
    affected_rows: int = 0
    columns: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "severity": self.severity.value,
            "category": self.category,
            "code": self.code,
            "message": self.message,
            "affected_rows": self.affected_rows,
            "columns": list(self.columns),
            "examples": list(self.examples),
        }


@dataclass
class DatasetValidationReport:
    """Summary and evidence produced by one validation run."""

    source_rows: int
    source_columns: int
    findings: list[ValidationFinding] = field(default_factory=list)
    valid_date_rows: int = 0
    reporting_start: str | None = None
    reporting_end: str | None = None
    expected_daily_rows: int = 0
    missing_calendar_dates: int = 0
    flagged_row_count: int = 0
    flagged_rows_truncated: bool = False

    def add(
        self,
        severity: ValidationSeverity,
        category: str,
        code: str,
        message: str,
        *,
        affected_rows: int = 0,
        columns: tuple[str, ...] = (),
        examples: tuple[str, ...] = (),
    ) -> None:
        """Append a structured finding."""
        self.findings.append(
            ValidationFinding(
                severity=severity,
                category=category,
                code=code,
                message=message,
                affected_rows=int(affected_rows),
                columns=columns,
                examples=examples,
            )
        )

    @property
    def error_count(self) -> int:
        """Return the number of error or critical finding types."""
        return sum(
            _SEVERITY_RANK[item.severity] >= _SEVERITY_RANK[ValidationSeverity.ERROR]
            for item in self.findings
        )

    @property
    def warning_count(self) -> int:
        """Return the number of warning finding types."""
        return sum(item.severity is ValidationSeverity.WARNING for item in self.findings)

    @property
    def is_valid(self) -> bool:
        """Whether no error or critical finding was detected."""
        return self.error_count == 0

    @property
    def status(self) -> str:
        """Return a presentation-ready validation status."""
        if not self.is_valid:
            return "Failed"
        if self.warning_count:
            return "Passed with warnings"
        return "Passed"

    @property
    def quality_score(self) -> int:
        """Return a transparent 0-100 heuristic quality score.

        The score complements, but never overrides, individual findings.  Each
        distinct critical, error, and warning type deducts 35, 12, and 4 points.
        """
        deductions = {
            ValidationSeverity.INFO: 0,
            ValidationSeverity.WARNING: 4,
            ValidationSeverity.ERROR: 12,
            ValidationSeverity.CRITICAL: 35,
        }
        return max(0, 100 - sum(deductions[item.severity] for item in self.findings))

    def to_frame(self) -> pd.DataFrame:
        """Return findings as a display-ready table."""
        columns = [
            "Severity",
            "Category",
            "Code",
            "Message",
            "Affected Rows",
            "Columns",
            "Examples",
        ]
        if not self.findings:
            return pd.DataFrame(
                [["info", "summary", "OK", "No issues found.", 0, "", ""]],
                columns=columns,
            )
        records = [
            {
                "Severity": item.severity.value,
                "Category": item.category,
                "Code": item.code,
                "Message": item.message,
                "Affected Rows": item.affected_rows,
                "Columns": ", ".join(item.columns),
                "Examples": ", ".join(item.examples),
            }
            for item in sorted(
                self.findings,
                key=lambda finding: (-_SEVERITY_RANK[finding.severity], finding.code),
            )
        ]
        return pd.DataFrame.from_records(records, columns=columns)

    def to_dict(self) -> dict[str, Any]:
        """Return the full report as JSON-compatible primitives."""
        return {
            "status": self.status,
            "is_valid": self.is_valid,
            "quality_score": self.quality_score,
            "source_rows": self.source_rows,
            "source_columns": self.source_columns,
            "valid_date_rows": self.valid_date_rows,
            "reporting_start": self.reporting_start,
            "reporting_end": self.reporting_end,
            "expected_daily_rows": self.expected_daily_rows,
            "missing_calendar_dates": self.missing_calendar_dates,
            "flagged_row_count": self.flagged_row_count,
            "flagged_rows_truncated": self.flagged_rows_truncated,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "findings": [item.to_dict() for item in self.findings],
        }


@dataclass
class DatasetValidationResult:
    """Validation report plus a bounded row-level evidence table."""

    report: DatasetValidationReport
    flagged_rows: pd.DataFrame

    def raise_for_errors(self, *, include_warnings: bool = False) -> None:
        """Raise when the result violates the requested strictness level."""
        failed = not self.report.is_valid or (include_warnings and self.report.warning_count > 0)
        if failed:
            raise DatasetValidationError(
                f"Dataset validation {self.report.status.lower()}: "
                f"{self.report.error_count} error type(s), "
                f"{self.report.warning_count} warning type(s)."
            )


def _canonical_column_name(value: object) -> str:
    """Normalize header whitespace, UTF-8 BOMs, and official footnote markers."""
    return " ".join(str(value).replace("\ufeff", "").strip().split()).rstrip("*").strip()


def _parse_dates(values: pd.Series) -> pd.Series:
    """Parse mixed-format timestamps without changing invalid values silently."""
    try:
        parsed = pd.to_datetime(values, format="mixed", errors="coerce", utc=True)
    except (TypeError, ValueError):
        parsed = values.apply(lambda value: pd.to_datetime(value, errors="coerce", utc=True))
    parsed = parsed.dt.tz_convert(None)
    return parsed.dt.normalize()


def _numeric_values(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Parse comma-formatted and parenthesized counts and flag invalid text."""
    text = values.astype("string").str.strip().mask(lambda item: item.eq(""), pd.NA)
    text = text.str.replace(",", "", regex=False)
    text = text.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    numeric = pd.to_numeric(text, errors="coerce")
    return numeric, text.notna() & numeric.isna()


def _boolean_values(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Parse common boolean encodings and return invalid-value flags."""
    text = values.astype("string").str.strip().str.casefold()
    mapped = text.map(
        {
            "true": True,
            "1": True,
            "yes": True,
            "y": True,
            "false": False,
            "0": False,
            "no": False,
            "n": False,
        }
    )
    invalid = text.notna() & ~text.eq("") & mapped.isna()
    return mapped.fillna(False).astype(bool), invalid


def _close_enough(
    actual: pd.Series,
    expected: pd.Series,
    config: ValidationConfig,
) -> pd.Series:
    """Return row-level equality under configured absolute/relative tolerances."""
    actual_values = pd.to_numeric(actual, errors="coerce").astype(float)
    expected_values = pd.to_numeric(expected, errors="coerce").astype(float)
    comparable = actual_values.notna() & expected_values.notna()
    both_missing = actual_values.isna() & expected_values.isna()
    close = np.isclose(
        actual_values.fillna(0),
        expected_values.fillna(0),
        atol=config.metric_absolute_tolerance,
        rtol=config.metric_relative_tolerance,
        equal_nan=False,
    )
    return both_missing | (comparable & pd.Series(close, index=actual.index))


def _expected_backlog_streak(net_intake: pd.Series) -> pd.Series:
    positive = pd.to_numeric(net_intake, errors="coerce").fillna(0).gt(0)
    groups = positive.ne(positive.shift(fill_value=False)).cumsum()
    return positive.groupby(groups).cumsum().astype(float)


class CapacityDataValidator:
    """Audit raw, prepared, or metric-enriched capacity data without mutation."""

    def __init__(self, config: ValidationConfig | None = None) -> None:
        self.config = config or ValidationConfig()

    def validate(self, data: pd.DataFrame) -> DatasetValidationResult:
        """Execute all applicable structural and business-rule checks."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        if len(data) > self.config.maximum_rows:
            raise DatasetValidationError(
                f"Dataset contains {len(data):,} rows; limit is {self.config.maximum_rows:,}."
            )

        report = DatasetValidationReport(
            source_rows=len(data),
            source_columns=len(data.columns),
        )
        if data.empty:
            report.add(
                ValidationSeverity.CRITICAL,
                "structure",
                "EMPTY_DATASET",
                "The dataset contains no observations.",
            )
            return DatasetValidationResult(report, pd.DataFrame())

        original = data.reset_index() if DATE_COLUMN not in data.columns else data.copy()
        original = original.reset_index(drop=True)
        original[ROW_NUMBER_COLUMN] = np.arange(1, len(original) + 1)
        row_codes: list[set[str]] = [set() for _ in range(len(original))]

        def flag_rows(code: str, mask: pd.Series | np.ndarray) -> None:
            positions = np.flatnonzero(np.asarray(mask, dtype=bool))
            for position in positions:
                row_codes[int(position)].add(code)

        canonical_map: dict[str, object] = {}
        canonical_duplicates: set[str] = set()
        for column in original.columns:
            canonical = _canonical_column_name(column)
            if canonical in canonical_map:
                canonical_duplicates.add(canonical)
            canonical_map[canonical] = column
        if canonical_duplicates:
            report.add(
                ValidationSeverity.CRITICAL,
                "schema",
                "AMBIGUOUS_COLUMNS",
                "Multiple source columns normalize to the same canonical name.",
                columns=tuple(sorted(canonical_duplicates)),
            )
            return DatasetValidationResult(report, pd.DataFrame())

        missing_columns = [column for column in REQUIRED_COLUMNS if column not in canonical_map]
        if missing_columns:
            report.add(
                ValidationSeverity.CRITICAL,
                "schema",
                "MISSING_REQUIRED_COLUMNS",
                "One or more required HHS capacity columns are absent.",
                columns=tuple(missing_columns),
            )
            return DatasetValidationResult(report, pd.DataFrame())

        renamed = original.rename(
            columns={source: canonical for canonical, source in canonical_map.items()}
        )
        empty_mask = (
            renamed[REQUIRED_COLUMNS]
            .apply(lambda column: column.isna() | column.astype("string").str.strip().eq(""))
            .all(axis=1)
        )
        if empty_mask.any():
            count = int(empty_mask.sum())
            report.add(
                ValidationSeverity.WARNING,
                "completeness",
                "EMPTY_ROWS",
                "Completely empty export-padding rows are present.",
                affected_rows=count,
            )
            flag_rows("EMPTY_ROWS", empty_mask)

        dates = _parse_dates(renamed[DATE_COLUMN])
        invalid_dates = dates.isna() & ~empty_mask
        if invalid_dates.any():
            report.add(
                ValidationSeverity.ERROR,
                "temporal",
                "INVALID_DATES",
                "Reporting dates contain unparseable values.",
                affected_rows=int(invalid_dates.sum()),
                columns=(DATE_COLUMN,),
                examples=tuple(renamed.loc[invalid_dates, DATE_COLUMN].astype(str).head(3)),
            )
            flag_rows("INVALID_DATES", invalid_dates)

        valid_dates = dates[dates.notna() & ~empty_mask]
        report.valid_date_rows = int(len(valid_dates))
        if not valid_dates.empty:
            report.reporting_start = valid_dates.min().date().isoformat()
            report.reporting_end = valid_dates.max().date().isoformat()
            report.expected_daily_rows = int((valid_dates.max() - valid_dates.min()).days + 1)

            chronological = valid_dates.reset_index(drop=True).is_monotonic_increasing
            if not chronological:
                report.add(
                    ValidationSeverity.WARNING,
                    "temporal",
                    "NON_CHRONOLOGICAL_ORDER",
                    "Rows are not ordered chronologically.",
                )

            duplicate_dates = dates.notna() & dates.duplicated(keep=False)
            if duplicate_dates.any():
                report.add(
                    ValidationSeverity.ERROR,
                    "temporal",
                    "DUPLICATE_DATES",
                    "Multiple observations share the same reporting date.",
                    affected_rows=int(duplicate_dates.sum()),
                    columns=(DATE_COLUMN,),
                    examples=tuple(
                        dates.loc[duplicate_dates].dt.date.astype(str).drop_duplicates().head(3)
                    ),
                )
                flag_rows("DUPLICATE_DATES", duplicate_dates)

            complete = pd.date_range(valid_dates.min(), valid_dates.max(), freq="D")
            missing_dates = complete.difference(pd.DatetimeIndex(valid_dates.unique()))
            report.missing_calendar_dates = len(missing_dates)
            if len(missing_dates):
                report.add(
                    (
                        ValidationSeverity.ERROR
                        if self.config.require_complete_daily_series
                        else ValidationSeverity.WARNING
                    ),
                    "temporal",
                    "MISSING_CALENDAR_DATES",
                    "The reporting period is not a complete daily time series.",
                    affected_rows=len(missing_dates),
                    columns=(DATE_COLUMN,),
                    examples=tuple(item.date().isoformat() for item in missing_dates[:3]),
                )

        numeric: dict[str, pd.Series] = {}
        for column in NUMERIC_COLUMNS:
            values, invalid = _numeric_values(renamed[column])
            numeric[column] = values
            missing = values.isna() & ~invalid & ~empty_mask
            nonfinite = pd.Series(np.isinf(values.astype(float).fillna(0)), index=values.index)
            fractional = values.notna() & values.mod(1).abs().gt(1e-9)
            negative = values.lt(0).fillna(False)

            for code, mask, severity, message in (
                (
                    "INVALID_NUMERIC_VALUES",
                    invalid & ~empty_mask,
                    ValidationSeverity.ERROR,
                    "Non-numeric count values are present.",
                ),
                (
                    "MISSING_NUMERIC_VALUES",
                    missing,
                    ValidationSeverity.ERROR,
                    "Required count values are missing.",
                ),
                (
                    "NONFINITE_VALUES",
                    nonfinite,
                    ValidationSeverity.ERROR,
                    "Infinite count values are present.",
                ),
                (
                    "FRACTIONAL_COUNTS",
                    fractional,
                    (
                        ValidationSeverity.WARNING
                        if self.config.allow_fractional_counts
                        else ValidationSeverity.ERROR
                    ),
                    "Child counts contain fractional values.",
                ),
                (
                    "NEGATIVE_COUNTS",
                    negative,
                    ValidationSeverity.ERROR,
                    "Child counts contain negative values.",
                ),
            ):
                if mask.any():
                    report.add(
                        severity,
                        "numeric",
                        f"{code}:{column}",
                        message,
                        affected_rows=int(mask.sum()),
                        columns=(column,),
                    )
                    flag_rows(f"{code}:{column}", mask)

            threshold = self.config.outlier_robust_z_threshold
            finite_values = values.replace([np.inf, -np.inf], np.nan)
            if threshold is not None and finite_values.notna().sum() >= 10:
                median = finite_values.median()
                deviation = (finite_values - median).abs()
                mad = deviation.median()
                if pd.notna(mad) and mad > 0:
                    robust_z = 0.6745 * deviation / mad
                    outliers = robust_z.gt(threshold).fillna(False)
                    if outliers.any():
                        code = f"ROBUST_OUTLIERS:{column}"
                        report.add(
                            ValidationSeverity.WARNING,
                            "plausibility",
                            code,
                            "Statistical outliers warrant source verification.",
                            affected_rows=int(outliers.sum()),
                            columns=(column,),
                        )
                        flag_rows(code, outliers)

        transfer_violation = numeric[TRANSFER_COLUMN].gt(numeric[CBP_COLUMN]).fillna(False)
        discharge_violation = numeric[DISCHARGE_COLUMN].gt(numeric[HHS_COLUMN]).fillna(False)
        for code, mask, columns, message in (
            (
                "TRANSFER_EXCEEDS_CBP",
                transfer_violation,
                (TRANSFER_COLUMN, CBP_COLUMN),
                "Transfers exceed active CBP custody.",
            ),
            (
                "DISCHARGE_EXCEEDS_HHS",
                discharge_violation,
                (DISCHARGE_COLUMN, HHS_COLUMN),
                "Discharges exceed active HHS care.",
            ),
        ):
            if mask.any():
                report.add(
                    ValidationSeverity.ERROR,
                    "logical",
                    code,
                    message,
                    affected_rows=int(mask.sum()),
                    columns=columns,
                )
                flag_rows(code, mask)

        self._validate_existing_flags(
            renamed,
            transfer_violation,
            discharge_violation,
            report,
            flag_rows,
        )
        if self.config.check_derived_metrics:
            self._validate_derived_metrics(renamed, numeric, dates, report, flag_rows)
        if self.config.check_stock_flow_balance:
            self._validate_stock_flow_balance(numeric, dates, report, flag_rows)

        issue_text = ["; ".join(sorted(codes)) for codes in row_codes]
        flagged_mask = pd.Series([bool(codes) for codes in row_codes])
        evidence = renamed.loc[flagged_mask].copy()
        evidence[ISSUE_CODES_COLUMN] = pd.Series(issue_text).loc[flagged_mask].to_numpy()
        report.flagged_row_count = int(flagged_mask.sum())
        if len(evidence) > self.config.maximum_flagged_rows:
            evidence = evidence.head(self.config.maximum_flagged_rows).copy()
            report.flagged_rows_truncated = True
        evidence.attrs.clear()
        return DatasetValidationResult(report=report, flagged_rows=evidence)

    def _validate_existing_flags(
        self,
        frame: pd.DataFrame,
        expected_transfer: pd.Series,
        expected_discharge: pd.Series,
        report: DatasetValidationReport,
        flag_rows: Any,
    ) -> None:
        """Check that supplied anomaly columns agree with source values."""
        for column, expected in (
            (TRANSFER_ANOMALY_COLUMN, expected_transfer),
            (DISCHARGE_ANOMALY_COLUMN, expected_discharge),
        ):
            if column not in frame.columns:
                continue
            supplied, invalid = _boolean_values(frame[column])
            if invalid.any():
                invalid_code = f"INVALID_BOOLEAN_FLAG:{column}"
                report.add(
                    ValidationSeverity.ERROR,
                    "audit",
                    invalid_code,
                    "An anomaly flag contains an unsupported boolean value.",
                    affected_rows=int(invalid.sum()),
                    columns=(column,),
                )
                flag_rows(invalid_code, invalid)
            mismatch = supplied.ne(expected)
            if mismatch.any():
                code = f"INCONSISTENT_FLAG:{column}"
                report.add(
                    ValidationSeverity.ERROR,
                    "audit",
                    code,
                    "A supplied anomaly flag disagrees with its business rule.",
                    affected_rows=int(mismatch.sum()),
                    columns=(column,),
                )
                flag_rows(code, mismatch)

    def _validate_derived_metrics(
        self,
        frame: pd.DataFrame,
        numeric: dict[str, pd.Series],
        dates: pd.Series,
        report: DatasetValidationReport,
        flag_rows: Any,
    ) -> None:
        """Recalculate supplied analytical metrics and compare row by row."""
        expected: dict[str, pd.Series] = {
            TOTAL_LOAD_COLUMN: numeric[CBP_COLUMN] + numeric[HHS_COLUMN],
            NET_INTAKE_COLUMN: numeric[TRANSFER_COLUMN] - numeric[DISCHARGE_COLUMN],
            OFFSET_RATIO_COLUMN: numeric[DISCHARGE_COLUMN].div(numeric[TRANSFER_COLUMN] + 1),
        }
        ordered = pd.DataFrame({"Date": dates, "load": expected[TOTAL_LOAD_COLUMN]}).sort_values(
            "Date", kind="stable"
        )
        growth = ordered["load"].pct_change(fill_method=None).mul(100)
        expected[GROWTH_RATE_COLUMN] = growth.reindex(ordered.index).reindex(frame.index)

        date_order = dates.sort_values(kind="stable").index
        ordered_load = expected[TOTAL_LOAD_COLUMN].loc[date_order]
        expected[ROLLING_7_COLUMN] = (
            ordered_load.rolling(7, min_periods=1).mean().reindex(frame.index)
        )
        expected[ROLLING_14_COLUMN] = (
            ordered_load.rolling(14, min_periods=1).mean().reindex(frame.index)
        )
        ordered_net = expected[NET_INTAKE_COLUMN].loc[date_order]
        expected[BACKLOG_STREAK_COLUMN] = _expected_backlog_streak(ordered_net).reindex(frame.index)

        for column, expected_values in expected.items():
            if column not in frame.columns:
                continue
            actual = pd.to_numeric(frame[column], errors="coerce")
            mismatch = ~_close_enough(actual, expected_values, self.config)
            if mismatch.any():
                code = f"DERIVED_METRIC_MISMATCH:{column}"
                report.add(
                    ValidationSeverity.ERROR,
                    "metric_integrity",
                    code,
                    "Stored derived values do not match recalculated values.",
                    affected_rows=int(mismatch.sum()),
                    columns=(column,),
                )
                flag_rows(code, mismatch)

    def _validate_stock_flow_balance(
        self,
        numeric: dict[str, pd.Series],
        dates: pd.Series,
        report: DatasetValidationReport,
        flag_rows: Any,
    ) -> None:
        """Optionally audit day-over-day stock/flow accounting residuals."""
        ordered_index = dates.sort_values(kind="stable").index
        cbp = numeric[CBP_COLUMN].loc[ordered_index]
        hhs = numeric[HHS_COLUMN].loc[ordered_index]
        expected_cbp = (
            cbp.shift(1)
            + numeric[INTAKE_COLUMN].loc[ordered_index]
            - numeric[TRANSFER_COLUMN].loc[ordered_index]
        )
        expected_hhs = (
            hhs.shift(1)
            + numeric[TRANSFER_COLUMN].loc[ordered_index]
            - numeric[DISCHARGE_COLUMN].loc[ordered_index]
        )
        for code, residual, columns in (
            (
                "CBP_STOCK_FLOW_RESIDUAL",
                (cbp - expected_cbp).abs(),
                (CBP_COLUMN, INTAKE_COLUMN, TRANSFER_COLUMN),
            ),
            (
                "HHS_STOCK_FLOW_RESIDUAL",
                (hhs - expected_hhs).abs(),
                (HHS_COLUMN, TRANSFER_COLUMN, DISCHARGE_COLUMN),
            ),
        ):
            mismatch = residual.gt(self.config.stock_flow_tolerance).reindex(
                dates.index, fill_value=False
            )
            if mismatch.any():
                report.add(
                    ValidationSeverity.WARNING,
                    "reconciliation",
                    code,
                    "Daily stock changes do not reconcile with reported flows.",
                    affected_rows=int(mismatch.sum()),
                    columns=columns,
                )
                flag_rows(code, mismatch)


def validate_capacity_data(
    data: pd.DataFrame,
    config: ValidationConfig | None = None,
) -> DatasetValidationResult:
    """Functional entry point for the complete capacity-data audit."""
    return CapacityDataValidator(config).validate(data)


def validate_or_raise(
    data: pd.DataFrame,
    config: ValidationConfig | None = None,
) -> DatasetValidationResult:
    """Validate and raise if errors (or configured strict warnings) are found."""
    validator = CapacityDataValidator(config)
    result = validator.validate(data)
    result.raise_for_errors(include_warnings=validator.config.strict_warnings)
    return result
