"""Data engineering and analytics utilities for the HHS care-load dashboard.

The functions in this module deliberately contain no Streamlit code.  Keeping the
data layer independent makes it straightforward to test, reuse in a scheduled
pipeline, or expose through an API later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from io import BytesIO, StringIO
from pathlib import Path
from typing import BinaryIO, TextIO

import numpy as np
import pandas as pd


DATE_COLUMN = "Date"
INTAKE_COLUMN = "Children apprehended and placed in CBP custody"
CBP_COLUMN = "Children in CBP custody"
TRANSFER_COLUMN = "Children transferred out of CBP custody"
HHS_COLUMN = "Children in HHS Care"
DISCHARGE_COLUMN = "Children discharged from HHS Care"

FLOW_COLUMNS = [INTAKE_COLUMN, TRANSFER_COLUMN, DISCHARGE_COLUMN]
STOCK_COLUMNS = [CBP_COLUMN, HHS_COLUMN]
NUMERIC_COLUMNS = FLOW_COLUMNS + STOCK_COLUMNS
REQUIRED_COLUMNS = [DATE_COLUMN, *NUMERIC_COLUMNS]

# Names for calculated fields are constants so the frontend never depends on
# fragile, repeated string literals.
TOTAL_LOAD_COLUMN = "Total System Load"
NET_INTAKE_COLUMN = "Net Daily Intake"
GROWTH_RATE_COLUMN = "Care Load Growth Rate"
ROLLING_7_COLUMN = "7-Day Rolling Average"
ROLLING_14_COLUMN = "14-Day Rolling Average"
OFFSET_RATIO_COLUMN = "Discharge Offset Ratio"
BACKLOG_STREAK_COLUMN = "Backlog Accumulation Rate"
QUALITY_FLAG_COLUMN = "Data Quality Flag"
TRANSFER_ANOMALY_COLUMN = "Anomaly_Transfer_Exceeds_CBP"
DISCHARGE_ANOMALY_COLUMN = "Anomaly_Discharge_Exceeds_HHS"


class DataValidationError(ValueError):
    """Raised when input data cannot be made safe enough for analysis."""


@dataclass(frozen=True)
class ValidationIssue:
    """A validation finding suitable for logs and the dashboard UI."""

    severity: str
    code: str
    message: str
    affected_rows: int = 0


@dataclass
class ValidationReport:
    """Collect all findings rather than failing on the first recoverable issue."""

    issues: list[ValidationIssue] = field(default_factory=list)

    def add(self, severity: str, code: str, message: str, affected_rows: int = 0) -> None:
        self.issues.append(ValidationIssue(severity, code, message, int(affected_rows)))

    @property
    def error_count(self) -> int:
        return sum(issue.severity == "error" for issue in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(issue.severity == "warning" for issue in self.issues)

    def to_frame(self) -> pd.DataFrame:
        """Return findings in a display-friendly table."""
        if not self.issues:
            return pd.DataFrame(
                [{"Severity": "info", "Code": "OK", "Message": "No issues found", "Rows": 0}]
            )
        return pd.DataFrame(
            [
                {
                    "Severity": item.severity,
                    "Code": item.code,
                    "Message": item.message,
                    "Rows": item.affected_rows,
                }
                for item in self.issues
            ]
        )


def _canonical_column_name(name: object) -> str:
    """Normalize whitespace and the footnote marker present in the HHS CSV."""
    value = " ".join(str(name).replace("\ufeff", "").strip().split())
    return value.rstrip("*").strip()


def _parse_reporting_dates(values: pd.Series) -> pd.Series:
    """Parse ISO and human-readable dates, including mixed-format extracts."""
    try:
        # Pandas 2+ can parse mixed-format arrays without locking onto the first
        # observed format.
        return pd.to_datetime(values, format="mixed", errors="coerce")
    except (TypeError, ValueError):
        # Compatibility fallback for older pandas versions.
        return values.apply(lambda value: pd.to_datetime(value, errors="coerce"))


def read_csv_data(
    source: str | Path | bytes | BinaryIO | TextIO,
) -> pd.DataFrame:
    """Read a CSV from a path, uploaded bytes, or file-like object.

    All fields are initially read as strings.  Numeric and date conversion is
    performed centrally by :func:`validate_and_prepare_data`, where failures can
    be reported rather than silently converted.
    """
    try:
        if isinstance(source, bytes):
            source = BytesIO(source)
        return pd.read_csv(source, dtype=str, encoding="utf-8-sig")
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DataValidationError(f"The CSV could not be read: {exc}") from exc


def generate_synthetic_data(
    start: str | date = "2023-01-01",
    end: str | date = "2025-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate deterministic, realistic-looking daily data for demonstrations.

    The simulation uses stock/flow accounting with seasonality, random shocks,
    and gentle mean reversion.  It is clearly synthetic and is not intended to
    reproduce or forecast official HHS statistics.
    """
    dates = pd.date_range(start=start, end=end, freq="D")
    if dates.empty:
        raise ValueError("The synthetic-data end date must be on or after the start date.")

    rng = np.random.default_rng(seed)
    records: list[dict[str, int | pd.Timestamp]] = []
    cbp_load = 680
    hhs_load = 7_800

    for day_number, current_date in enumerate(dates):
        annual_wave = np.sin(2 * np.pi * day_number / 365.25)
        weekly_wave = np.sin(2 * np.pi * day_number / 7)
        # Several short, deterministic surges make the demo useful for showing
        # backlog shading and rolling-average behavior.
        surge = 105 if (day_number % 310) in range(0, 18) else 0
        intake_mean = max(40, 215 + 48 * annual_wave + 16 * weekly_wave + surge)
        apprehended = int(rng.poisson(intake_mean))

        transfer_target = apprehended + 0.12 * (cbp_load - 680) + rng.normal(0, 22)
        transferred = max(0, int(round(transfer_target)))
        # Retain enough children in the end-of-day stock to satisfy the logical
        # constraint requested for the project.
        transferred = min(transferred, (cbp_load + apprehended) // 2)
        cbp_load = max(0, cbp_load + apprehended - transferred)

        discharge_target = transferred + 0.025 * (hhs_load - 7_800) + rng.normal(0, 26)
        discharged = max(0, int(round(discharge_target)))
        discharged = min(discharged, (hhs_load + transferred) // 2)
        hhs_load = max(0, hhs_load + transferred - discharged)

        records.append(
            {
                DATE_COLUMN: current_date,
                INTAKE_COLUMN: apprehended,
                CBP_COLUMN: cbp_load,
                TRANSFER_COLUMN: transferred,
                HHS_COLUMN: hhs_load,
                DISCHARGE_COLUMN: discharged,
            }
        )

    return pd.DataFrame.from_records(records, columns=REQUIRED_COLUMNS)


def generate_mock_data() -> pd.DataFrame:
    """Generate reproducible daily mock data from 2023-01-01 to 2025-12-31.

    The generated values include annual and weekly influx cycles, temporary
    surge periods, random operational variation, and mean-reverting CBP/HHS
    care loads.  The output is synthetic and must not be represented as official
    HHS or CBP data.

    Returns:
        A chronologically ordered DataFrame containing the six required columns.

    Raises:
        RuntimeError: If mock-data generation unexpectedly fails.
    """
    try:
        return generate_synthetic_data(
            start="2023-01-01",
            end="2025-12-31",
            seed=42,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"Unable to generate mock capacity data: {exc}") from exc


def validate_and_prepare_data(
    raw_data: pd.DataFrame,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate, repair recoverable gaps, and add transparent quality flags.

    Repairs are intentionally conservative:

    * rows are sorted chronologically;
    * duplicate dates retain the last reported row;
    * missing calendar dates are inserted;
    * stocks are time-interpolated, while unreported daily flows become zero;
    * invalid logical relationships remain in the data and are visibly flagged.

    Unusable rows (for example, an invalid date) are removed and reported.  A
    completely unusable dataset raises :class:`DataValidationError`.
    """
    if raw_data is None or raw_data.empty:
        raise DataValidationError("The dataset is empty.")

    report = ValidationReport()
    frame = raw_data.copy()

    # Match the official file's footnoted intake header to the expected schema.
    normalized_names = {_canonical_column_name(name): name for name in frame.columns}
    missing_columns = [name for name in REQUIRED_COLUMNS if name not in normalized_names]
    if missing_columns:
        raise DataValidationError("Missing required column(s): " + ", ".join(missing_columns))
    frame = frame.rename(
        columns={original: canonical for canonical, original in normalized_names.items()}
    )[REQUIRED_COLUMNS]

    # Some published spreadsheets export hundreds of comma-only padding rows.
    # They contain no observations and should not inflate the invalid-date count.
    empty_row_mask = frame.apply(
        lambda column: column.isna() | column.astype("string").str.strip().eq("")
    ).all(axis=1)
    if empty_row_mask.any():
        report.add(
            "warning",
            "EMPTY_ROWS_IGNORED",
            "Completely empty rows were ignored.",
            int(empty_row_mask.sum()),
        )
        frame = frame.loc[~empty_row_mask].copy()
    if frame.empty:
        raise DataValidationError("The dataset contains no populated rows.")

    original_dates = frame[DATE_COLUMN].copy()
    frame[DATE_COLUMN] = _parse_reporting_dates(frame[DATE_COLUMN])
    invalid_date_mask = frame[DATE_COLUMN].isna()
    if invalid_date_mask.any():
        report.add(
            "error",
            "INVALID_DATE",
            "Rows with unparseable dates were removed.",
            int(invalid_date_mask.sum()),
        )
        frame = frame.loc[~invalid_date_mask].copy()
    if frame.empty:
        examples = original_dates.dropna().astype(str).head(3).tolist()
        raise DataValidationError(f"No valid reporting dates were found. Examples: {examples}")

    # A descending official extract is common and harmless once explicitly fixed.
    if not frame[DATE_COLUMN].is_monotonic_increasing:
        report.add(
            "warning",
            "CHRONOLOGY_REPAIRED",
            "Rows were not in chronological order and have been sorted.",
        )

    for column in NUMERIC_COLUMNS:
        cleaned = frame[column].astype("string").str.replace(",", "", regex=False).str.strip()
        cleaned = cleaned.mask(cleaned.eq(""), pd.NA)
        converted = pd.to_numeric(cleaned, errors="coerce")
        invalid_numeric = cleaned.notna() & converted.isna()
        if invalid_numeric.any():
            report.add(
                "error",
                "INVALID_NUMBER",
                f"Non-numeric values in '{column}' were treated as missing.",
                int(invalid_numeric.sum()),
            )
        frame[column] = converted

    frame = frame.sort_values(DATE_COLUMN, kind="stable")
    duplicate_mask = frame.duplicated(DATE_COLUMN, keep="last")
    if duplicate_mask.any():
        report.add(
            "error",
            "DUPLICATE_DATE",
            "Duplicate reporting dates were found; the last row was retained.",
            int(duplicate_mask.sum()),
        )
        frame = frame.loc[~duplicate_mask].copy()

    frame = frame.set_index(DATE_COLUMN)
    complete_index = pd.date_range(frame.index.min(), frame.index.max(), freq="D")
    inserted_dates = ~complete_index.isin(frame.index)
    inserted_count = int(inserted_dates.sum())
    frame = frame.reindex(complete_index)
    frame.index.name = DATE_COLUMN
    frame["Is Imputed Date"] = inserted_dates
    if inserted_count:
        report.add(
            "warning",
            "MISSING_DATES_FILLED",
            "Missing calendar dates were inserted; stocks were interpolated and flows set to zero.",
            inserted_count,
        )

    # Missing values on reported dates are also imputed, but called out separately.
    reported_missing = frame.loc[~frame["Is Imputed Date"], NUMERIC_COLUMNS].isna()
    missing_value_count = int(reported_missing.sum().sum())
    if missing_value_count:
        report.add(
            "error",
            "MISSING_VALUE",
            "Missing reported numeric values were imputed.",
            missing_value_count,
        )

    # Stocks are state variables and can be interpolated over time.  Flow values on
    # inserted dates are not observed, so zero is the least-assumptive placeholder.
    frame[STOCK_COLUMNS] = frame[STOCK_COLUMNS].interpolate(method="time", limit_direction="both")
    frame[FLOW_COLUMNS] = frame[FLOW_COLUMNS].fillna(0)
    frame[NUMERIC_COLUMNS] = frame[NUMERIC_COLUMNS].fillna(0).round().astype("int64")

    negative_mask = frame[NUMERIC_COLUMNS].lt(0).any(axis=1)
    if negative_mask.any():
        report.add(
            "error",
            "NEGATIVE_COUNT",
            "Negative child counts violate the non-negative count constraint.",
            int(negative_mask.sum()),
        )

    frame["Transfer Constraint Violation"] = frame[TRANSFER_COLUMN] > frame[CBP_COLUMN]
    frame["Discharge Constraint Violation"] = frame[DISCHARGE_COLUMN] > frame[HHS_COLUMN]
    transfer_errors = int(frame["Transfer Constraint Violation"].sum())
    discharge_errors = int(frame["Discharge Constraint Violation"].sum())
    if transfer_errors:
        report.add(
            "error",
            "TRANSFER_EXCEEDS_CBP",
            "Transfers exceed active CBP custody on one or more dates.",
            transfer_errors,
        )
    if discharge_errors:
        report.add(
            "error",
            "DISCHARGE_EXCEEDS_HHS",
            "Discharges exceed active HHS care on one or more dates.",
            discharge_errors,
        )

    # A concise row-level string supports filtering and CSV exports.
    def row_quality_flags(row: pd.Series) -> str:
        flags: list[str] = []
        if bool(row["Is Imputed Date"]):
            flags.append("Imputed date")
        if bool(row["Transfer Constraint Violation"]):
            flags.append("Transfer > CBP custody")
        if bool(row["Discharge Constraint Violation"]):
            flags.append("Discharge > HHS care")
        if bool((row[NUMERIC_COLUMNS] < 0).any()):
            flags.append("Negative count")
        return "; ".join(flags) if flags else "OK"

    frame[QUALITY_FLAG_COLUMN] = frame.apply(row_quality_flags, axis=1)
    return frame.reset_index(), report


def validate_and_clean_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean HHS UAC capacity data into a daily time series.

    Processing includes schema validation, flexible date parsing, chronological
    sorting, duplicate-date handling, daily reindexing, numeric coercion, stock
    interpolation, flow-value filling, and logical-anomaly detection.  ``Date``
    is returned as a ``DatetimeIndex`` as required by downstream time-series
    operations.

    The returned DataFrame contains these explicit boolean audit columns:

    * ``Anomaly_Transfer_Exceeds_CBP``
    * ``Anomaly_Discharge_Exceeds_HHS``

    A structured validation report is retained in ``result.attrs`` under the
    ``validation_report`` key so callers can inspect repairs and anomalies.

    Args:
        raw_data: DataFrame containing the six required source columns.

    Returns:
        A cleaned, daily, chronologically sorted DataFrame indexed by ``Date``.

    Raises:
        DataValidationError: If the input is empty or lacks required columns.
        TypeError: If ``raw_data`` is not a pandas DataFrame.
    """
    if not isinstance(raw_data, pd.DataFrame):
        raise TypeError("raw_data must be a pandas DataFrame.")

    prepared, report = validate_and_prepare_data(raw_data)
    cleaned = prepared.rename(
        columns={
            "Transfer Constraint Violation": TRANSFER_ANOMALY_COLUMN,
            "Discharge Constraint Violation": DISCHARGE_ANOMALY_COLUMN,
        }
    ).set_index(DATE_COLUMN)

    cleaned.index = pd.DatetimeIndex(cleaned.index, name=DATE_COLUMN)
    cleaned = cleaned.sort_index()
    cleaned[TRANSFER_ANOMALY_COLUMN] = cleaned[TRANSFER_ANOMALY_COLUMN].fillna(False).astype(bool)
    cleaned[DISCHARGE_ANOMALY_COLUMN] = cleaned[DISCHARGE_ANOMALY_COLUMN].fillna(False).astype(bool)
    cleaned.attrs["validation_report"] = report
    return cleaned


def calculate_metrics(prepared_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate row-level load, pressure, smoothing, and backlog metrics."""
    if prepared_data is None or prepared_data.empty:
        raise DataValidationError("No prepared rows are available for analytics.")

    frame = prepared_data.copy().sort_values(DATE_COLUMN).reset_index(drop=True)
    frame[TOTAL_LOAD_COLUMN] = frame[CBP_COLUMN] + frame[HHS_COLUMN]
    frame[NET_INTAKE_COLUMN] = frame[TRANSFER_COLUMN] - frame[DISCHARGE_COLUMN]
    frame[GROWTH_RATE_COLUMN] = (
        frame[TOTAL_LOAD_COLUMN].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        * 100
    )
    frame[ROLLING_7_COLUMN] = frame[TOTAL_LOAD_COLUMN].rolling(7, min_periods=1).mean()
    frame[ROLLING_14_COLUMN] = frame[TOTAL_LOAD_COLUMN].rolling(14, min_periods=1).mean()
    frame[OFFSET_RATIO_COLUMN] = frame[DISCHARGE_COLUMN].div(frame[TRANSFER_COLUMN].fillna(0) + 1)

    # Count each run of positive net intake and reset to zero on non-positive days.
    positive = frame[NET_INTAKE_COLUMN].gt(0)
    groups = positive.ne(positive.shift(fill_value=False)).cumsum()
    frame[BACKLOG_STREAK_COLUMN] = positive.groupby(groups).cumsum().astype(int)
    return frame


def compute_capacity_metrics(cleaned_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate system-load, intake-pressure, growth, and rolling metrics.

    ``cleaned_data`` may contain ``Date`` as either a column or a DatetimeIndex.
    The returned object follows the input convention.  Growth is expressed as a
    percentage and non-finite values caused by a zero prior load become ``NaN``.

    Args:
        cleaned_data: Validated daily capacity data.

    Returns:
        A copy containing Total System Load, Net Daily Intake, Care Load Growth
        Rate, and 7-day/14-day rolling averages.

    Raises:
        DataValidationError: If the input is empty or lacks required fields.
        TypeError: If the input is not a pandas DataFrame.
    """
    if not isinstance(cleaned_data, pd.DataFrame):
        raise TypeError("cleaned_data must be a pandas DataFrame.")
    if cleaned_data.empty:
        raise DataValidationError("No cleaned rows are available for analytics.")

    was_date_indexed = DATE_COLUMN not in cleaned_data.columns
    if was_date_indexed:
        if cleaned_data.index.name != DATE_COLUMN:
            raise DataValidationError(
                "Date must be provided as a column or as an index named 'Date'."
            )
        working = cleaned_data.reset_index()
    else:
        working = cleaned_data.copy()

    missing = [
        column
        for column in [CBP_COLUMN, TRANSFER_COLUMN, HHS_COLUMN, DISCHARGE_COLUMN]
        if column not in working.columns
    ]
    if missing:
        raise DataValidationError("Missing metric input column(s): " + ", ".join(missing))

    result = calculate_metrics(working)
    # Guard explicitly against division by zero and corrupt infinite values.
    result[GROWTH_RATE_COLUMN] = pd.to_numeric(result[GROWTH_RATE_COLUMN], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    result[ROLLING_7_COLUMN] = result[ROLLING_7_COLUMN].fillna(result[TOTAL_LOAD_COLUMN])
    result[ROLLING_14_COLUMN] = result[ROLLING_14_COLUMN].fillna(result[TOTAL_LOAD_COLUMN])

    if was_date_indexed:
        result = result.set_index(DATE_COLUMN)
        result.index = pd.DatetimeIndex(result.index, name=DATE_COLUMN)
    return result


def calculate_kpis(analytics_data: pd.DataFrame) -> dict[str, float | int]:
    """Return five headline KPIs for a filtered analytical period.

    The backlog accumulation rate is the *longest* consecutive run of positive
    net daily intake within the filtered period.  Volatility ignores ``NaN`` and
    infinite growth values and uses population standard deviation (``ddof=0``).

    Args:
        analytics_data: Filtered output from :func:`compute_capacity_metrics` or
            :func:`calculate_metrics`.

    Returns:
        A dictionary of numeric KPI values using stable snake_case keys.

    Raises:
        DataValidationError: If no rows are supplied or required metric columns
            are missing.
        TypeError: If ``analytics_data`` is not a pandas DataFrame.
    """
    if not isinstance(analytics_data, pd.DataFrame):
        raise TypeError("analytics_data must be a pandas DataFrame.")
    if analytics_data.empty:
        raise DataValidationError("No rows are available for KPI calculation.")

    required_metrics = [
        TOTAL_LOAD_COLUMN,
        NET_INTAKE_COLUMN,
        GROWTH_RATE_COLUMN,
        TRANSFER_COLUMN,
        DISCHARGE_COLUMN,
    ]
    missing = [column for column in required_metrics if column not in analytics_data.columns]
    if missing:
        raise DataValidationError("Missing KPI input column(s): " + ", ".join(missing))

    latest = analytics_data.iloc[-1]
    growth = (
        pd.to_numeric(analytics_data[GROWTH_RATE_COLUMN], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    volatility = float(growth.std(ddof=0)) if not growth.empty else 0.0
    latest_transfers = pd.to_numeric(pd.Series([latest[TRANSFER_COLUMN]]), errors="coerce").iloc[0]
    latest_discharges = pd.to_numeric(pd.Series([latest[DISCHARGE_COLUMN]]), errors="coerce").iloc[
        0
    ]
    latest_transfers = 0.0 if pd.isna(latest_transfers) else float(latest_transfers)
    latest_discharges = 0.0 if pd.isna(latest_discharges) else float(latest_discharges)
    ratio = latest_discharges / (latest_transfers + 1.0)
    if not np.isfinite(ratio):
        ratio = 0.0

    positive_intake = (
        pd.to_numeric(analytics_data[NET_INTAKE_COLUMN], errors="coerce").fillna(0).gt(0)
    )
    streak_groups = positive_intake.ne(positive_intake.shift(fill_value=False)).cumsum()
    streaks = positive_intake.groupby(streak_groups).cumsum()
    longest_positive_streak = int(streaks.max()) if not streaks.empty else 0

    return {
        "total_children_under_care": int(
            pd.to_numeric(pd.Series([latest[TOTAL_LOAD_COLUMN]]), errors="coerce").fillna(0).iloc[0]
        ),
        "net_intake_pressure": int(
            pd.to_numeric(pd.Series([latest[NET_INTAKE_COLUMN]]), errors="coerce").fillna(0).iloc[0]
        ),
        "care_load_volatility_index": volatility,
        "backlog_accumulation_rate": longest_positive_streak,
        "discharge_offset_ratio": ratio,
    }


def aggregate_by_granularity(prepared_data: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """Aggregate for chart display, retaining end-of-period active loads.

    Daily flow values are summed.  CBP and HHS care loads are stocks, so the last
    observation in each week/month represents the period endpoint.
    """
    normalized = granularity.strip().lower()
    if normalized == "daily":
        return calculate_metrics(prepared_data)
    frequency = {
        "weekly": pd.offsets.Week(weekday=6),
        "monthly": pd.offsets.MonthEnd(),
    }.get(normalized)
    if frequency is None:
        raise ValueError("Granularity must be Daily, Weekly, or Monthly.")

    # Calculate rolling averages at the daily level first.  Weekly/monthly charts
    # then sample those daily statistics at period end; they never relabel a
    # 7-week or 7-month statistic as a "7-day" average.
    frame = calculate_metrics(prepared_data).set_index(DATE_COLUMN).sort_index()
    aggregations: dict[str, str] = {
        INTAKE_COLUMN: "sum",
        TRANSFER_COLUMN: "sum",
        DISCHARGE_COLUMN: "sum",
        CBP_COLUMN: "last",
        HHS_COLUMN: "last",
        ROLLING_7_COLUMN: "last",
        ROLLING_14_COLUMN: "last",
        BACKLOG_STREAK_COLUMN: "last",
    }
    aggregated = frame.resample(frequency).agg(aggregations).dropna(how="all").reset_index()
    aggregated[TOTAL_LOAD_COLUMN] = aggregated[CBP_COLUMN] + aggregated[HHS_COLUMN]
    aggregated[NET_INTAKE_COLUMN] = aggregated[TRANSFER_COLUMN] - aggregated[DISCHARGE_COLUMN]
    aggregated[GROWTH_RATE_COLUMN] = (
        aggregated[TOTAL_LOAD_COLUMN]
        .pct_change(fill_method=None)
        .replace([np.inf, -np.inf], np.nan)
        * 100
    )
    aggregated[OFFSET_RATIO_COLUMN] = aggregated[DISCHARGE_COLUMN].div(
        aggregated[TRANSFER_COLUMN] + 1
    )
    return aggregated


def find_high_backlog_periods(
    daily_analytics: pd.DataFrame, threshold_days: int = 3
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return contiguous intervals where the positive-intake streak is high."""
    if daily_analytics.empty:
        return []
    qualifying = daily_analytics[BACKLOG_STREAK_COLUMN].ge(threshold_days)
    dates = pd.to_datetime(daily_analytics.loc[qualifying, DATE_COLUMN]).sort_values()
    if dates.empty:
        return []

    groups = dates.diff().dt.days.ne(1).cumsum()
    return [(part.min(), part.max()) for _, part in dates.groupby(groups)]


def dataframe_to_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize a dataframe for a Streamlit download button."""
    buffer = StringIO()
    frame.to_csv(buffer, index=False, date_format="%Y-%m-%d")
    return buffer.getvalue().encode("utf-8")
