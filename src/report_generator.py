"""Executive and technical report generation for HHS UAC capacity analytics.

The generator is presentation-framework independent.  It accepts either source
data or an existing :class:`backend.analytics.AnalysisResult` and produces a
portable report model that can be exported as accessible HTML, strict JSON, or
individual CSV tables without requiring Streamlit.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app_utils import (
    BACKLOG_STREAK_COLUMN,
    GROWTH_RATE_COLUMN,
    NET_INTAKE_COLUMN,
    OFFSET_RATIO_COLUMN,
    TOTAL_LOAD_COLUMN,
)
from backend.analytics import (
    AnalysisResult,
    TimeGranularity,
    run_capacity_analysis,
)
from src.kpi import KPIDashboardResult, KPIStatus, calculate_kpi_dashboard
from src.validation import (
    DatasetValidationResult,
    ValidationConfig,
    validate_capacity_data,
)


class ReportGenerationError(RuntimeError):
    """Raised when a capacity report cannot be generated or exported."""


class ReportFormat(str, Enum):
    """Supported report serialization formats."""

    HTML = "html"
    JSON = "json"

    @classmethod
    def parse(cls, value: str | ReportFormat) -> ReportFormat:
        """Normalize a string or enum into a supported output format."""
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().casefold()
        for member in cls:
            if member.value == normalized:
                return member
        raise ValueError("format must be 'html' or 'json'.")


class InsightSeverity(str, Enum):
    """Visual priority for a deterministic report insight."""

    INFORMATIONAL = "informational"
    POSITIVE = "positive"
    WATCH = "watch"
    CRITICAL = "critical"


@dataclass(frozen=True)
class ReportConfig:
    """Content, analysis, and disclosure settings for generated reports."""

    title: str = "System Capacity & Care Load Analytics"
    subtitle: str = "Unaccompanied Children Program"
    agency_name: str = "U.S. Department of Health and Human Services Framework"
    source_label: str = "User-provided or configured dataset"
    classification_label: str = "FOR ANALYTICAL USE"
    synthetic_data: bool = False
    start_date: str | date | datetime | pd.Timestamp | None = None
    end_date: str | date | datetime | pd.Timestamp | None = None
    granularity: str | TimeGranularity = TimeGranularity.DAILY
    backlog_threshold_days: int = 3
    max_backlog_rows: int = 10
    max_anomaly_rows: int = 100
    max_daily_appendix_rows: int = 500
    include_daily_appendix: bool = True
    validation_config: ValidationConfig = field(default_factory=ValidationConfig)

    def __post_init__(self) -> None:
        if not self.title.strip() or not self.subtitle.strip():
            raise ValueError("Report title and subtitle must not be empty.")
        if self.backlog_threshold_days < 1:
            raise ValueError("backlog_threshold_days must be at least 1.")
        if (
            min(
                self.max_backlog_rows,
                self.max_anomaly_rows,
                self.max_daily_appendix_rows,
            )
            < 1
        ):
            raise ValueError("Report table row limits must be positive.")


@dataclass(frozen=True)
class ReportInsight:
    """One evidence-based, deterministic narrative finding."""

    severity: InsightSeverity
    title: str
    message: str
    evidence: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-compatible representation."""
        return {
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "evidence": self.evidence,
        }


@dataclass
class CapacityReport:
    """Portable report model containing narrative, tables, and audit evidence."""

    config: ReportConfig
    generated_at: datetime
    metadata: dict[str, Any]
    executive_summary: str
    insights: list[ReportInsight]
    kpi_table: pd.DataFrame
    operational_table: pd.DataFrame
    validation_table: pd.DataFrame
    backlog_table: pd.DataFrame
    anomaly_table: pd.DataFrame
    daily_appendix: pd.DataFrame
    analysis: AnalysisResult
    validation: DatasetValidationResult

    @property
    def filename_stem(self) -> str:
        """Return a stable, filesystem-safe report filename stem."""
        start = str(self.metadata["period_start"]).replace("-", "")
        end = str(self.metadata["period_end"]).replace("-", "")
        return f"hhs_uac_capacity_report_{start}_{end}"

    def table(self, name: str) -> pd.DataFrame:
        """Return a defensive copy of a named report table."""
        tables = {
            "kpis": self.kpi_table,
            "operations": self.operational_table,
            "operational": self.operational_table,
            "validation": self.validation_table,
            "backlog": self.backlog_table,
            "backlog_episodes": self.backlog_table,
            "anomalies": self.anomaly_table,
            "daily": self.daily_appendix,
            "daily_appendix": self.daily_appendix,
        }
        key = name.strip().casefold()
        if key not in tables:
            raise KeyError(f"Unknown report table '{name}'. Available: {', '.join(tables)}")
        return tables[key].copy()

    def to_dict(self, *, include_daily_appendix: bool | None = None) -> dict[str, Any]:
        """Serialize report content into JSON-safe primitives."""
        include_daily = (
            self.config.include_daily_appendix
            if include_daily_appendix is None
            else include_daily_appendix
        )
        payload: dict[str, Any] = {
            "metadata": self.metadata,
            "executive_summary": self.executive_summary,
            "insights": [item.to_dict() for item in self.insights],
            "kpis": _records(self.kpi_table),
            "operational_summary": _records(self.operational_table),
            "validation": {
                "summary": self.validation.report.to_dict(),
                "findings": _records(self.validation_table),
            },
            "backlog_episodes": _records(self.backlog_table),
            "anomalies": _records(self.anomaly_table),
        }
        if include_daily:
            payload["daily_appendix"] = _records(self.daily_appendix)
        return _json_safe(payload)

    def to_json_bytes(
        self,
        *,
        indent: int = 2,
        include_daily_appendix: bool | None = None,
    ) -> bytes:
        """Return a strict UTF-8 JSON report suitable for download or an API."""
        return json.dumps(
            self.to_dict(include_daily_appendix=include_daily_appendix),
            indent=indent,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")

    def to_html(self) -> str:
        """Render a standalone, accessible government-analytics-style report."""
        return _render_html(self)

    def to_html_bytes(self) -> bytes:
        """Return the standalone HTML report as UTF-8 bytes."""
        return self.to_html().encode("utf-8")

    def table_to_csv_bytes(self, name: str) -> bytes:
        """Export one named report table as UTF-8 CSV bytes."""
        table = self.table(name)
        table.attrs.clear()
        return table.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _json_safe(value: Any) -> Any:
    """Recursively convert pandas and NumPy values into strict JSON values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame, including a meaningful index, into record objects."""
    if frame.empty:
        return []
    output = frame.copy()
    if not isinstance(output.index, pd.RangeIndex):
        index_name = output.index.name or "Index"
        if index_name not in output.columns:
            output = output.reset_index(names=index_name)
    output.attrs.clear()
    return [_json_safe(record) for record in output.to_dict(orient="records")]


def _format_number(value: Any) -> str:
    """Format scalar report values without hiding unavailable observations."""
    if value is None:
        return "Not available"
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(float(value)):
            return "Not available"
        return f"{float(value):,.2f}"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return str(value)


def _build_operational_table(summary: dict[str, int | float | str]) -> pd.DataFrame:
    """Convert stable operational summary keys into readable labels."""
    labels = {
        "period_start": "Reporting period start",
        "period_end": "Reporting period end",
        "daily_observations": "Daily observations",
        "average_system_load": "Average system load",
        "peak_system_load": "Peak system load",
        "peak_system_load_date": "Peak system load date",
        "cumulative_net_intake": "Cumulative net intake",
        "positive_pressure_days": "Positive-pressure days",
        "current_backlog_streak": "Current backlog streak (days)",
        "elevated_backlog_episodes": "Elevated backlog episodes",
        "total_transfers": "Total transfers",
        "total_discharges": "Total discharges",
        "logical_anomaly_rows": "Logical anomaly rows",
        "imputed_dates": "Imputed dates",
    }
    rows = [
        {
            "Measure": label,
            "Value": _format_number(summary[key]),
        }
        for key, label in labels.items()
        if key in summary
    ]
    return pd.DataFrame.from_records(rows, columns=["Measure", "Value"])


def _build_insights(
    analysis: AnalysisResult,
    kpis: KPIDashboardResult,
    validation: DatasetValidationResult,
) -> list[ReportInsight]:
    """Create deterministic operational findings from calculated evidence."""
    summary = analysis.operational_summary
    insights: list[ReportInsight] = []

    for alert in kpis.alerts:
        severity = (
            InsightSeverity.CRITICAL
            if alert.severity == KPIStatus.CRITICAL.value
            else InsightSeverity.WATCH
        )
        insights.append(
            ReportInsight(
                severity=severity,
                title=alert.title,
                message=alert.message,
                evidence=f"KPI as of {alert.as_of_date}",
            )
        )

    current_streak = int(summary.get("current_backlog_streak", 0))
    threshold = int(summary.get("backlog_threshold_days", 3))
    if current_streak >= threshold:
        insights.append(
            ReportInsight(
                severity=InsightSeverity.WATCH,
                title="Active backlog accumulation episode",
                message=(
                    "Transfers have exceeded discharges for a sustained run at "
                    "the end of the selected period."
                ),
                evidence=f"Current positive-pressure streak: {current_streak} days",
            )
        )

    latest_pressure = int(summary.get("latest_net_intake", 0))
    if latest_pressure <= 0:
        insights.append(
            ReportInsight(
                severity=InsightSeverity.POSITIVE,
                title="Latest flow balance shows relief",
                message="Latest HHS discharges meet or exceed transfers into HHS care.",
                evidence=f"Latest net daily intake: {latest_pressure:+,}",
            )
        )

    anomaly_rows = int(summary.get("logical_anomaly_rows", 0))
    if anomaly_rows:
        insights.append(
            ReportInsight(
                severity=InsightSeverity.CRITICAL,
                title="Logical data anomalies require review",
                message=(
                    "At least one transfer or discharge value exceeds its active "
                    "care-load constraint."
                ),
                evidence=f"Flagged analytical rows: {anomaly_rows:,}",
            )
        )

    if validation.report.warning_count and validation.report.is_valid:
        insights.append(
            ReportInsight(
                severity=InsightSeverity.INFORMATIONAL,
                title="Validation completed with warnings",
                message="Review the data-quality appendix before external use.",
                evidence=(
                    f"Quality score: {validation.report.quality_score}/100; "
                    f"warning types: {validation.report.warning_count}"
                ),
            )
        )

    if not insights:
        insights.append(
            ReportInsight(
                severity=InsightSeverity.INFORMATIONAL,
                title="No threshold-based alerts",
                message="No configured KPI or data-quality alert threshold was crossed.",
                evidence=f"Report period ends {summary['period_end']}",
            )
        )
    return insights


def _executive_summary(
    analysis: AnalysisResult,
    validation: DatasetValidationResult,
) -> str:
    """Build a concise, neutral summary grounded in calculated values."""
    summary = analysis.operational_summary
    offset = float(summary["latest_discharge_offset_ratio"])
    return (
        f"From {summary['period_start']} through {summary['period_end']}, the system "
        f"recorded an average active care load of {summary['average_system_load']:,.0f} "
        f"children and a peak of {summary['peak_system_load']:,} on "
        f"{summary['peak_system_load_date']}. The latest total load was "
        f"{summary['latest_system_load']:,}, while latest net intake was "
        f"{summary['latest_net_intake']:+,}. The latest discharge offset ratio was "
        f"{offset:.1%}. Dataset validation status: {validation.report.status} "
        f"({validation.report.quality_score}/100)."
    )


def _prepare_appendix(frame: pd.DataFrame, maximum_rows: int) -> pd.DataFrame:
    """Create a bounded, newest-first daily appendix with useful core metrics."""
    columns = [
        column
        for column in (
            TOTAL_LOAD_COLUMN,
            NET_INTAKE_COLUMN,
            GROWTH_RATE_COLUMN,
            OFFSET_RATIO_COLUMN,
            BACKLOG_STREAK_COLUMN,
        )
        if column in frame.columns
    ]
    output = frame[columns].sort_index(ascending=False).head(maximum_rows).reset_index()
    output.attrs.clear()
    return output


class CapacityReportGenerator:
    """Build consistent analytical reports from source data or analysis results."""

    def __init__(self, config: ReportConfig | None = None) -> None:
        self.config = config or ReportConfig()

    def generate(
        self,
        source: pd.DataFrame | AnalysisResult,
        *,
        generated_at: datetime | None = None,
    ) -> CapacityReport:
        """Generate a complete report model without writing to disk."""
        try:
            analysis = self._analysis_result(source)
            validation = validate_capacity_data(
                analysis.daily_metrics,
                self.config.validation_config,
            )
            kpis = calculate_kpi_dashboard(analysis.daily_metrics)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ReportGenerationError(f"Unable to build capacity report: {exc}") from exc

        timestamp = generated_at or datetime.now(timezone.utc)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        timestamp = timestamp.astimezone(timezone.utc)

        backlog = analysis.backlog_episodes.copy()
        if not backlog.empty and "Elevated" in backlog.columns:
            backlog = backlog.loc[backlog["Elevated"].fillna(False)].copy()
        backlog = backlog.head(self.config.max_backlog_rows).reset_index(drop=True)

        anomalies = analysis.anomaly_rows.head(self.config.max_anomaly_rows).reset_index()
        metadata: dict[str, Any] = {
            "report_title": self.config.title,
            "report_subtitle": self.config.subtitle,
            "agency_framework": self.config.agency_name,
            "classification": self.config.classification_label,
            "source_label": self.config.source_label,
            "synthetic_data": self.config.synthetic_data,
            "generated_at_utc": timestamp.isoformat(),
            "period_start": analysis.operational_summary["period_start"],
            "period_end": analysis.operational_summary["period_end"],
            "granularity": TimeGranularity.parse(analysis.config.granularity).value,
            "daily_observations": len(analysis.daily_metrics),
            "validation_status": validation.report.status,
            "validation_quality_score": validation.report.quality_score,
            "methodology_note": (
                "Counts describe operational system load and flow; they are not "
                "individual-level clinical records or official capacity ceilings."
            ),
        }
        if self.config.synthetic_data:
            metadata["synthetic_data_disclaimer"] = (
                "This report uses synthetic demonstration data and must not be "
                "represented as official HHS or CBP statistics."
            )

        return CapacityReport(
            config=self.config,
            generated_at=timestamp,
            metadata=metadata,
            executive_summary=_executive_summary(analysis, validation),
            insights=_build_insights(analysis, kpis, validation),
            kpi_table=kpis.summary_table.copy(),
            operational_table=_build_operational_table(analysis.operational_summary),
            validation_table=validation.report.to_frame(),
            backlog_table=backlog,
            anomaly_table=anomalies,
            daily_appendix=_prepare_appendix(
                analysis.daily_metrics,
                self.config.max_daily_appendix_rows,
            ),
            analysis=analysis,
            validation=validation,
        )

    def _analysis_result(self, source: pd.DataFrame | AnalysisResult) -> AnalysisResult:
        """Use an existing result or execute the configured analytical request."""
        if isinstance(source, AnalysisResult):
            return source.copy()
        if not isinstance(source, pd.DataFrame):
            raise TypeError("source must be a pandas DataFrame or AnalysisResult.")
        return run_capacity_analysis(
            source,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            granularity=self.config.granularity,
            backlog_threshold_days=self.config.backlog_threshold_days,
        )


def generate_capacity_report(
    source: pd.DataFrame | AnalysisResult,
    config: ReportConfig | None = None,
    *,
    generated_at: datetime | None = None,
) -> CapacityReport:
    """Functional entry point for complete capacity-report generation."""
    return CapacityReportGenerator(config).generate(source, generated_at=generated_at)


def export_report(
    report: CapacityReport,
    output_format: str | ReportFormat = ReportFormat.HTML,
) -> bytes:
    """Serialize a generated report to HTML or JSON bytes."""
    if not isinstance(report, CapacityReport):
        raise TypeError("report must be a CapacityReport.")
    selected = ReportFormat.parse(output_format)
    return report.to_html_bytes() if selected is ReportFormat.HTML else report.to_json_bytes()


def _html_table(frame: pd.DataFrame, empty_message: str) -> str:
    """Render a DataFrame as escaped semantic HTML."""
    if frame.empty:
        return f'<p class="empty">{html.escape(empty_message)}</p>'
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_datetime64_any_dtype(display[column]):
            display[column] = display[column].dt.strftime("%Y-%m-%d")
    return display.to_html(index=False, border=0, classes="data-table", escape=True)


def _render_html(report: CapacityReport) -> str:
    """Render a self-contained report using escaped content and inline CSS."""
    config = report.config
    metadata = report.metadata
    insight_html = "".join(
        (
            f'<article class="insight {item.severity.value}">'
            f"<h3>{html.escape(item.title)}</h3>"
            f"<p>{html.escape(item.message)}</p>"
            f"<small>{html.escape(item.evidence)}</small>"
            "</article>"
        )
        for item in report.insights
    )
    disclaimer = (
        '<div class="disclaimer"><strong>Synthetic data notice:</strong> '
        + html.escape(str(metadata.get("synthetic_data_disclaimer", "")))
        + "</div>"
        if config.synthetic_data
        else ""
    )
    daily_section = (
        "<section><h2>Daily metric appendix</h2>"
        '<p class="section-note">Newest observations appear first; the appendix '
        f"is limited to {config.max_daily_appendix_rows:,} rows.</p>"
        f"{_html_table(report.daily_appendix, 'No daily observations available.')}</section>"
        if config.include_daily_appendix
        else ""
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(config.title)}</title>
<style>
:root{{--navy:#102a43;--slate:#334e68;--blue:#2878b5;--pale:#f3f7fa;
--line:#d8e2ea;--text:#172b4d;--muted:#627d98;--red:#a61b1b;--amber:#8a5700;
--green:#176b3a;}}*{{box-sizing:border-box}}body{{margin:0;background:#edf2f6;
color:var(--text);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1240px;margin:24px auto;background:white;box-shadow:0 8px 30px #102a4320}}
header{{padding:30px 38px;background:var(--navy);color:white;border-top:6px solid var(--blue)}}
header h1{{margin:2px 0;font-size:28px}}header p{{margin:3px 0;color:#d9e8f3}}
.classification{{text-transform:uppercase;letter-spacing:.12em;font-size:11px;font-weight:700}}
.meta{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px;
padding:18px 38px;background:var(--pale);border-bottom:1px solid var(--line)}}
.meta div{{font-size:12px;color:var(--muted)}}.meta strong{{display:block;color:var(--slate)}}
section{{padding:22px 38px;border-bottom:1px solid var(--line)}}h2{{color:var(--navy);
font-size:19px;margin:0 0 12px}}h3{{font-size:15px;margin:0 0 5px}}p{{margin:6px 0}}
.insights{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px}}
.insight{{border:1px solid var(--line);border-left:5px solid var(--blue);padding:14px}}
.insight.critical{{border-left-color:var(--red)}}.insight.watch{{border-left-color:var(--amber)}}
.insight.positive{{border-left-color:var(--green)}}.insight small,.section-note{{color:var(--muted)}}
.data-table{{border-collapse:collapse;width:100%;font-size:12px;display:block;overflow-x:auto}}
.data-table th{{background:var(--slate);color:white;text-align:left;white-space:nowrap}}
.data-table th,.data-table td{{padding:8px 10px;border:1px solid var(--line)}}
.data-table tr:nth-child(even){{background:var(--pale)}}.empty{{color:var(--muted);font-style:italic}}
.disclaimer{{margin:18px 38px 0;padding:12px;border:1px solid #edc46b;background:#fff8e6}}
footer{{padding:20px 38px;color:var(--muted);font-size:12px;background:var(--pale)}}
@media print{{body{{background:white}}main{{margin:0;box-shadow:none}}section{{break-inside:avoid}}
.data-table{{display:table}}}}@media(max-width:600px){{header,section,.meta,footer{{padding:18px}}}}
</style>
</head>
<body><main>
<header><div class="classification">{html.escape(config.classification_label)}</div>
<h1>{html.escape(config.title)}</h1><p>{html.escape(config.subtitle)}</p>
<p>{html.escape(config.agency_name)}</p></header>
<div class="meta">
<div><strong>Reporting period</strong>{html.escape(str(metadata["period_start"]))} to {html.escape(str(metadata["period_end"]))}</div>
<div><strong>Generated (UTC)</strong>{html.escape(str(metadata["generated_at_utc"]))}</div>
<div><strong>Source</strong>{html.escape(config.source_label)}</div>
<div><strong>Validation</strong>{html.escape(str(metadata["validation_status"]))} · {metadata["validation_quality_score"]}/100</div>
</div>{disclaimer}
<section><h2>Executive summary</h2><p>{html.escape(report.executive_summary)}</p></section>
<section><h2>Key findings</h2><div class="insights">{insight_html}</div></section>
<section><h2>Key performance indicators</h2>{_html_table(report.kpi_table, "No KPIs available.")}</section>
<section><h2>Operational summary</h2>{_html_table(report.operational_table, "No operational summary available.")}</section>
<section><h2>Elevated backlog episodes</h2>{_html_table(report.backlog_table, "No elevated backlog episodes met the configured threshold.")}</section>
<section><h2>Data quality and validation</h2>{_html_table(report.validation_table, "No validation findings.")}</section>
<section><h2>Logical anomaly rows</h2>{_html_table(report.anomaly_table, "No logical anomaly rows were identified.")}</section>
{daily_section}
<footer>{html.escape(str(metadata["methodology_note"]))}<br>
Generated by the HHS UAC Capacity Analytics reporting pipeline.</footer>
</main></body></html>"""
