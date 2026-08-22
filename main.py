"""Unified command-line entry point for the HHS UAC analytics project.

Running ``python main.py`` launches the Streamlit dashboard.  Subcommands expose
the same backend services for API hosting, repeatable data generation, dataset
validation, headless analysis, and report export.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from app_utils import DataValidationError, generate_mock_data
from backend.analytics import AnalyticsError, run_capacity_analysis
from backend.utils import dataframe_records, json_safe
from generate_sample_data import (
    DEFAULT_DATA_DIRECTORY,
    DEFAULT_SOURCE,
    DataArtifactGenerationError,
    generate_data_artifacts,
)
from src.report_generator import (
    ReportConfig,
    ReportFormat,
    ReportGenerationError,
    export_report,
    generate_capacity_report,
)
from src.validation import (
    DatasetValidationError,
    ValidationConfig,
    validate_capacity_data,
)


PROJECT_ROOT = Path(__file__).resolve().parent
STREAMLIT_ENTRYPOINT = PROJECT_ROOT / "app" / "streamlit_app.py"
DEFAULT_PROJECT_SOURCE = (
    PROJECT_ROOT / "data" / "raw" / "HHS_Unaccompanied_Alien_Children_Program.csv"
)
DEFAULT_REPORT_DIRECTORY = PROJECT_ROOT / "reports"
APPLICATION_VERSION = "1.0.0"
MAX_CSV_BYTES = 50 * 1024 * 1024
MAX_CSV_ROWS = 1_000_000

EXIT_SUCCESS = 0
EXIT_RUNTIME_ERROR = 1
EXIT_VALIDATION_FAILED = 2
EXIT_INTERRUPTED = 130


class CommandError(RuntimeError):
    """Raised for a safe, user-facing CLI command failure."""


def _port(value: str) -> int:
    """Parse a valid TCP port for argparse."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 1 <= parsed <= 65_535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return parsed


def _positive_integer(value: str) -> int:
    """Parse a positive integer for argparse."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _existing_source_path(value: str | Path | None) -> Path:
    """Resolve a selected source or the project/raw then Downloads fallback."""
    if value is not None:
        candidate = Path(value).expanduser().resolve()
    elif DEFAULT_PROJECT_SOURCE.is_file():
        candidate = DEFAULT_PROJECT_SOURCE.resolve()
    else:
        candidate = DEFAULT_SOURCE.expanduser().resolve()
    if not candidate.is_file():
        raise CommandError(f"CSV source not found: {candidate}")
    if candidate.suffix.casefold() != ".csv":
        raise CommandError(f"Source must be a CSV file: {candidate}")
    return candidate


def _read_csv_source(path: Path) -> pd.DataFrame:
    """Read a bounded CSV as strings for transparent downstream validation."""
    try:
        byte_count = path.stat().st_size
    except OSError as exc:
        raise CommandError(f"Unable to inspect CSV source '{path}': {exc}") from exc
    if byte_count > MAX_CSV_BYTES:
        raise CommandError(
            f"CSV contains {byte_count:,} bytes; limit is {MAX_CSV_BYTES:,}."
        )
    try:
        frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    except (
        OSError,
        UnicodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as exc:
        raise CommandError(f"Unable to read CSV source '{path}': {exc}") from exc
    if len(frame) > MAX_CSV_ROWS:
        raise CommandError(
            f"CSV contains {len(frame):,} rows; limit is {MAX_CSV_ROWS:,}."
        )
    return frame


def _resolve_source(arguments: argparse.Namespace) -> tuple[pd.DataFrame, str, bool]:
    """Return a source DataFrame, display label, and synthetic-data flag."""
    if getattr(arguments, "mock", False):
        return generate_mock_data(), "Synthetic 2023–2025 demonstration data", True
    path = _existing_source_path(getattr(arguments, "source", None))
    return _read_csv_source(path), path.name, False


def _atomic_write(
    target: Path,
    content: bytes,
    *,
    force: bool,
) -> None:
    """Write bytes atomically while protecting existing artifacts by default."""
    target = target.expanduser().resolve()
    if target.exists() and not force:
        raise CommandError(
            f"Output already exists: {target}. Pass --force to replace it."
        )
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.unlink(missing_ok=True)
        temporary.write_bytes(content)
        os.replace(temporary, target)
    except OSError as exc:
        raise CommandError(f"Unable to write output '{target}': {exc}") from exc
    finally:
        if "temporary" in locals():
            temporary.unlink(missing_ok=True)


def _json_bytes(payload: Any) -> bytes:
    """Serialize strict, readable JSON using the backend's scientific converter."""
    return (
        json.dumps(
            json_safe(payload),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _emit_json(
    payload: Any,
    output: Path | None,
    *,
    force: bool,
) -> None:
    """Write JSON to a file or standard output."""
    content = _json_bytes(payload)
    if output is None:
        sys.stdout.write(content.decode("utf-8"))
        return
    _atomic_write(output, content, force=force)
    print(f"Wrote {output.expanduser().resolve()}")


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    """Add mutually exclusive official-source and synthetic-source options."""
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--source",
        type=Path,
        help=(
            "Input CSV. Defaults to data/raw/HHS_Unaccompanied_Alien_Children_Program.csv "
            "and then the Downloads copy."
        ),
    )
    source.add_argument(
        "--mock",
        action="store_true",
        help="Use deterministic synthetic 2023–2025 demonstration data.",
    )


def _add_analysis_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common reporting-period and analytical configuration options."""
    parser.add_argument("--start-date", help="Inclusive reporting start (YYYY-MM-DD).")
    parser.add_argument("--end-date", help="Inclusive reporting end (YYYY-MM-DD).")
    parser.add_argument(
        "--granularity",
        choices=["Daily", "Weekly", "Monthly"],
        default="Daily",
        help="Presentation aggregation; stock endpoints and summed flows are preserved.",
    )
    parser.add_argument(
        "--backlog-threshold",
        type=_positive_integer,
        default=3,
        metavar="DAYS",
        help="Minimum streak length classified as an elevated backlog episode.",
    )


def command_dashboard(arguments: argparse.Namespace) -> int:
    """Launch the Streamlit multipage dashboard in a child process."""
    if importlib.util.find_spec("streamlit") is None:
        raise CommandError("Streamlit is not installed in the active Python environment.")
    if not STREAMLIT_ENTRYPOINT.is_file():
        raise CommandError(f"Streamlit entry point not found: {STREAMLIT_ENTRYPOINT}")

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(STREAMLIT_ENTRYPOINT),
        "--server.address",
        arguments.host,
        "--server.port",
        str(arguments.port),
        "--server.headless",
        str(arguments.headless).lower(),
        "--browser.gatherUsageStats",
        "false",
    ]
    passthrough = list(arguments.streamlit_args)
    if passthrough[:1] == ["--"]:
        passthrough = passthrough[1:]
    command.extend(passthrough)
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return int(completed.returncode)


def command_api(arguments: argparse.Namespace) -> int:
    """Launch the Starlette API with Uvicorn."""
    if arguments.reload and arguments.workers != 1:
        raise CommandError("--reload requires --workers 1.")
    try:
        import uvicorn
    except ImportError as exc:
        raise CommandError("Uvicorn is not installed in the active environment.") from exc

    uvicorn.run(
        "backend.api:app",
        host=arguments.host,
        port=arguments.port,
        log_level=arguments.log_level,
        reload=arguments.reload,
        workers=arguments.workers,
    )
    return EXIT_SUCCESS


def command_generate_data(arguments: argparse.Namespace) -> int:
    """Regenerate raw, processed, feature, and audit artifacts."""
    summary = generate_data_artifacts(arguments.source, arguments.data_dir)
    _emit_json(summary, arguments.output, force=arguments.force)
    return EXIT_SUCCESS


def command_validate(arguments: argparse.Namespace) -> int:
    """Audit a CSV without mutating or repairing its contents."""
    data, source_label, synthetic = _resolve_source(arguments)
    config = ValidationConfig(
        require_complete_daily_series=not arguments.allow_date_gaps,
        allow_fractional_counts=arguments.allow_fractional_counts,
        check_derived_metrics=not arguments.skip_derived_metrics,
        check_stock_flow_balance=arguments.check_stock_flow_balance,
        strict_warnings=arguments.strict_warnings,
        maximum_flagged_rows=arguments.max_flagged_rows,
    )
    result = validate_capacity_data(data, config)
    payload = {
        "source": source_label,
        "synthetic_data": synthetic,
        "report": result.report.to_dict(),
        "flagged_rows": dataframe_records(result.flagged_rows),
    }
    _emit_json(payload, arguments.output, force=arguments.force)

    failed = not result.report.is_valid or (
        arguments.strict_warnings and result.report.warning_count > 0
    )
    return EXIT_VALIDATION_FAILED if failed else EXIT_SUCCESS


def command_analyze(arguments: argparse.Namespace) -> int:
    """Run headless capacity analytics and emit a compact JSON summary."""
    data, source_label, synthetic = _resolve_source(arguments)
    result = run_capacity_analysis(
        data,
        start_date=arguments.start_date,
        end_date=arguments.end_date,
        granularity=arguments.granularity,
        backlog_threshold_days=arguments.backlog_threshold,
    )
    report = result.validation_report
    payload = {
        "source": source_label,
        "synthetic_data": synthetic,
        "config": {
            "start_date": result.config.start_date,
            "end_date": result.config.end_date,
            "granularity": arguments.granularity,
            "backlog_threshold_days": arguments.backlog_threshold,
        },
        "kpis": result.kpis,
        "operational_summary": result.operational_summary,
        "backlog_episodes": dataframe_records(result.backlog_episodes),
        "validation": (
            {
                "errors": report.error_count,
                "warnings": report.warning_count,
                "findings": dataframe_records(report.to_frame()),
            }
            if report is not None
            else {"errors": 0, "warnings": 0, "findings": []}
        ),
    }
    if arguments.include_anomalies:
        payload["anomaly_rows"] = dataframe_records(result.anomaly_rows)
    _emit_json(payload, arguments.output, force=arguments.force)
    return EXIT_SUCCESS


def command_report(arguments: argparse.Namespace) -> int:
    """Generate a standalone HTML or strict JSON analytical report."""
    data, source_label, synthetic = _resolve_source(arguments)
    config = ReportConfig(
        source_label=source_label,
        synthetic_data=synthetic,
        start_date=arguments.start_date,
        end_date=arguments.end_date,
        granularity=arguments.granularity,
        backlog_threshold_days=arguments.backlog_threshold,
        include_daily_appendix=arguments.include_daily_appendix,
        max_daily_appendix_rows=arguments.max_daily_rows,
    )
    report = generate_capacity_report(data, config)
    selected_format = ReportFormat.parse(arguments.format)
    output = arguments.output
    if output is None:
        output = DEFAULT_REPORT_DIRECTORY / (
            f"{report.filename_stem}.{selected_format.value}"
        )
    _atomic_write(
        output,
        export_report(report, selected_format),
        force=arguments.force,
    )
    print(f"Wrote {output.expanduser().resolve()}")
    return EXIT_SUCCESS


def build_parser() -> argparse.ArgumentParser:
    """Construct the complete command-line parser."""
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="HHS UAC System Capacity & Care Load Analytics",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {APPLICATION_VERSION}",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    dashboard = subparsers.add_parser(
        "dashboard",
        help="Launch the Streamlit multipage dashboard (default).",
    )
    dashboard.add_argument("--host", default="127.0.0.1")
    dashboard.add_argument("--port", type=_port, default=8501)
    dashboard.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    dashboard.add_argument(
        "streamlit_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments passed to Streamlit after '--'.",
    )
    dashboard.set_defaults(handler=command_dashboard)

    api = subparsers.add_parser("api", help="Launch the Starlette/Uvicorn API.")
    api.add_argument("--host", default="127.0.0.1")
    api.add_argument("--port", type=_port, default=8000)
    api.add_argument(
        "--log-level",
        choices=["critical", "error", "warning", "info", "debug"],
        default="info",
    )
    api.add_argument("--reload", action="store_true")
    api.add_argument("--workers", type=_positive_integer, default=1)
    api.set_defaults(handler=command_api)

    generate = subparsers.add_parser(
        "generate-data",
        help="Populate data/raw and data/processed reproducibly.",
    )
    generate.add_argument(
        "--source",
        type=Path,
        default=(
            DEFAULT_PROJECT_SOURCE
            if DEFAULT_PROJECT_SOURCE.is_file()
            else DEFAULT_SOURCE
        ),
    )
    generate.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIRECTORY)
    generate.add_argument("--output", type=Path, help="Optional JSON summary path.")
    generate.add_argument("--force", action="store_true", help="Replace summary output.")
    generate.set_defaults(handler=command_generate_data)

    validate = subparsers.add_parser(
        "validate",
        help="Audit a raw, cleaned, or metric-enriched CSV without repairing it.",
    )
    _add_source_arguments(validate)
    validate.add_argument("--output", type=Path, help="Write results as JSON.")
    validate.add_argument("--force", action="store_true")
    validate.add_argument("--strict-warnings", action="store_true")
    validate.add_argument("--allow-date-gaps", action="store_true")
    validate.add_argument("--allow-fractional-counts", action="store_true")
    validate.add_argument("--skip-derived-metrics", action="store_true")
    validate.add_argument("--check-stock-flow-balance", action="store_true")
    validate.add_argument(
        "--max-flagged-rows",
        type=_positive_integer,
        default=1_000,
    )
    validate.set_defaults(handler=command_validate)

    analyze = subparsers.add_parser(
        "analyze",
        help="Calculate KPIs and operational summaries without starting a server.",
    )
    _add_source_arguments(analyze)
    _add_analysis_arguments(analyze)
    analyze.add_argument("--include-anomalies", action="store_true")
    analyze.add_argument("--output", type=Path, help="Write results as JSON.")
    analyze.add_argument("--force", action="store_true")
    analyze.set_defaults(handler=command_analyze)

    report = subparsers.add_parser(
        "report",
        help="Generate a standalone HTML or JSON analytical report.",
    )
    _add_source_arguments(report)
    _add_analysis_arguments(report)
    report.add_argument("--format", choices=["html", "json"], default="html")
    report.add_argument("--output", type=Path)
    report.add_argument("--force", action="store_true")
    report.add_argument(
        "--include-daily-appendix",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    report.add_argument(
        "--max-daily-rows",
        type=_positive_integer,
        default=500,
    )
    report.set_defaults(handler=command_report)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, execute one command, and return a process exit code."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        arguments = ["dashboard"]
    parser = build_parser()
    parsed = parser.parse_args(arguments)
    handler: Callable[[argparse.Namespace], int] | None = getattr(
        parsed,
        "handler",
        None,
    )
    if handler is None:
        parser.print_help()
        return EXIT_RUNTIME_ERROR

    try:
        return int(handler(parsed))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return EXIT_INTERRUPTED
    except (
        AnalyticsError,
        CommandError,
        DataArtifactGenerationError,
        DataValidationError,
        DatasetValidationError,
        ImportError,
        ReportGenerationError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
