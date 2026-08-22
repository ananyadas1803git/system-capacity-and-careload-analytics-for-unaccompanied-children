"""Build reproducible raw and processed datasets for the HHS UAC project.

The official source is copied byte-for-byte into ``data/raw``.  All derived
artifacts are written atomically so a failed run cannot leave partially written
CSV, JSON, or Parquet files behind.

Usage::

    python generate_sample_data.py
    python generate_sample_data.py --source /path/to/source.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from app_utils import (
    CBP_COLUMN,
    DATE_COLUMN,
    DISCHARGE_COLUMN,
    HHS_COLUMN,
    INTAKE_COLUMN,
    TRANSFER_COLUMN,
    compute_capacity_metrics,
    generate_mock_data,
)
from src.feature_engineering import build_feature_matrix
from src.preprocessor import preprocess_data
from src.validation import validate_capacity_data


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = Path.home() / "Downloads" / "HHS_Unaccompanied_Alien_Children_Program.csv"
DEFAULT_DATA_DIRECTORY = PROJECT_ROOT / "data"

OFFICIAL_RAW_NAME = "HHS_Unaccompanied_Alien_Children_Program.csv"
SYNTHETIC_RAW_NAME = "uac_capacity_synthetic_2023_2025.csv"
CLEANED_NAME = "uac_capacity_cleaned_daily.csv"
METRICS_NAME = "uac_capacity_metrics_daily.csv"
FEATURES_NAME = "uac_capacity_ml_features.parquet"
PREPROCESSING_REPORT_NAME = "preprocessing_report.json"
VALIDATION_REPORT_NAME = "validation_report.json"
SOURCE_EXPORT_COLUMNS = [
    DATE_COLUMN,
    INTAKE_COLUMN,
    CBP_COLUMN,
    TRANSFER_COLUMN,
    HHS_COLUMN,
    DISCHARGE_COLUMN,
]


class DataArtifactGenerationError(RuntimeError):
    """Raised when project data artifacts cannot be generated safely."""


@contextmanager
def _atomic_target(target: Path) -> Iterator[Path]:
    """Yield a sibling temporary path and replace the target after success."""
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        temporary.unlink(missing_ok=True)
        yield temporary
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    """Calculate a source or output file SHA-256 checksum in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    """Normalize common scientific values for strict JSON serialization."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_csv(frame: pd.DataFrame, target: Path, *, index: bool) -> None:
    """Write a deterministic UTF-8 CSV through an atomic temporary file."""
    export = frame.copy()
    export.attrs.clear()
    with _atomic_target(target) as temporary:
        export.to_csv(
            temporary,
            index=index,
            encoding="utf-8",
            date_format="%Y-%m-%d",
            lineterminator="\n",
        )


def _source_first(frame: pd.DataFrame) -> pd.DataFrame:
    """Place documented source fields before quality and derived columns."""
    first = [column for column in SOURCE_EXPORT_COLUMNS if column in frame.columns]
    remaining = [column for column in frame.columns if column not in first]
    return frame[first + remaining].copy()


def _write_json(payload: dict[str, Any], target: Path) -> None:
    """Write strict, human-readable JSON through an atomic temporary file."""
    with _atomic_target(target) as temporary:
        temporary.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
                default=_json_default,
            )
            + "\n",
            encoding="utf-8",
        )


def _copy_official_source(source: Path, target: Path) -> str:
    """Copy the official source and verify that its bytes remain unchanged."""
    source_checksum = _sha256(source)
    with _atomic_target(target) as temporary:
        shutil.copyfile(source, temporary)
    target_checksum = _sha256(target)
    if target_checksum != source_checksum:
        raise DataArtifactGenerationError(
            "The raw source checksum changed during the copy operation."
        )
    return source_checksum


def _data_readme(
    *,
    source_checksum: str,
    cleaned_rows: int,
    feature_columns: int,
    reporting_start: str,
    reporting_end: str,
) -> str:
    """Return lineage and regeneration documentation for the data directory."""
    return f"""# Project data

This directory contains aggregate operational counts used by the HHS UAC
capacity analytics application. It contains no individual-level child records.

## Raw data

- `raw/{OFFICIAL_RAW_NAME}` is an unchanged copy of the supplied HHS CSV.
  SHA-256: `{source_checksum}`
- `raw/{SYNTHETIC_RAW_NAME}` is deterministic demonstration data generated by
  `app_utils.generate_mock_data()` for 2023-01-01 through 2025-12-31. It must
  not be represented as official HHS or CBP data.

Raw files are immutable inputs. Corrections and imputations belong only in
`processed/` outputs.

## Processed data

- `processed/{CLEANED_NAME}` — complete daily series after schema normalization,
  date repair, stock/flow imputation, and row-level quality flags.
- `processed/{METRICS_NAME}` — cleaned data plus system load, net intake, growth,
  rolling averages, discharge offset ratio, and backlog streak.
- `processed/{FEATURES_NAME}` — model-ready source, calendar, lag, rolling,
  momentum, quality, and future-target fields stored as Parquet.
- `processed/{PREPROCESSING_REPORT_NAME}` — structured audit of repairs applied
  while producing the cleaned dataset.
- `processed/{VALIDATION_REPORT_NAME}` — non-mutating audits of both the raw
  source and processed analytical metrics.

Official processed coverage: **{reporting_start} through {reporting_end}**
({cleaned_rows:,} daily rows). The ML artifact contains {feature_columns:,}
columns, including source signals, engineered features, and targets.

## Required source schema

All count fields use units of children; `Date` is a reporting date.

1. `Date`
2. `Children apprehended and placed in CBP custody`
3. `Children in CBP custody`
4. `Children transferred out of CBP custody`
5. `Children in HHS Care`
6. `Children discharged from HHS Care`

## Regeneration

From the repository root:

```bash
python generate_sample_data.py
```

Use `--source` to select a different official CSV and `--data-dir` only when an
alternate artifact root is required. Review both JSON audits before relying on
the processed data for operational decisions.
"""


def generate_data_artifacts(source: Path, data_directory: Path) -> dict[str, Any]:
    """Generate all raw, processed, feature, and data-lineage artifacts."""
    source = source.expanduser().resolve()
    data_directory = data_directory.expanduser().resolve()
    if not source.is_file():
        raise DataArtifactGenerationError(f"Official source CSV not found: {source}")
    if source.suffix.casefold() != ".csv":
        raise DataArtifactGenerationError("Official source must be a CSV file.")

    raw_directory = data_directory / "raw"
    processed_directory = data_directory / "processed"
    raw_directory.mkdir(parents=True, exist_ok=True)
    processed_directory.mkdir(parents=True, exist_ok=True)

    official_target = raw_directory / OFFICIAL_RAW_NAME
    source_checksum = _copy_official_source(source, official_target)

    synthetic = generate_mock_data()
    _write_csv(synthetic, raw_directory / SYNTHETIC_RAW_NAME, index=False)

    preprocessed = preprocess_data(official_target)
    cleaned = preprocessed.data.copy()
    metrics = compute_capacity_metrics(cleaned)
    feature_result = build_feature_matrix(metrics)
    feature_frame = feature_result.frame.copy()
    feature_frame.attrs.clear()

    _write_csv(
        _source_first(cleaned.reset_index()),
        processed_directory / CLEANED_NAME,
        index=False,
    )
    _write_csv(
        _source_first(metrics.reset_index()),
        processed_directory / METRICS_NAME,
        index=False,
    )
    with _atomic_target(processed_directory / FEATURES_NAME) as temporary:
        feature_frame.to_parquet(temporary, index=True, engine="pyarrow")

    preprocessing_payload = {
        "artifact": CLEANED_NAME,
        "source": f"raw/{OFFICIAL_RAW_NAME}",
        "source_sha256": source_checksum,
        "report": preprocessed.report.to_dict(),
    }
    _write_json(
        preprocessing_payload,
        processed_directory / PREPROCESSING_REPORT_NAME,
    )

    raw_frame = pd.read_csv(official_target, dtype=str, encoding="utf-8-sig")
    raw_validation = validate_capacity_data(raw_frame)
    processed_validation = validate_capacity_data(metrics)
    validation_payload = {
        "source": f"raw/{OFFICIAL_RAW_NAME}",
        "raw_source_audit": raw_validation.report.to_dict(),
        "processed_metrics_audit": processed_validation.report.to_dict(),
    }
    _write_json(validation_payload, processed_directory / VALIDATION_REPORT_NAME)

    readme = _data_readme(
        source_checksum=source_checksum,
        cleaned_rows=len(cleaned),
        feature_columns=len(feature_frame.columns),
        reporting_start=cleaned.index.min().date().isoformat(),
        reporting_end=cleaned.index.max().date().isoformat(),
    )
    with _atomic_target(data_directory / "README.md") as temporary:
        temporary.write_text(readme, encoding="utf-8")

    return {
        "official_source_sha256": source_checksum,
        "official_source_rows": len(raw_frame),
        "synthetic_rows": len(synthetic),
        "cleaned_rows": len(cleaned),
        "metrics_rows": len(metrics),
        "feature_rows": len(feature_frame),
        "feature_columns": len(feature_frame.columns),
        "raw_validation_status": raw_validation.report.status,
        "processed_validation_status": processed_validation.report.status,
        "data_directory": str(data_directory),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line source and output locations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Official source CSV (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIRECTORY,
        help=f"Data artifact directory (default: {DEFAULT_DATA_DIRECTORY})",
    )
    return parser.parse_args()


def main() -> None:
    """Generate artifacts and print a compact machine-readable summary."""
    arguments = parse_args()
    try:
        summary = generate_data_artifacts(arguments.source, arguments.data_dir)
    except (
        DataArtifactGenerationError,
        OSError,
        TypeError,
        ValueError,
        ImportError,
    ) as exc:
        raise SystemExit(f"Data artifact generation failed: {exc}") from exc
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
