"""Integration checks for orchestration and offline model monitoring."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.forecasting import (
    FINGERPRINT_ALGORITHM,
    PREPARED_DATASET_CONTRACT_VERSION,
    TARGET_ABSOLUTE,
    TARGET_CHANGE,
)
from src.monitoring import MonitoringConfig, evaluate_monitoring
from src.pipeline import (
    PipelineConfig,
    PipelineError,
    _assert_prepared_frames_equivalent,
    _verify_file_sha256,
    run_pipeline,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_prepared_frame() -> pd.DataFrame:
    """Return a small representative canonical prepared forecasting frame."""

    index = pd.date_range("2025-01-01", periods=4, freq="D", name="Date")
    return pd.DataFrame(
        {
            "derived_rolling_feature": [1.25, 2.5, float("nan"), 4.75],
            "quality_flag": [False, True, False, False],
            "regime": ["normal", "backlog", "normal", "normal"],
            TARGET_ABSOLUTE: [1_010.0, 1_020.0, 1_030.0, 1_040.0],
            TARGET_CHANGE: [10.0, 10.0, 10.0, 10.0],
            "target_date": index + pd.Timedelta(days=7),
        },
        index=index,
    )


class PipelineMonitoringTests(unittest.TestCase):
    """Exercise deterministic quick mode, artifact verification, and fallback."""

    def test_quick_pipeline_writes_valid_reproducible_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = run_pipeline(PipelineConfig(quick=True, output_dir=root, overwrite=True))
            first_summary = json.loads(Path(first.artifacts["summary"]).read_text(encoding="utf-8"))
            first_predictions = pd.read_csv(first.artifacts["predictions"])
            second = run_pipeline(PipelineConfig(quick=True, output_dir=root, overwrite=True))
            second_summary = json.loads(
                Path(second.artifacts["summary"]).read_text(encoding="utf-8")
            )
            second_predictions = pd.read_csv(second.artifacts["predictions"])

            self.assertEqual(first.mode, "quick")
            self.assertTrue(all(first_summary["leakage_checks"].values()))
            self.assertEqual(first_summary, second_summary)
            pd.testing.assert_frame_equal(first_predictions, second_predictions)
            self.assertTrue(Path(second.artifacts["model"]).is_file())

    def test_default_pipeline_verifies_frozen_artifacts(self) -> None:
        result = run_pipeline(PipelineConfig())
        self.assertEqual(result.mode, "approved-artifact-verification")
        self.assertEqual(result.stages["model_action"], "loaded_and_verified_approved_artifacts")
        self.assertEqual(result.stages["fingerprint_algorithm"], FINGERPRINT_ALGORITHM)
        self.assertEqual(
            result.stages["verification_contract_version"],
            PREPARED_DATASET_CONTRACT_VERSION,
        )
        self.assertTrue(result.stages["leakage_audit_passed"])

    def test_tiny_derived_floating_point_differences_pass(self) -> None:
        canonical = make_prepared_frame()
        regenerated = canonical.copy()
        regenerated.loc[regenerated.index[0], "derived_rolling_feature"] += 1e-12
        _assert_prepared_frames_equivalent(regenerated, canonical)

    def test_meaningful_numeric_changes_fail(self) -> None:
        canonical = make_prepared_frame()
        regenerated = canonical.copy()
        regenerated.loc[regenerated.index[0], "derived_rolling_feature"] += 1e-4
        with self.assertRaisesRegex(PipelineError, "Prepared values changed"):
            _assert_prepared_frames_equivalent(regenerated, canonical)

    def test_missingness_changes_fail(self) -> None:
        canonical = make_prepared_frame()
        regenerated = canonical.copy()
        regenerated.loc[regenerated.index[1], "derived_rolling_feature"] = float("nan")
        with self.assertRaisesRegex(PipelineError, "missing-value locations changed"):
            _assert_prepared_frames_equivalent(regenerated, canonical)

    def test_column_order_and_schema_changes_fail(self) -> None:
        canonical = make_prepared_frame()
        reordered = canonical.loc[:, list(reversed(canonical.columns))]
        schema_changed = canonical.copy()
        schema_changed["derived_rolling_feature"] = schema_changed[
            "derived_rolling_feature"
        ].astype("string")
        with self.subTest(change="column_order"):
            with self.assertRaisesRegex(PipelineError, "column order or names changed"):
                _assert_prepared_frames_equivalent(reordered, canonical)
        with self.subTest(change="schema"):
            with self.assertRaisesRegex(PipelineError, "schema fingerprint changed"):
                _assert_prepared_frames_equivalent(schema_changed, canonical)

    def test_row_and_date_changes_fail(self) -> None:
        canonical = make_prepared_frame()
        row_removed = canonical.iloc[:-1]
        date_changed = canonical.copy()
        date_changed.index = date_changed.index + pd.Timedelta(days=1)
        with self.subTest(change="row"):
            with self.assertRaisesRegex(PipelineError, "row or column count changed"):
                _assert_prepared_frames_equivalent(row_removed, canonical)
        with self.subTest(change="date"):
            with self.assertRaisesRegex(PipelineError, "row dates or order changed"):
                _assert_prepared_frames_equivalent(date_changed, canonical)

    def test_target_changes_fail(self) -> None:
        canonical = make_prepared_frame()
        regenerated = canonical.copy()
        regenerated.loc[regenerated.index[0], TARGET_CHANGE] += 1.0
        with self.assertRaisesRegex(PipelineError, "target_change_7d"):
            _assert_prepared_frames_equivalent(regenerated, canonical)

    def test_raw_source_hash_changes_fail_with_expected_and_actual_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.csv"
            original = b"Date,value\n2025-01-01,1\n"
            source.write_bytes(original)
            expected = hashlib.sha256(original).hexdigest()
            source.write_bytes(b"Date,value\n2025-01-01,2\n")
            actual = hashlib.sha256(source.read_bytes()).hexdigest()
            with self.assertRaisesRegex(
                PipelineError,
                f"expected='{expected}'; actual='{actual}'",
            ):
                _verify_file_sha256(source, expected, "Raw source")

    def test_monitoring_is_finite_and_uses_approved_champion(self) -> None:
        result = evaluate_monitoring(
            MonitoringConfig(
                artifact_root=PROJECT_ROOT / "output" / "forecasting",
                write_artifacts=False,
            )
        )
        self.assertEqual(result.model_status, "approved")
        self.assertEqual(result.active_model, result.configured_champion)
        self.assertGreater(result.metrics["rolling_window"], 0)
        self.assertGreaterEqual(result.metrics["rolling_mae"], 0)

    def test_missing_component_triggers_explicit_persistence_fallback(self) -> None:
        with patch(
            "src.monitoring._component_artifacts_available",
            return_value=(False, "test component unavailable"),
        ):
            result = evaluate_monitoring(
                MonitoringConfig(
                    artifact_root=PROJECT_ROOT / "output" / "forecasting",
                    write_artifacts=False,
                )
            )
        self.assertEqual(result.active_model, "persistence")
        self.assertEqual(result.model_status, "fallback")
        self.assertEqual(result.event["event_type"], "fallback")


if __name__ == "__main__":
    unittest.main()
