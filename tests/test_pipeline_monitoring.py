"""Integration checks for orchestration and offline model monitoring."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.forecasting import FINGERPRINT_ALGORITHM
from src.monitoring import MonitoringConfig, evaluate_monitoring
from src.pipeline import PipelineConfig, run_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
        self.assertTrue(result.stages["leakage_audit_passed"])

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
