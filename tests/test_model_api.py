"""Read-only API contract tests for approved forecasting artifacts."""

from __future__ import annotations

import unittest
import warnings

warnings.filterwarnings(
    "ignore",
    message="Using `httpx` with `starlette.testclient` is deprecated",
)

from starlette.testclient import TestClient  # noqa: E402

from backend.api import app  # noqa: E402


class ModelAPITests(unittest.TestCase):
    """Verify versioned model metadata and forecast response schemas."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.__exit__(None, None, None)

    def test_model_metadata_and_provenance(self) -> None:
        model = self.client.get("/api/v1/model")
        provenance = self.client.get("/api/v1/model/provenance")
        self.assertEqual(model.status_code, 200)
        self.assertEqual(provenance.status_code, 200)
        self.assertEqual(model.json()["promotion_status"], "promote")
        self.assertTrue(provenance.json()["leakage_audit"]["passed"])
        self.assertEqual(len(provenance.json()["fingerprints"]["source_sha256"]), 64)

    def test_metrics_and_monitoring_are_available(self) -> None:
        metrics = self.client.get("/api/v1/model/metrics")
        monitoring = self.client.get("/api/v1/model/monitoring")
        self.assertEqual(metrics.status_code, 200)
        self.assertIn("persistence", metrics.json()["comparison"]["models"])
        self.assertEqual(monitoring.status_code, 200)
        self.assertIn(monitoring.json()["model_status"], {"approved", "degraded", "fallback"})

    def test_forecast_contract_and_date_validation(self) -> None:
        response = self.client.get("/api/v1/forecast?as_of=2025-12-14")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["forecast"]["horizon_days"], 7)
        self.assertLessEqual(
            payload["forecast"]["lower_interval"],
            payload["forecast"]["upper_interval"],
        )
        self.assertEqual(
            self.client.get("/api/v1/forecast?as_of=not-a-date").status_code,
            422,
        )
        self.assertEqual(
            self.client.get("/api/v1/forecast?as_of=2000-01-01").status_code,
            404,
        )


if __name__ == "__main__":
    unittest.main()
