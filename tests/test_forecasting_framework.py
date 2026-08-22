"""Fast invariants for the multi-model forecasting research framework."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from app_utils import (
    CBP_COLUMN,
    DISCHARGE_COLUMN,
    HHS_COLUMN,
    INTAKE_COLUMN,
    NET_INTAKE_COLUMN,
    TOTAL_LOAD_COLUMN,
    TRANSFER_COLUMN,
)
from src.forecasting import (
    CURRENT_LOAD,
    FINGERPRINT_ALGORITHM,
    FINGERPRINT_FLOAT_DECIMAL_PLACES,
    PREDICTION_COLUMNS,
    TARGET_ABSOLUTE,
    TARGET_CHANGE,
    FeatureProcessor,
    ForecastConfig,
    ForecastingFrameworkError,
    build_oof_residuals,
    chronological_partitions,
    construct_change_target,
    drift_forecast,
    expanding_window_folds,
    optimize_ensemble_weights,
    order_prediction_intervals,
    persistence_forecast,
    prepare_forecast_dataset,
    reconstruct_absolute_forecast,
    regression_metrics,
    validate_prediction_schema,
    _frame_fingerprint,
    _schema_fingerprint,
)


def make_source(rows: int = 520) -> pd.DataFrame:
    """Return deterministic daily aggregate data with quality fields."""

    index = pd.date_range("2023-01-01", periods=rows, freq="D", name="Date")
    time = np.arange(rows, dtype=float)
    cbp = 400 + 0.2 * time + 10 * np.sin(time / 8)
    hhs = 4_000 + 0.7 * time + 20 * np.sin(time / 15)
    intake = 100 + time % 13
    transfers = 95 + time % 11
    discharges = 93 + time % 9
    frame = pd.DataFrame(index=index)
    frame[CBP_COLUMN] = cbp
    frame[HHS_COLUMN] = hhs
    frame[INTAKE_COLUMN] = intake
    frame[TRANSFER_COLUMN] = transfers
    frame[DISCHARGE_COLUMN] = discharges
    frame[NET_INTAKE_COLUMN] = transfers - discharges
    frame[TOTAL_LOAD_COLUMN] = cbp + hhs
    frame["Is Imputed Date"] = False
    frame["Anomaly_Any"] = False
    return frame


def make_config(root: Path | None = None) -> ForecastConfig:
    base = root or Path("/private/tmp/hhs-forecast-tests")
    return ForecastConfig(
        feature_path=base / "features.parquet",
        raw_path=base / "source.csv",
        provenance_path=base / "preprocessing.json",
        output_dir=base / "forecasting",
        cv_splits=4,
        cv_test_size=28,
        overwrite=True,
    )


class ForecastingFrameworkTests(unittest.TestCase):
    """Validate targets, leakage controls, splits, metrics, and schemas."""

    def test_frame_fingerprint_ignores_datetime_storage_units(self) -> None:
        index_ns = pd.date_range("2025-01-01", periods=3, freq="D", name="Date")
        index_us = index_ns.as_unit("us")
        first = pd.DataFrame(
            {
                "target_date": index_ns + pd.Timedelta(days=7),
            },
            index=index_ns,
        )
        second = pd.DataFrame(
            {
                "target_date": (index_us + pd.Timedelta(days=7)).as_unit("us"),
            },
            index=index_us,
        )
        self.assertEqual(_frame_fingerprint(first), _frame_fingerprint(second))
        self.assertEqual(_schema_fingerprint(first), _schema_fingerprint(second))

    def test_frame_fingerprint_ignores_equivalent_numeric_dtypes(self) -> None:
        index = pd.date_range("2025-01-01", periods=3, name="Date")
        integers = pd.DataFrame({"value": pd.Series([1, 2, 3], index=index)}, index=index)
        floats = pd.DataFrame(
            {"value": pd.Series([1.0, 2.0, 3.0], index=index, dtype="float32")},
            index=index,
        )
        self.assertEqual(_frame_fingerprint(integers), _frame_fingerprint(floats))
        self.assertEqual(_schema_fingerprint(integers), _schema_fingerprint(floats))

    def test_frame_fingerprint_ignores_platform_scale_numeric_noise(self) -> None:
        index = pd.date_range("2025-01-01", periods=3, name="Date")
        baseline = pd.DataFrame({"value": [1.25, -0.0, 8_000.0]}, index=index)
        noisy = baseline.copy()
        noisy.loc[index[0], "value"] += 1e-12
        noisy.loc[index[1], "value"] = 0.0
        self.assertEqual(FINGERPRINT_FLOAT_DECIMAL_PLACES, 10)
        self.assertEqual(_frame_fingerprint(baseline), _frame_fingerprint(noisy))

    def test_frame_fingerprint_detects_meaningful_numeric_change(self) -> None:
        index = pd.date_range("2025-01-01", periods=3, name="Date")
        baseline = pd.DataFrame({"value": [1.25, 2.5, 8_000.0]}, index=index)
        changed = baseline.copy()
        changed.loc[index[0], "value"] += 1e-4
        self.assertNotEqual(_frame_fingerprint(baseline), _frame_fingerprint(changed))

    def test_fingerprints_detect_column_order_and_schema_changes(self) -> None:
        index = pd.date_range("2025-01-01", periods=2, name="Date")
        baseline = pd.DataFrame({"count": [1, 2], "flag": [True, False]}, index=index)
        reordered = baseline.loc[:, ["flag", "count"]]
        renamed = baseline.rename(columns={"count": "active_count"})
        for changed in (reordered, renamed):
            self.assertNotEqual(_frame_fingerprint(baseline), _frame_fingerprint(changed))
            self.assertNotEqual(_schema_fingerprint(baseline), _schema_fingerprint(changed))

    def test_frame_fingerprint_detects_missing_value_location(self) -> None:
        index = pd.date_range("2025-01-01", periods=3, name="Date")
        first = pd.DataFrame({"value": [1.0, np.nan, 3.0]}, index=index)
        second = pd.DataFrame({"value": [np.nan, 1.0, 3.0]}, index=index)
        self.assertNotEqual(_frame_fingerprint(first), _frame_fingerprint(second))

    def test_fingerprint_algorithm_version_is_v3(self) -> None:
        self.assertEqual(FINGERPRINT_ALGORITHM, "canonical-semantic-v3")

    def test_change_target_and_absolute_reconstruction(self) -> None:
        index = pd.date_range("2024-01-01", periods=3)
        current = pd.Series([100.0, 110.0, 90.0], index=index)
        future = pd.Series([108.0, 105.0, 100.0], index=index)
        change = construct_change_target(future, current)
        np.testing.assert_array_equal(change, [8.0, -5.0, 10.0])
        np.testing.assert_array_equal(reconstruct_absolute_forecast(current, change), future)

    def test_chronological_holdout_and_seven_day_gap(self) -> None:
        config = make_config()
        prepared = prepare_forecast_dataset(make_source(), config)
        partitions = chronological_partitions(prepared, config)
        self.assertLess(partitions.development.index.max(), partitions.embargo.index.min())
        self.assertLess(partitions.embargo.index.max(), partitions.holdout.index.min())
        self.assertEqual(len(partitions.embargo), 7)
        folds = expanding_window_folds(len(partitions.development), config)
        self.assertEqual(len(folds), 4)
        self.assertTrue(all(fold.observed_gap == 7 for fold in folds))
        self.assertTrue(all(fold.train_indices[-1] < fold.validation_indices[0] for fold in folds))

    def test_holdout_is_untouched_by_fold_boundaries(self) -> None:
        config = make_config()
        prepared = prepare_forecast_dataset(make_source(), config)
        partitions = chronological_partitions(prepared, config)
        folds = expanding_window_folds(len(partitions.development), config)
        latest_development_position = max(fold.validation_indices[-1] for fold in folds)
        self.assertLess(latest_development_position, len(partitions.development))
        self.assertTrue(set(partitions.holdout.index).isdisjoint(partitions.development.index))

    def test_feature_availability_excludes_future_and_same_day_flows(self) -> None:
        prepared = prepare_forecast_dataset(make_source(), make_config())
        features = set(prepared.expanded_features)
        self.assertIn(CURRENT_LOAD, features)
        self.assertFalse(any("target" in name or "future" in name for name in features))
        self.assertNotIn(TRANSFER_COLUMN, features)
        self.assertNotIn(DISCHARGE_COLUMN, features)
        targets = prepared.availability.loc[
            prepared.availability["feature"].isin([TARGET_ABSOLUTE, TARGET_CHANGE])
        ]
        self.assertTrue((~targets["included"]).all())

    def test_training_only_processor_fit_and_correlation_filter(self) -> None:
        index = pd.date_range("2024-01-01", periods=20)
        training = pd.DataFrame(
            {
                "a": np.arange(20, dtype=float),
                "b": np.arange(20, dtype=float) * 2,
                "c": [1.0, np.nan] * 10,
            },
            index=index,
        )
        processor = FeatureProcessor(0.6, 0.95).fit(training.iloc[:15])
        self.assertEqual(processor.fitted_rows[1], index[14].date().isoformat())
        self.assertIn("b", processor.exclusion_reasons)
        transformed = processor.transform(training.iloc[15:])
        self.assertFalse(transformed.isna().any().any())

    def test_persistence_and_drift_formulas(self) -> None:
        frame = pd.DataFrame({CURRENT_LOAD: [100.0, 120.0], "load_lag_7": [90.0, 125.0]})
        np.testing.assert_array_equal(persistence_forecast(frame), [100.0, 120.0])
        np.testing.assert_array_equal(drift_forecast(frame), [110.0, 115.0])

    def test_mase_and_metric_aggregation(self) -> None:
        metrics = regression_metrics(
            actual=[10.0, 20.0, 30.0],
            predicted=[11.0, 18.0, 33.0],
            persistence_mae=4.0,
        )
        self.assertAlmostEqual(metrics["mae"], 2.0)
        self.assertAlmostEqual(metrics["mase_vs_persistence"], 0.5)
        self.assertAlmostEqual(metrics["mae_improvement_vs_persistence_percent"], 50.0)

    def test_prediction_interval_ordering(self) -> None:
        lower, median, upper = order_prediction_intervals(
            [120.0, 70.0], [100.0, 90.0], [110.0, 80.0]
        )
        np.testing.assert_array_equal(lower, [100.0, 70.0])
        np.testing.assert_array_equal(median, [110.0, 80.0])
        np.testing.assert_array_equal(upper, [120.0, 90.0])

    def test_hybrid_residuals_require_oof_alignment(self) -> None:
        dates = pd.date_range("2024-01-01", periods=4)
        residuals = build_oof_residuals(
            pd.Series([10.0, 12.0, 15.0, 14.0], index=dates),
            pd.Series([9.0, 13.0, 13.0], index=dates[:3]),
        )
        np.testing.assert_array_equal(residuals, [1.0, -1.0, 2.0])
        duplicated = pd.Series([9.0, 10.0], index=[dates[0], dates[0]])
        with self.assertRaisesRegex(ForecastingFrameworkError, "unique"):
            build_oof_residuals(pd.Series([10.0], index=dates[:1]), duplicated)

    def test_ensemble_weights_are_deterministic_nonnegative_and_normalized(self) -> None:
        actual = np.array([1.0, 2.0, 3.0, 4.0])
        predictions = np.column_stack([actual, np.array([2.0, 2.0, 2.0, 2.0]), actual + 0.5])
        first = optimize_ensemble_weights(actual, predictions)
        second = optimize_ensemble_weights(actual, predictions)
        np.testing.assert_array_equal(first, second)
        self.assertTrue((first >= 0).all())
        self.assertAlmostEqual(float(first.sum()), 1.0)

    def test_prediction_schema(self) -> None:
        row = {column: 0 for column in PREDICTION_COLUMNS}
        row.update(
            {
                "forecast_origin_date": "2024-01-01",
                "target_date": "2024-01-08",
                "model_name": "persistence",
                "evaluation_label": "fixture",
                "fold": 1,
            }
        )
        validate_prediction_schema(pd.DataFrame([row]))
        with self.assertRaisesRegex(ForecastingFrameworkError, "missing"):
            validate_prediction_schema(pd.DataFrame([row]).drop(columns=["actual_value"]))

    def test_missing_dependency_has_actionable_error(self) -> None:
        from src import forecasting

        original = forecasting.importlib.import_module

        def fail_for_catboost(name: str) -> object:
            if name == "catboost":
                raise ImportError("fixture")
            return original(name)

        with patch.object(forecasting.importlib, "import_module", side_effect=fail_for_catboost):
            with self.assertRaisesRegex(ForecastingFrameworkError, "requirements.txt"):
                forecasting._require("catboost")

    def test_empty_duplicate_and_non_daily_data_fail_clearly(self) -> None:
        config = make_config()
        with self.assertRaisesRegex(ForecastingFrameworkError, "missing required"):
            prepare_forecast_dataset(pd.DataFrame(), config)
        duplicated = pd.concat([make_source(420), make_source(420).iloc[[-1]]]).sort_index()
        with self.assertRaisesRegex(ForecastingFrameworkError, "unique"):
            prepare_forecast_dataset(duplicated, config)
        non_daily = make_source(420).drop(make_source(420).index[50])
        with self.assertRaisesRegex(ForecastingFrameworkError, "complete daily"):
            prepare_forecast_dataset(non_daily, config)

    def test_dashboard_page_has_no_training_dependency(self) -> None:
        page = Path("app/pages/forecasting.py").read_text(encoding="utf-8")
        self.assertNotIn("from src.forecasting", page)
        self.assertNotIn("train_forecasting_models(", page)
        self.assertIn("python main.py train-models", page)

    def test_provenance_input_files_are_configurable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory))
            self.assertEqual(config.raw_path.name, "source.csv")
            self.assertEqual(config.provenance_path.name, "preprocessing.json")
            self.assertEqual(config.output_dir.name, "forecasting")


if __name__ == "__main__":
    unittest.main()
