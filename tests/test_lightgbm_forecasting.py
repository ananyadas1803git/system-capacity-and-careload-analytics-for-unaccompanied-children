"""Focused invariants for the leakage-safe LightGBM forecasting pipeline."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from app_utils import DATE_COLUMN, TOTAL_LOAD_COLUMN
from src.lightgbm_forecasting import (
    ABSOLUTE_TARGET_COLUMN,
    CHANGE_TARGET_COLUMN,
    CURRENT_LOAD_FEATURE,
    IMPORTANCE_ARTIFACT_COLUMNS,
    PREDICTION_ARTIFACT_COLUMNS,
    ForecastingError,
    LightGBMForecastConfig,
    chronological_holdout_split,
    construct_change_target,
    expanding_window_folds,
    find_leakage_features,
    fit_deterministic_lightgbm,
    order_prediction_intervals,
    prepare_forecasting_data,
    reconstruct_absolute_forecast,
    train_lightgbm_forecast,
    validate_artifact_schemas,
)


SMALL_CANDIDATE = {
    "learning_rate": 0.05,
    "n_estimators": 30,
    "num_leaves": 7,
    "max_depth": 3,
    "min_child_samples": 10,
    "colsample_bytree": 0.90,
    "subsample": 0.90,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
}


def make_feature_frame(rows: int = 150) -> pd.DataFrame:
    """Create a deterministic daily feature artifact for unit tests."""

    index = pd.date_range("2023-01-01", periods=rows, freq="D", name=DATE_COLUMN)
    time = np.arange(rows, dtype=float)
    current = pd.Series(
        1_000.0 + 0.4 * time + 15.0 * np.sin(time / 9.0),
        index=index,
    )
    future = current.shift(-7)
    frame = pd.DataFrame(index=index)
    frame[TOTAL_LOAD_COLUMN] = current
    frame[ABSOLUTE_TARGET_COLUMN] = future
    frame["target_load_change_t_plus_7d"] = future - current
    frame["target_future_leak"] = future
    frame["calendar_day_of_week"] = index.dayofweek
    frame["calendar_month_sin"] = np.sin(2 * np.pi * (index.month - 1) / 12)
    frame["lag_1d_total_system_load"] = current.shift(1)
    frame["lag_7d_total_system_load"] = current.shift(7)
    frame["lag_1d_cbp_transfers"] = pd.Series(100 + time % 11, index=index).shift(1)
    frame["rolling_7d_total_system_load_mean"] = current.shift(1).rolling(7).mean()
    frame["ema_7d_total_system_load"] = current.shift(1).ewm(span=7).mean()
    frame["momentum_total_load_change_1d"] = current.diff()
    frame["momentum_net_intake_change_1d"] = pd.Series(time, index=index).diff()
    frame["operational_transfer_discharge_gap"] = time % 5
    frame["quality_transfer_anomaly"] = 0
    frame["quality_is_imputed_date"] = 0
    return frame


class LightGBMForecastingTests(unittest.TestCase):
    """Validate targets, splits, determinism, leakage, and artifact contracts."""

    def make_config(self, root: Path | None = None) -> LightGBMForecastConfig:
        base = root or Path("/private/tmp/hhs-uac-lightgbm-test")
        return LightGBMForecastConfig(
            feature_path=base / "features.parquet",
            provenance_path=None,
            ridge_predictions_path=None,
            model_path=base / "models" / "model.txt",
            metadata_path=base / "models" / "metadata.json",
            evaluation_path=base / "models" / "evaluation.json",
            predictions_path=base / "exports" / "predictions.csv",
            feature_importance_path=base / "exports" / "importance.csv",
            minimum_training_rows=40,
            cv_splits=2,
            cv_validation_rows=10,
            candidate_parameters=(SMALL_CANDIDATE,),
            require_ridge_baseline=False,
            overwrite=True,
            source_label="unit-test synthetic data",
            synthetic_data=True,
        )

    def test_constructs_seven_day_change_target(self) -> None:
        index = pd.date_range("2024-01-01", periods=3)
        current = pd.Series([100.0, 120.0, 140.0], index=index)
        future = pd.Series([108.0, 115.0, 161.0], index=index)
        result = construct_change_target(future, current)
        np.testing.assert_array_equal(result.to_numpy(), [8.0, -5.0, 21.0])
        self.assertEqual(result.name, CHANGE_TARGET_COLUMN)

    def test_chronological_split_and_walk_forward_gap(self) -> None:
        config = self.make_config()
        prepared = prepare_forecasting_data(make_feature_frame(), config)
        split = chronological_holdout_split(prepared, config)
        self.assertLess(split.train.index.max(), split.gap.index.min())
        self.assertLess(split.gap.index.max(), split.holdout.index.min())
        self.assertEqual(len(split.gap), 7)
        folds = expanding_window_folds(len(split.train), config)
        self.assertEqual(len(folds), 2)
        for fold in folds:
            self.assertEqual(fold.observed_gap_rows, 7)
            self.assertLess(fold.train_indices[-1], fold.validation_indices[0])

    def test_feature_selection_excludes_targets_and_same_day_operations(self) -> None:
        config = self.make_config()
        prepared = prepare_forecasting_data(make_feature_frame(), config)
        features = set(prepared.feature_columns)
        self.assertFalse(find_leakage_features(prepared.feature_columns))
        self.assertNotIn("target_future_leak", features)
        self.assertNotIn("operational_transfer_discharge_gap", features)
        self.assertNotIn("momentum_net_intake_change_1d", features)
        self.assertNotIn("quality_transfer_anomaly", features)
        self.assertIn("lag_1d_cbp_transfers", features)
        self.assertIn(CURRENT_LOAD_FEATURE, features)

    def test_deterministic_model_training(self) -> None:
        config = self.make_config()
        prepared = prepare_forecasting_data(make_feature_frame(), config)
        split = chronological_holdout_split(prepared, config)
        columns = list(prepared.feature_columns)
        first = fit_deterministic_lightgbm(
            split.train[columns],
            split.train[CHANGE_TARGET_COLUMN],
            SMALL_CANDIDATE,
            config,
        )
        second = fit_deterministic_lightgbm(
            split.train[columns],
            split.train[CHANGE_TARGET_COLUMN],
            SMALL_CANDIDATE,
            config,
        )
        np.testing.assert_array_equal(
            first.predict(split.holdout[columns]),
            second.predict(split.holdout[columns]),
        )

    def test_reconstructs_absolute_predictions(self) -> None:
        reconstructed = reconstruct_absolute_forecast(
            [1_000.0, 1_100.0],
            [25.0, -10.0],
        )
        np.testing.assert_array_equal(reconstructed, [1_025.0, 1_090.0])

    def test_prediction_interval_ordering(self) -> None:
        lower, median, upper = order_prediction_intervals(
            [120.0, 80.0],
            [100.0, 90.0],
            [110.0, 70.0],
        )
        np.testing.assert_array_equal(lower, [100.0, 70.0])
        np.testing.assert_array_equal(median, [110.0, 80.0])
        np.testing.assert_array_equal(upper, [120.0, 90.0])

    def test_full_pipeline_writes_stable_artifact_schemas(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.make_config(root)
            make_feature_frame().to_parquet(config.feature_path, engine="pyarrow")
            result = train_lightgbm_forecast(config)

            validate_artifact_schemas(result.predictions, result.feature_importance)
            self.assertEqual(
                tuple(result.predictions.columns), PREDICTION_ARTIFACT_COLUMNS
            )
            self.assertEqual(
                tuple(result.feature_importance.columns),
                IMPORTANCE_ARTIFACT_COLUMNS,
            )
            for path in config.artifact_paths:
                self.assertTrue(path.is_file(), path)
            metadata = json.loads(config.metadata_path.read_text(encoding="utf-8"))
            evaluation = json.loads(config.evaluation_path.read_text(encoding="utf-8"))
            self.assertIn("data_fingerprint_sha256", metadata)
            self.assertIn("feature_schema_fingerprint_sha256", metadata)
            self.assertIn("promotion", evaluation)
            self.assertIn("prediction_interval", evaluation["holdout"])

    def test_missing_and_invalid_data_raise_clear_errors(self) -> None:
        config = self.make_config()
        missing_target = make_feature_frame().drop(columns=[ABSOLUTE_TARGET_COLUMN])
        with self.assertRaisesRegex(ForecastingError, "Missing required"):
            prepare_forecasting_data(missing_target, config)

        invalid = make_feature_frame()
        invalid.loc[invalid.index[20], "lag_1d_total_system_load"] = np.inf
        with self.assertRaisesRegex(ForecastingError, "infinite"):
            prepare_forecasting_data(invalid, config)

        with self.assertRaisesRegex(ForecastingError, "empty"):
            prepare_forecasting_data(pd.DataFrame(), config)


if __name__ == "__main__":
    unittest.main()
