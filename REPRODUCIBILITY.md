# Reproducibility Guide

## Supported environment

The reproducibility target is Python 3.13 with pinned runtime and development dependencies; CatBoost 1.2.8 does not provide a Python 3.14 wheel. An exact reproduction should use Python 3.13 and the package versions recorded in `output/forecasting/models/model_registry.json`.

The project was verified on macOS/Apple Silicon. CI targets Ubuntu Linux. Windows is supported through Python commands and the documented activation syntax, but shell quoting and Docker Desktop setup differ. Use a fresh environment rather than an Anaconda base environment to avoid conflicts with unrelated scientific packages.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

On Windows, activate with `.venv\Scripts\activate`.

Anaconda alternative:

```bash
conda env create -f environment.yml
conda activate hhs-uac-analytics
```

## Determinism controls

- Global random seed: 42.
- Expanding-window boundaries and seven-day gaps are deterministic.
- Ensemble weights use development OOF predictions only.
- Feature preprocessing is fit independently inside each fold.
- LightGBM, CatBoost, XGBoost, and scikit-learn receive fixed seeds and CPU thread limits where supported.
- Source, prepared dataset, schema, configuration, library versions, and selected features are persisted.

Small floating-point differences may occur across CPU architectures and compiled numerical libraries. Promotion conclusions should remain unchanged; compare metrics with an appropriate numerical tolerance rather than requiring byte-identical model binaries.

The registry records the Git revision, whether the worktree was clean, training configuration, development/holdout dates, fingerprints, seed, timestamp, and dependency versions. A `false` worktree-clean value means the revision alone is insufficient; preserve the accompanying diff or commit the reviewed code before a publication-quality rerun.

## Fast verification

```bash
python -m ruff check .
python -m ruff format --check .
python -m unittest discover -s tests -v
python main.py pipeline --quick --output-dir output/quick_pipeline --force
python main.py pipeline
python main.py monitor-model --no-write
```

Quick mode uses deterministic synthetic data and a Ridge smoke model. It exercises ingestion, validation, feature engineering, chronological splitting, fold-only preprocessing, fitting, prediction, schema checks, and artifact writing in seconds. It is a software check, not a promoted research result.

## Rebuild data artifacts

```bash
python main.py generate-data
python main.py validate
```

The raw CSV is read-only input. Review `data/processed/preprocessing_report.json` and `validation_report.json` after regeneration.

## Verify or retrain the forecasting framework

The safe default does not train:

```bash
python main.py pipeline
```

It regenerates preprocessing and features, compares them to the canonical artifact, verifies hashes and schemas, checks promotion/leakage consistency, and loads only frozen metadata and CSV artifacts.

Full deliberate retraining overwrites only the selected forecasting output directory:

```bash
python main.py pipeline --train --force
```

Do not choose a champion by final-holdout performance after retraining. Selection must remain based on development OOF MAE, with the holdout used only for the predefined promotion gate.

On the verified laptop, quick mode finishes in about 2 seconds and full training in about 25 seconds. Allow approximately 2 GB RAM and two minutes for slower CI or container hosts. The largest current approved model artifact is roughly 18 MB.

## Run services

```bash
python main.py dashboard
python main.py api --host 127.0.0.1 --port 8000
```

Or run both containers:

```bash
docker compose up --build
```

Dashboard: `http://127.0.0.1:8501`; API health: `http://127.0.0.1:8000/health`.

## CI parity

`.github/workflows/ci.yml` installs `requirements-dev.txt`, runs lint and formatting checks, executes all tests, runs quick mode, verifies approved artifacts, and evaluates offline monitoring. Run the commands above before pushing.
