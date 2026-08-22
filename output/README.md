# Generated technical outputs

This directory contains reproducible runtime artifacts, not source data.

- `charts/` contains standalone Plotly HTML figures.
- `exports/` contains analytical JSON/CSV extracts, ridge and LightGBM holdout
  predictions, and LightGBM feature importance.
- `models/` contains the preserved NumPy ridge benchmark, the LightGBM text
  model, provenance metadata, evaluation metrics, and promotion decision.
- `logs/` is reserved for runtime application or batch-pipeline logs.

The ridge and LightGBM candidates forecast Total System Load seven days ahead.
They are research benchmarks, not official HHS forecasts or operational
capacity decisions. Regenerate analysis JSON with `python main.py analyze`,
reports with `python main.py report`, ridge artifacts with notebook 05, and
LightGBM artifacts with `python main.py train-lightgbm --force`.
