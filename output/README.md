# Generated technical outputs

This directory contains reproducible runtime artifacts, not source data.

- `charts/` contains standalone Plotly HTML figures.
- `exports/` contains analytical JSON/CSV extracts and test predictions.
- `models/` contains the NumPy ridge baseline and its evaluation metadata.
- `logs/` is reserved for runtime application or batch-pipeline logs.

The baseline model predicts Total System Load seven days ahead. It is a research
benchmark, not an official HHS forecast or an operational capacity decision.
Regenerate analysis JSON with `python main.py analyze`, reports with
`python main.py report`, and model artifacts by running notebook 05.
