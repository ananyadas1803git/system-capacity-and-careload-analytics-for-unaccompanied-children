# Research notebooks

These notebooks provide an auditable research path from source-data validation
through forecasting evaluation. Run them in numeric order from the repository
root or from this directory.

| Notebook | Purpose |
| --- | --- |
| `01_data_quality_audit.ipynb` | Audit the unverified source, repairs, gaps, and logical anomalies. |
| `02_exploratory_capacity_analysis.ipynb` | Explore care-load composition, seasonality, growth, and flows. |
| `03_backlog_pressure_analysis.ipynb` | Analyze positive-pressure streaks and elevated backlog episodes. |
| `04_feature_engineering.ipynb` | Build leakage-aware features, targets, and a model-audit manifest. |
| `05_capacity_forecasting.ipynb` | Train a dependency-light seven-day ridge baseline chronologically. |
| `06_model_evaluation.ipynb` | Compare the baseline with a naive forecast and inspect residuals. |
| `07_multimodel_forecasting.ipynb` | Load precomputed multi-model artifacts, provenance, promotion, intervals, and diagnostics. |

The source contains aggregate operational counts rather than individual-level
records, but its publisher and acquisition details are not recorded. Do not
represent the data or model outputs as official HHS information or as decisions
about individual children.

Notebook outputs are intentionally cleared in version control. Reusable logic
belongs under `src/`; generated charts, predictions, and model files belong
under `output/`.
