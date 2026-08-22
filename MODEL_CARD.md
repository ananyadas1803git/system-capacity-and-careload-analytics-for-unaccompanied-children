# Model Card: Seven-Day Total System Load Forecast

## Model summary

The approved research champion forecasts the seven-day change in Total System Load and reconstructs the absolute load from the current origin-day load. It is a validation-weighted ensemble of 60% seven-day drift and 40% CatBoost. The champion was selected using development-period out-of-fold predictions only; the final chronological holdout was used as a promotion gate, not for selection.

Status: **promoted research prototype**, model-registry version 2.0.0. It is not an official HHS/CBP model and is not approved for operational decision-making.

## Intended and out-of-scope uses

Intended uses are reproducible methods research, aggregate scenario exploration, software testing, and portfolio review. Out-of-scope uses include official forecasts, child-level decisions, automated placement or custody recommendations, staffing orders, causal claims, and any use that assumes the source data is verified.

## Target and prediction contract

```text
target_change_7d[t] = total_system_load[t+7] - total_system_load[t]
forecast[t+7] = total_system_load[t] + predicted_change_7d[t]
```

Forecast-origin-safe features include current Total System Load, lagged load and care-system counts, lagged operational flows, shifted rolling statistics, calendar fields, backlog state, capacity reference features, and data-quality flags. Same-day apprehensions, transfers, and discharges are excluded. Targets, leads, and future-derived columns are prohibited.

## Evaluation design

- Development: 847 origins, 2023-01-12 to 2025-05-07.
- Five expanding-window folds with 56 validation days each.
- Seven-day gap between each training and validation period.
- Seven-row embargo before the final holdout.
- Frozen holdout: 214 origins, 2025-05-15 to 2025-12-14.
- Fold-specific imputation, near-constant removal, and correlation filtering.
- Fixed random seed 42 and single-threaded deterministic settings where supported.

The leakage audit records ten passed checks, including chronological folds, training-only preprocessing, future-feature exclusion, OOF ensemble weighting, and holdout isolation.

## Performance

| Model | Walk-forward MAE | Worst-fold MAE | Holdout MAE | Holdout MASE | Improvement vs persistence |
|---|---:|---:|---:|---:|---:|
| Persistence | 188.139 | 362.732 | 37.547 | 1.000 | 0.000% |
| Seven-day drift | 132.004 | 237.625 | **26.869** | 0.716 | 28.438% |
| CatBoost | 148.289 | 330.291 | 45.348 | 1.208 | -20.777% |
| Validation-weighted ensemble | **115.547** | **194.970** | 27.906 | **0.743** | **25.676%** |

The ensemble won three of five development folds and passed all predefined promotion gates. Seven-day drift happened to achieve the lowest holdout MAE, but changing the champion after observing the holdout would invalidate the evaluation design.

## Uncertainty

LightGBM quantile models estimate 10th, 50th, and 90th percentiles. Raw walk-forward coverage was 85.36% for a nominal 80% interval. Holdout coverage was 100% with mean width 655.45 children. These unusually conservative holdout intervals come from limited, nonstationary data and must not be interpreted as validated operational coverage guarantees.

## Error analysis

Champion holdout MAE was 44.56 in June and 36.09 in July, versus 18.21 in August and 19.88 in November. High forecast-load days had MAE 33.94, compared with 21.06 for low-load days. Moderate net-intake days had MAE 35.29, compared with 26.51 for low-magnitude days. Errors were similar on imputed and observed dates (27.07 versus 28.36 MAE), but this result does not prove that imputation is harmless.

Permutation importance places annual calendar position, seven-day load momentum, one-day load momentum, and rolling load slopes among the strongest signals. Correlated predictors can divide or mask importance; rankings are diagnostic, not causal.

## Monitoring and fallback

`python main.py monitor-model` evaluates rolling champion MAE, persistence MAE, MASE, interval coverage, imputation rate, input PSI, and standardized residual shift. The service explicitly switches its `active_model` metadata to persistence when promotion metadata disagrees, a required component artifact is missing, or rolling champion MAE exceeds 1.25 times persistence MAE. The event is recorded; no silent model switching occurs.

The current stored evaluation reports rolling 30-day MAE 23.84 versus persistence 31.40 and status `approved`. Input PSI is flagged, showing why performance and data drift must be reviewed together.

## Limitations and ethical considerations

- Source provenance is unknown/unverified and cannot establish external validity.
- Only one short aggregate series is available; policy and reporting regimes may shift.
- Hundreds of dates and values were imputed.
- Hyperparameter searches are deliberately small and CPU-friendly.
- Prediction-interval sample sizes are limited.
- Aggregate forecasting cannot capture facility geography, bed type, staffing, age, case complexity, legal constraints, or welfare outcomes.
- A capacity signal must never substitute for child-centered review, safeguards, or professional judgment.

## Artifacts and reproducibility

Registry, metrics, predictions, audits, diagnostics, and the HTML research report live under `output/forecasting/`. Run `python main.py pipeline` to reproduce preprocessing/feature stages and verify frozen artifacts, or `python main.py pipeline --train --force` for deliberate retraining. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) and [DATA_CARD.md](DATA_CARD.md).
