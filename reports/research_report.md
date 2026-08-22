# Leakage-Safe Seven-Day Forecasting of Aggregate Care Load

## Abstract

This study evaluates whether leakage-safe forecasting methods can improve seven-day aggregate care-load forecasts over a persistence baseline in a small, imperfect operational time series. Eleven statistical, linear, gradient-boosting, hybrid, and ensemble candidates were evaluated with five expanding-window folds, seven-day gaps, and a frozen chronological holdout. A validation-weighted ensemble selected solely from development out-of-fold predictions reduced mean walk-forward absolute error from 188.14 for persistence to 115.55 and achieved holdout MAE 27.91 versus 37.55. A seven-day drift baseline achieved slightly better holdout MAE (26.87), illustrating why final-holdout ranking must not replace a predefined selection rule. Source provenance is unverified and 355 dates were inserted, so results establish a reproducible research workflow rather than operational validity.

## Research questions and hypotheses

**RQ1.** Can a candidate selected only from development-period evidence reduce error relative to persistence?  
**H1.** At least one candidate will reduce mean expanding-window MAE and pass a final baseline promotion gate.

**RQ2.** Does improvement persist across time rather than arising from one favorable fold?  
**H2.** A promotable candidate will beat persistence in a majority of folds and control worst-fold error.

**RQ3.** How do errors vary by load, pressure, calendar, and data-quality regimes?  
**H3.** Error will be higher during higher-load or stronger-intake regimes.

## Data and preprocessing

The input is a daily six-column aggregate file describing apprehensions, CBP custody, transfers, HHS care, and discharges. Provenance is classified as unknown/unverified: no authoritative publisher URL, retrieval timestamp, license, or external signature is recorded. The local SHA-256 establishes lineage only.

Preprocessing removed empty padding, parsed and ordered dates, reconstructed a complete daily calendar, filled flow fields with zero on inserted dates, interpolated stock fields, clipped negatives, and added imputation/anomaly flags. The resulting series contains 1,075 days from 2023-01-12 through 2025-12-21; 355 dates and 1,775 numeric values were imputed. Logical validation findings remain visible.

## Methods

The target is the seven-day change in total system load. Absolute predictions are reconstructed by adding the origin-day load, which is known at forecast time. Features include historical lags, shifted rolling statistics, calendar encodings, backlog state, capacity references, and quality indicators. Same-day operational flows and all target/future fields are excluded.

Evaluated methods were persistence, seven-day drift, Ridge, Elastic Net, ETS/Holt-Winters, SARIMAX, LightGBM, CatBoost, XGBoost, a SARIMAX-plus-CatBoost OOF-residual hybrid, and a nonnegative validation-weighted ensemble. Small deterministic searches were used to keep the experiment reproducible on CPU hardware.

## Evaluation design and leakage controls

The first 847 eligible origins form the development period (2023-01-12 through 2025-05-07). Five expanding-window folds each contain 56 validation days and a seven-day gap. A second seven-row embargo separates development from 214 holdout origins (2025-05-15 through 2025-12-14). Every fold fits imputation and feature filters only on its training rows. Ensemble weights use aligned development OOF predictions only. The holdout is opened after configurations and weights are frozen.

Primary metrics are MAE, RMSE, MASE relative to persistence, and fold wins. MASE provides scale-relative context, while persistence is a strong and interpretable operational baseline. Prediction intervals use LightGBM quantile estimates with development-only calibration.

## Results

| Model | CV MAE | Worst-fold MAE | Holdout MAE | Holdout MASE |
|---|---:|---:|---:|---:|
| Persistence | 188.139 | 362.732 | 37.547 | 1.000 |
| Seven-day drift | 132.004 | 237.625 | **26.869** | 0.716 |
| LightGBM | 157.927 | 348.445 | 32.834 | 0.874 |
| CatBoost | 148.289 | 330.291 | 45.348 | 1.208 |
| XGBoost | 166.646 | 374.019 | 35.834 | 0.954 |
| Validation-weighted ensemble | **115.547** | **194.970** | 27.906 | **0.743** |

The frozen ensemble assigns 0.60 weight to seven-day drift and 0.40 to CatBoost. It beat persistence in three of five folds and improved holdout MAE by 25.68%, so all predefined gates passed. H1 and H2 are supported within this dataset. Seven-day drift's lower holdout MAE is reported but is not used to change the champion.

Raw nominal-80% walk-forward interval coverage was 85.36%. Holdout coverage was 100% with mean width 655.45 children. The width and small calibration sample imply conservative, uncertain intervals rather than proof of operational calibration.

## Error and sensitivity analysis

Higher-load origins had MAE 33.94 compared with 21.06 for low-load origins. Moderate net-intake magnitude had MAE 35.29 compared with 26.51 in the low band, broadly supporting H3. Monthly MAE peaked at 44.56 in June. Imputed and observed dates had similar measured MAE, but imputation can alter both inputs and targets; this comparison cannot validate the missingness assumptions.

The current offline monitor reports 30-day champion MAE 23.84 versus persistence 31.40. Population stability index is flagged even though residual shift and relative error are within thresholds, demonstrating that drift indicators should prompt review rather than automatically imply failure.

## Threats to validity

Internal validity is limited by missing dates, imputation, administrative anomalies, and uncertain stock-flow consistency. Construct validity is limited because Total System Load is only one dimension of capacity and omits facility type, geography, staffing, case complexity, and welfare outcomes. External validity is not established because the source is unverified and the observation window is short. Statistical validity is limited by five folds, one holdout period, small search spaces, and serial dependence.

## Ethical use

This analysis concerns a vulnerable population. It contains no child-level data, but its outputs could still be misinterpreted as prescriptive. Forecasts must not determine custody, placement, release, staffing, or services for an individual. Any applied use requires verified data, domain governance, child-welfare safeguards, uncertainty communication, and human review.

## Reproducibility

All boundaries, selected features, metrics, predictions, source/data/schema hashes, package versions, and promotion checks are stored under `output/forecasting/`. Run `python main.py pipeline` for non-training verification and `python main.py pipeline --train --force` for an explicit full rerun. Detailed instructions are in `REPRODUCIBILITY.md`.

## References

1. Administration for Children and Families. *FY 2025 Congressional Justification*, Unaccompanied Children program description. https://www.acf.hhs.gov/sites/default/files/documents/olab/fy-2025-congressional-justification.pdf
2. Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting*, 22(4), 679–688. https://doi.org/10.1016/j.ijforecast.2006.03.001
3. Bergmeir, C., Hyndman, R. J., & Koo, B. (2018). A note on the validity of cross-validation for evaluating autoregressive time series prediction. *Computational Statistics & Data Analysis*, 120, 70–83. https://doi.org/10.1016/j.csda.2017.11.003
4. Romano, Y., Patterson, E., & Candès, E. J. (2019). Conformalized quantile regression. *Advances in Neural Information Processing Systems*, 32. https://arxiv.org/abs/1905.03222
5. Wang, X., Hyndman, R. J., Li, F., & Kang, Y. (2023). Forecast combinations: an over 50-year review. *International Journal of Forecasting*, 39(4), 1518–1547. https://doi.org/10.1016/j.ijforecast.2022.11.005
