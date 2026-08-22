# Portfolio and Interview Guide

## Resume-ready summary

Built a reproducible Python forecasting and analytics platform for aggregate care-load data, combining audited preprocessing, leakage-safe expanding-window evaluation across 11 models, uncertainty intervals, artifact provenance, offline drift/fallback monitoring, a multipage Streamlit dashboard, a versioned Starlette API, tests, CI, and Docker deployment. The development-selected ensemble reduced frozen-holdout MAE 25.7% versus persistence while preserving honest baseline and limitation reporting.

## Two-minute demo script

1. Open the Overview page and explain the aggregate care-flow problem, source warning, and five operational KPIs.
2. Open Forecasting and point to the champion, promotion rule, and chronological holdout chart.
3. Compare the ensemble with persistence and seven-day drift; explain why the lowest holdout score did not replace the development-selected champion.
4. Show error-by-regime and interval width to demonstrate uncertainty and failure analysis.
5. Run `python main.py pipeline` and `python main.py monitor-model --no-write` to show reproducibility, fingerprint checks, model version, drift indicators, and explicit fallback.
6. Open `/api/v1/model/provenance` and the data/model cards to close with governance and deployment boundaries.

## Interview talking points

1. **Leakage control:** the seven-day forecast uses fold-specific preprocessing, seven-day gaps, a separate pre-holdout embargo, lagged operational flows, and OOF-only ensemble weights.
2. **Scientific honesty:** source provenance is explicitly unverified; a strong drift baseline is retained; seven-day drift's better holdout score is reported without post-hoc champion switching.
3. **Production thinking:** dashboard rendering never trains, APIs serve inert frozen artifacts, monitoring exposes model version and fallback events, CI covers a fast E2E path, and Docker runs non-root read-only services.

## Limitations to volunteer

1. The source file lacks authoritative acquisition metadata, so the results do not establish HHS operational generalization.
2. Missing dates and values required substantial imputation, and the tested stock-flow identity did not reconcile reliably.
3. One short aggregate series cannot represent facility, geographic, staffing, welfare, policy, or demographic heterogeneity.
