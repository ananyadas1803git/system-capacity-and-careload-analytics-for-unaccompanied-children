# Data Card: HHS UAC Capacity Analytics Dataset

## Status and intended use

This repository contains an aggregate daily operational-count file used to demonstrate data validation, capacity analytics, and seven-day forecasting. Its provenance is **unknown/unverified**. The filename resembles an HHS program export, but the repository does not contain an authoritative publisher URL, acquisition timestamp, data license, external signature, or chain-of-custody record. It must not be presented as verified HHS or CBP data.

The data is suitable for software testing, portfolio demonstration, and methods research with prominent caveats. It is not suitable for official statistics, causal inference, resource-allocation decisions, or decisions about individual children.

## Files and lineage

| Artifact | Role |
|---|---|
| `data/raw/HHS_Unaccompanied_Alien_Children_Program.csv` | Preserved source input; never modified by the pipeline |
| `data/processed/uac_capacity_cleaned_daily.csv` | Complete daily series with quality and imputation flags |
| `data/processed/uac_capacity_metrics_daily.csv` | Capacity metrics derived from the cleaned series |
| `data/processed/uac_capacity_ml_features.parquet` | Leakage-aware analytical and modeling features |
| `data/processed/preprocessing_report.json` | Transformation counts, hashes, and audit metadata |
| `data/processed/validation_report.json` | Machine-readable validation findings |

The raw file SHA-256 recorded by the approved model registry is `061af0a97a1b3bda7a36f0ce8df08b6994847b0a4c402828cf2ef835d4947198`. The canonical processed-model fingerprint is `52a994fa234fa89e6a904430afa9f9850c932b7b65d8a01a76b8ae782c4ef541`, generated with `canonical-semantic-v2` numeric canonicalization.

## Schema

| Field | Meaning | Unit |
|---|---|---|
| `Date` | Reporting date | calendar day |
| `Children apprehended and placed in CBP custody` | Daily intake | children/day |
| `Children in CBP custody` | Active CBP care load | children |
| `Children transferred out of CBP custody` | Daily transfers out of CBP custody | children/day |
| `Children in HHS Care` | Active HHS care load | children |
| `Children discharged from HHS Care` | Daily HHS discharges | children/day |

No direct identifiers, child-level records, demographic attributes, case files, or protected health information are present. Counts still concern a vulnerable population and require careful framing.

## Coverage and preprocessing audit

- Processed coverage: 2023-01-12 through 2025-12-21.
- Raw rows reported by the preprocessing audit: 1,170.
- Complete processed daily rows: 1,075.
- Missing dates inserted: 355.
- Numeric values imputed: 1,775.
- Duplicate dates after processing: 0.
- Target-complete seven-day forecast origins: 1,068.

The preprocessor parses and sorts dates, removes empty export padding, collapses duplicate dates deterministically, reindexes to a complete daily calendar, fills flow values with zero on inserted dates, interpolates stock values, clips impossible negative values, and records every repair through row- and cell-level flags. Validation remains marked failed when source anomalies are detected; repairs do not erase the audit finding.

## Known quality concerns

- The source contains missing calendar dates and nonchronological rows.
- Some transfers exceed recorded CBP custody and some discharges exceed recorded HHS care.
- Inserted and imputed observations comprise a material share of the processed series.
- The tested stock-flow identity, `change in total load ≈ apprehensions - discharges`, has a median absolute reconciliation error of 47 children; only 26.07% of days are within 10 children. The structural-flow forecast was therefore excluded.
- Collection definitions, revision policy, reporting timezone, suppression rules, and changes in administrative practice are not documented.
- Missingness may be informative and the interpolation assumptions may not match the original data-generating process.

## Synthetic fallback

`generate_mock_data()` creates a deterministic 2023–2025 daily dataset for UI demonstration and tests. Synthetic results are labeled separately and must never be described as observed government data.

## Access, redistribution, and maintenance

The original acquisition URL, retrieval date, and redistribution terms are not recorded. Users are responsible for verifying that they have permission to redistribute or reuse the raw file. Before any public or operational use, replace it with a documented authoritative source and record publisher, direct URL, retrieval timestamp, version, license, checksum, and field definitions.

Data-quality and model artifacts can be regenerated with `python main.py generate-data` and verified with `python main.py pipeline`. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
