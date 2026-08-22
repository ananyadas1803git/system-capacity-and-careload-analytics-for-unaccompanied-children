# Contributing

Contributions are welcome through focused issues and pull requests.

1. Create a branch from `main`.
2. Install `requirements-dev.txt` and run `pre-commit install`.
3. Keep raw inputs immutable and never commit credentials or personal data.
4. Add tests for changed behavior and preserve chronological/leakage invariants.
5. Run Ruff, all unit tests, quick mode, and approved-artifact verification.
6. Update data/model cards and the changelog when behavior, data, metrics, or limitations change.

Do not select models from final-holdout ranking, remove provenance warnings, or describe synthetic/unverified data as official. Large new dependencies should include a reproducibility and deployment rationale.
