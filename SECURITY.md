# Security Policy

## Supported scope

The current `main` branch is the supported research version. This repository is not an official government service and has no production security SLA.

## Reporting a vulnerability

Do not disclose secrets, personal data, or an exploitable vulnerability in a public issue. Use GitHub's private vulnerability reporting feature for this repository when available, or contact the maintainer privately through the GitHub profile listed in the README. Include affected files, reproduction steps, impact, and a suggested mitigation without including real sensitive records.

## Security assumptions

- The included data is aggregate and contains no child-level identifiers.
- Uploaded CSV bodies are size-limited and validated before analysis.
- CORS defaults to local dashboard origins and credentials are disabled.
- API errors do not expose tracebacks to clients.
- Forecast endpoints read JSON/CSV metadata and do not deserialize model binaries.
- Containers run as a non-root user with read-only filesystems in Compose.

Operators must provide TLS, authentication, authorization, rate limiting, centralized logs, dependency scanning, backup/recovery, and secret management before any public deployment. Never load an untrusted `.joblib` or pickle file; Python model serialization can execute code.
