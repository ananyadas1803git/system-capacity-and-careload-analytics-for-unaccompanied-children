# Deployment Guide

No deployment is performed automatically by this repository. Review data rights, security controls, and the research-only disclaimer before making any service public.

## Streamlit Community Cloud

1. Push the repository and approved artifacts to GitHub.
2. In Streamlit Community Cloud, create an app from the repository.
3. Set the entry point to `app/streamlit_app.py` and branch to `main`.
4. Use Python 3.13 when selectable. Dependencies are installed from root `requirements.txt`.
5. Do not add secrets unless a future authenticated source requires them. If secrets are introduced, store them in the platform secret manager, never in `.streamlit/secrets.toml` in Git.
6. Confirm the source warning, Forecast Research page, charts, and download behavior after deployment.

There is intentionally no live-demo badge or URL until a real deployment is verified.

## Docker Compose

```bash
docker compose build
docker compose up
```

This starts the dashboard on port 8501 and the API on port 8000. The image runs as a non-root user; Compose sets read-only filesystems with `/tmp` as `tmpfs`.

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/api/v1/model
```

Stop with `docker compose down`.

## Production boundary

Before a public or organizational deployment, add TLS at a reverse proxy, authentication and authorization, request throttling, centralized audit logs, secret management, dependency/image scanning, backups, a vulnerability-response process, and a documented rollback. Restrict CORS through `HHS_API_CORS_ORIGINS`. Never expose the API as an official government service or deserialize untrusted model files.
