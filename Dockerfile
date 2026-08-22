# syntax=docker/dockerfile:1.7
FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HHS_API_HOST=0.0.0.0 \
    HHS_API_PORT=8000

RUN groupadd --system app && useradd --system --gid app --create-home app

WORKDIR /workspace

COPY requirements.txt ./
RUN python -m pip install --upgrade pip && python -m pip install -r requirements.txt

COPY --chown=app:app . .
USER app

EXPOSE 8501 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"

CMD ["python", "main.py", "dashboard", "--host", "0.0.0.0", "--port", "8501", "--headless"]
