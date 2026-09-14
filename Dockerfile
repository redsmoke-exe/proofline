FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=10000 \
    CV_AGENT_CACHE_DIR=/app/runtime/cache \
    CV_AGENT_RUNS_DIR=/app/output/runs \
    CV_AGENT_EXAMPLES_DIR=/app/examples

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        fonts-liberation \
        libffi8 \
        libharfbuzz-subset0 \
        libjpeg62-turbo \
        libopenjp2-7 \
        libpango-1.0-0 \
        libpangoft2-1.0-0 \
        shared-mime-info \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml requirements.txt README.md ./
COPY src ./src
COPY examples ./examples
RUN python -m pip install --no-cache-dir . \
    && groupadd --system proofline \
    && useradd --system --gid proofline --home-dir /app proofline \
    && mkdir -p /app/output/runs /app/runtime/cache \
    && chown -R proofline:proofline /app

USER proofline

EXPOSE 10000

CMD ["sh", "-c", "uvicorn cv_agent.api:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
