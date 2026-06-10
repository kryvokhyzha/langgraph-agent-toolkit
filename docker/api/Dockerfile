# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Builder: resolve + install dependencies into an isolated venv. Kept out of
# the final image so the uv binary and build cache never ship to production.
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_PROJECT_ENVIRONMENT=/opt/venv

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /bin/uv

WORKDIR /app

# Only the manifest + lockfile are needed to install dependencies. Copying them
# before the source keeps this expensive layer cached across code changes; the
# cache mount reuses uv's download/wheel cache across builds.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev \
    --extra gunicorn-backend --extra all-observability --extra openai

# ---------------------------------------------------------------------------
# Runtime: minimal image with just the venv + application source.
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS runtime

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH

# curl is used by the container healthcheck (GET /health/ready).
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY langgraph_agent_toolkit/ ./langgraph_agent_toolkit/

# Run as a non-root user.
RUN useradd --create-home --shell /bin/bash appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# Gunicorn worker management (see run_api.py):
# --workers          N worker processes (recommend 2-4 per CPU core)
# --preload_app      load the app once before forking — faster worker startup,
#                    shared memory, agents initialized once (best for prod)
# --timeout          kill workers that don't respond in N s (> slowest request)
# --graceful_timeout time for in-flight requests to finish before a worker is killed
# --max_requests[_jitter]  recycle workers after N requests to bound memory growth
CMD ["python", "langgraph_agent_toolkit/run_api.py", "run_api", \
    "--runner_type", "gunicorn", \
    "--workers", "1", \
    "--preload_app", \
    "--timeout", "120", \
    "--graceful_timeout", "30"]

# ====================================================
# Health check endpoints for Kubernetes:
# - /health/live    - Liveness probe (is process alive?)
# - /health/ready   - Readiness probe (can accept traffic?)
# - /health/startup - Startup probe (has initialization completed?)
# - /health/db      - Database pool health (for monitoring)
# ====================================================
