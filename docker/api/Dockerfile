# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Builder: install dependencies in an isolated virtual environment.
# The final image does not contain the `uv` binary or build cache.
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_PROJECT_ENVIRONMENT=/opt/venv

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /bin/uv

WORKDIR /app

# Install dependencies from the manifest and lockfile.
# Copy them before the source to cache this layer across source changes.
# The cache mount reuses the `uv` download and wheel cache across builds.
# Include both OpenAI transports. The runtime selects aiohttp by default.
COPY pyproject.toml uv.lock ./
ARG INSTALL_DEEPAGENTS=false
RUN --mount=type=cache,target=/root/.cache/uv \
    case "$INSTALL_DEEPAGENTS" in \
      true) set -- --extra deepagents ;; \
      false) set -- ;; \
      *) echo "INSTALL_DEEPAGENTS must be true or false" >&2; exit 1 ;; \
    esac && \
    uv sync --frozen --no-install-project --no-dev \
    --extra gunicorn-backend --extra all-observability --extra openai-aiohttp --extra mcp "$@"

# ---------------------------------------------------------------------------
# Runtime: include only the virtual environment and application source.
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS runtime

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    LLM_HTTP_ASYNC_TRANSPORT=aiohttp \
    PATH=/opt/venv/bin:$PATH

# Install curl for the container health check.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Create the writable application directory before copying source files.
RUN useradd --create-home --shell /bin/bash appuser \
    && install -d -o appuser -g appuser /app

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --chown=appuser:appuser langgraph_agent_toolkit/ ./langgraph_agent_toolkit/

# Run as a non-root user.
USER appuser

EXPOSE 8080

# Check readiness without routing the probe through a proxy.
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl --fail --silent --show-error --noproxy '*' --max-time 5 \
        "http://127.0.0.1:${LANGGRAPH_PORT:-${PORT:-8080}}/health/ready"

# Gunicorn worker settings. See `run_api.py`.
# `--workers`: set the number of worker processes.
# `--preload_app`: load the app before forking workers.
# `--timeout`: stop a silent worker. This is not the async request deadline.
# `--graceful_timeout`: bound request draining before a worker is forced to stop.
# `--max_requests[_jitter]`: recycle workers to limit memory growth.
CMD ["python", "langgraph_agent_toolkit/run_api.py", \
    "--runner_type", "gunicorn", \
    "--workers", "1", \
    "--preload_app", \
    "--timeout", "120", \
    "--graceful_timeout", "30"]

# Kubernetes health-check endpoints:
# - `/health/live`: process liveness.
# - `/health/ready`: traffic readiness.
# - `/health/startup`: initialization status.
# - `/health/db`: database-pool status.
