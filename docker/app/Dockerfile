# syntax=docker/dockerfile:1

# Use a patched interpreter and pip certificate bundle in both stages.
FROM python:3.15.0-slim AS base
RUN --mount=from=ghcr.io/astral-sh/uv:0.13.6,source=/uv,target=/uv \
    /uv pip install --system --no-cache "pip>=26.2.2"

# ---------------------------------------------------------------------------
# Builder: install the minimum Streamlit client dependencies in an isolated virtual environment.
# ---------------------------------------------------------------------------
FROM base AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_PROJECT_ENVIRONMENT=/opt/venv

COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
# Use the source-only client group to omit the service dependency set.
# Published package users can install the `ui` extra instead.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --only-group client

# ---------------------------------------------------------------------------
# Runtime: include only the virtual environment and application source.
# ---------------------------------------------------------------------------
FROM base AS runtime

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH

# Install curl for the container health check.
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create the writable application directory before copying source files.
RUN useradd --create-home --shell /bin/bash appuser \
    && install -d -o appuser -g appuser /app

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --chown=appuser:appuser langgraph_agent_toolkit/ ./langgraph_agent_toolkit/

# Run as a non-root user.
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD ["curl", "--fail", "--silent", "--show-error", "--noproxy", "*", "--max-time", "5", "http://127.0.0.1:8501/_stcore/health"]

CMD ["streamlit", "run", "langgraph_agent_toolkit/run_app.py", \
    "--server.headless=true", \
    "--server.address=0.0.0.0", \
    "--server.port=8501", \
    "--server.fileWatcherType=none", \
    "--browser.gatherUsageStats=false"]
