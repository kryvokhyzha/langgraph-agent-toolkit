# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Builder: install the minimal Streamlit client deps into an isolated venv.
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_PROJECT_ENVIRONMENT=/opt/venv

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --only-group client

# ---------------------------------------------------------------------------
# Runtime: minimal image with just the venv + application source.
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS runtime

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY langgraph_agent_toolkit/ ./langgraph_agent_toolkit/

# Run as a non-root user.
RUN useradd --create-home --shell /bin/bash appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

CMD ["streamlit", "run", "langgraph_agent_toolkit/run_app.py", \
    "--server.headless=true", \
    "--server.address=0.0.0.0", \
    "--server.port=8501", \
    "--server.fileWatcherType=none", \
    "--browser.gatherUsageStats=false"]
