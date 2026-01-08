FROM python:3.13-slim

WORKDIR /app

ENV UV_PROJECT_ENVIRONMENT="/usr/local/"
ENV UV_COMPILE_BYTECODE=1
ENV PYTHONPATH="/app"

# Install uv from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./

# -----------------------------------------------
# ------ Use gunicorn as the backend server ------
# -----------------------------------------------

COPY langgraph_agent_toolkit/ ./langgraph_agent_toolkit/
RUN uv sync --frozen --no-install-project --no-dev --extra gunicorn-backend --extra all-observability --extra openai \
    && rm -rf /root/.cache

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# Gunicorn worker management configuration:
#
# --workers: Number of worker processes (recommend 2-4 per CPU core)
# --preload_app: Load app before forking workers
#   - Pros: Faster worker startup, shared memory (less RAM), agents initialized once
#   - Cons: Workers share state, code changes require full restart
#   - Recommended for production with stable code
#
# --timeout: Worker timeout in seconds (kills unresponsive workers)
#   - Set higher than your slowest request (agent calls can be slow)
#   - Workers that don't respond within this time are killed and restarted
#
# --graceful_timeout: Time for graceful shutdown
#   - Allows in-flight requests to complete before worker is killed
#
# --max_requests: Restart workers after N requests (prevents memory leaks)
# --max_requests_jitter: Randomize restart to avoid all workers restarting at once
#
# Worker failure handling:
# - Gunicorn master monitors all workers via heartbeat
# - If a worker crashes/times out, it's automatically restarted
# - With preload_app, new worker starts faster (no agent re-initialization)
#
CMD ["python", "langgraph_agent_toolkit/run_api.py", "run_api", \
    "--runner_type", "gunicorn", \
    "--workers", "1", \
    "--preload_app", \
    "--timeout", "120", \
    "--graceful_timeout", "30"]

# ====================================================
# Health check endpoints for Kubernetes:
# - /health/live   - Liveness probe (is process alive?)
# - /health/ready  - Readiness probe (can accept traffic?)
# - /health/startup - Startup probe (has initialization completed?)
# - /health/db     - Database pool health (for monitoring)
# ====================================================
