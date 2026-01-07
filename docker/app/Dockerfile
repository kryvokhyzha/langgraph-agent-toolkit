FROM python:3.13-slim

WORKDIR /app

ENV UV_PROJECT_ENVIRONMENT="/usr/local/"
ENV UV_COMPILE_BYTECODE=1
ENV PYTHONPATH="/app"

# Install uv from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-install-project --only-group client \
    && rm -rf /root/.cache

COPY langgraph_agent_toolkit/ ./langgraph_agent_toolkit/

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

CMD ["streamlit", "run", "langgraph_agent_toolkit/run_app.py", \
    "--server.headless=true", \
    "--server.address=0.0.0.0", \
    "--server.port=8501", \
    "--server.fileWatcherType=none", \
    "--browser.gatherUsageStats=false"]
