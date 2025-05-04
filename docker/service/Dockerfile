FROM python:3.12.10-slim

WORKDIR /app

ENV UV_PROJECT_ENVIRONMENT="/usr/local/"
ENV UV_COMPILE_BYTECODE=1
ENV PYTHONPATH="/app"

COPY pyproject.toml uv.lock ./
COPY langgraph_agent_toolkit/ ./langgraph_agent_toolkit/

RUN pip install --no-cache-dir uv

# -----------------------------------------------
# ------ Use unicorn as the backend server ------
# -----------------------------------------------

RUN uv sync --frozen --no-install-project --no-dev --extra uvicorn-backend --extra all-observability --extra openai \
    && rm -rf /root/.cache
CMD ["python", "langgraph_agent_toolkit/run_service.py", "run_service", "--runner_type", "uvicorn"]

# ====================================================

# ------------------------------------------------
# ------ Use gunicorn as the backend server ------
# ------------------------------------------------

# RUN uv sync --frozen --no-install-project --no-dev --extra gunicorn-backend --extra all-observability --extra openai \
#     && rm -rf /root/.cache
# CMD ["python", "langgraph_agent_toolkit/run_service.py", "run_service", "--runner_type", "gunicorn", "--workers", "4"]

# ====================================================
