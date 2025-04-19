FROM python:3.12.10-slim

WORKDIR /app

ENV UV_PROJECT_ENVIRONMENT="/usr/local/"
ENV UV_COMPILE_BYTECODE=1
ENV PYTHONPATH="/app"

COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv
RUN uv sync --frozen --no-install-project --no-dev

COPY langgraph_agent_toolkit/ ./langgraph_agent_toolkit/

CMD ["python", "langgraph_agent_toolkit/run_service.py"]
