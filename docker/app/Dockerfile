FROM python:3.12.10-slim

WORKDIR /app

ENV UV_PROJECT_ENVIRONMENT="/usr/local/"
ENV UV_COMPILE_BYTECODE=1
ENV PYTHONPATH="/app"

COPY pyproject.toml uv.lock ./

RUN pip install --no-cache-dir uv
RUN uv sync --frozen --no-install-project --only-group client \
    && rm -rf /root/.cache

COPY langgraph_agent_toolkit/ ./langgraph_agent_toolkit/

CMD ["streamlit", "run", "langgraph_agent_toolkit/streamlit_app.py"]
