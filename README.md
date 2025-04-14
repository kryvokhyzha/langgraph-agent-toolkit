# 🧰 Langgraph Agent Toolkit

[![build status](https://github.com/kryvokhyzha/langgraph-agent-toolkit/actions/workflows/test.yml/badge.svg)](https://github.com/kryvokhyzha/langgraph-agent-toolkit/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/kryvokhyzha/langgraph-agent-toolkit/graph/badge.svg?token=OHSACTNSWZ)](https://codecov.io/gh/kryvokhyzha/langgraph-agent-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Python Version](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fkryvokhyzha%2Flanggraph-agent-toolkit%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)](https://github.com/kryvokhyzha/langgraph-agent-toolkit/blob/main/pyproject.toml)
[![GitHub License](https://img.shields.io/github/license/kryvokhyzha/langgraph-agent-toolkit)](https://github.com/kryvokhyzha/langgraph-agent-toolkit/blob/main/LICENSE)

<!-- [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_red.svg)](https://langgraph-agent-toolkit.streamlit.app/) -->

A full toolkit for running an AI agent service built with LangGraph, FastAPI and
Streamlit:

- It includes a [LangGraph](https://langchain-ai.github.io/langgraph/) agent, a
  [FastAPI](https://fastapi.tiangolo.com/) service to serve it, a client to
  interact with the service, and a [Streamlit](https://streamlit.io/) app that
  uses the client to provide a chat interface.
- Data structures and settings are built with
  [Pydantic](https://github.com/pydantic/pydantic).
- Instead of implementing dozens of classes to support various models, we use
  the [LiteLLM](https://github.com/BerriAI/litellm) proxy server to work with
  various LLM providers.
- This project offers a template for you to easily build and run your own agents
  using the `LangGraph` framework. It demonstrates a complete setup from agent
  definition to user interface, making it easier to get started with
  LangGraph-based projects by providing a full, robust toolkit.

## Environment Setup

Before running the toolkit, you need to set up your environment variables by
creating a `.env` file in the project root directory.

### Creating Your `.env` File

1. Copy the `.env.example` file to create your own `.env` file:

   ```sh
   cp .env.example .env
   ```

2. Open the `.env` file in your favorite text editor and configure the following
   essential sections:

   - **LLM API Configuration**: At minimum, include one of these:

     ```env
     # LiteLLM configuration
     COMPATIBLE_MODEL="gpt4o"                        # LiteLLM model name
     COMPATIBLE_BASE_URL=http://litellm:4000/v1      # LiteLLM endpoint
     COMPATIBLE_API_KEY=sk-any-string                # LiteLLM master key
     ```

   - **Database Configuration**: For Postgres storage:

     ```env
     POSTGRES_HOST=postgres
     POSTGRES_PORT=5432
     POSTGRES_USER=postgres
     POSTGRES_PASSWORD=postgres
     POSTGRES_DB=postgres
     POSTGRES_SCHEMA=public
     ```

   - **Observability Configuration**:

     ```env
     LANGFUSE_HOST=http://langfuse-web:3000
     LANGFUSE_PUBLIC_KEY=lf-pk-1234567890
     LANGFUSE_SECRET_KEY=lf-sk-1234567890
     ```

     or

     ```env
     LANGSMITH_TRACING=true
     LANGSMITH_API_KEY=api-key-xxxxxxxxcxxxxxx
     LANGSMITH_PROJECT=default
     LANGSMITH_ENDPOINT=https://api.smith.langchain.com
     ```

3. Customize other sections as needed for your specific use case, including
   Redis configuration, memory options, and logging levels.

4. Save your `.env` file.

> [!WARNING]\
> The `.env` file contains sensitive credentials and should never be committed to
> version control. It's already included in `.gitignore`.

### LiteLLM Configuration

This project includes LiteLLM as a proxy to manage access to multiple LLM
providers. To configure LiteLLM:

1. Review the sample configuration in `configs/litellm/config.example.yaml` to
   understand the structure.

2. Create your own configuration by copying and modifying `config.example.yaml`:

   ```sh
   cp configs/litellm/config.example.yaml configs/litellm/config.yaml
   ```

3. Edit `configs/litellm/config.yaml` to include your preferred language models
   and credentials. Key sections to configure:

   - **model_list**: Defines the available models and their settings

     ```yaml
     model_list:
       - model_name: gpt-4o-mini # This name is used when selecting the model in your app
         litellm_params:
           model: azure/gpt-4o-mini # Format for the underlying LiteLLM model
           litellm_credential_name: your_azure_credential # References credentials defined below
           rpm: 6 # Rate limit (requests per minute)
     ```

   - **credential_list**: Stores your API keys and endpoints

     ```yaml
     credential_list:
       - credential_name: your_azure_credential
         credential_values:
           api_key: "your-api-key-here" # Best practice: use os.environ/AZURE_API_KEY
           api_base: "https://your-azure-endpoint.openai.azure.com/"
           api_version: "2025-01-01-preview"
     ```

   - **router_settings**: Configuration for routing between models

     ```yaml
     router_settings:
       routing_strategy: simple-shuffle
       redis_host: redis
       redis_password: os.environ/REDIS_AUTH
     ```

4. Create and configure the `.litellm.env` file:

   ```sh
   cp configs/litellm/.litellm.env.example configs/litellm/.litellm.env
   ```

5. Edit `.litellm.env` to include:

   ```env
   LITELLM_MASTER_KEY=sk-your-master-key  # Create a strong key here
   LITELLM_ENVIRONMENT=development
   DATABASE_URL="postgresql://postgres:postgres@postgres:5432/litellm"
   STORE_MODEL_IN_DB=True
   ```

> **Note**: LiteLLM relies on Redis for request caching and rate limiting. Make
> sure Redis is properly configured in your environment.

## Overview

### Quickstart

1. Create a `.env` file in the root directory, based on the
   [`.env.example`](./.env.example) file.

2. Run directly in python

   ```sh
   # uv is recommended but "pip install ." also works
   pip install uv
   uv sync --frozen
   # "uv sync" creates .venv automatically
   source .venv/bin/activate
   python langgraph_agent_toolkit/run_service.py

   # In another shell
   source .venv/bin/activate
   streamlit run langgraph_agent_toolkit/streamlit_app.py
   ```

3. (optional) Run with docker

   ```sh
   docker compose watch
   ```

### Architecture Diagram

<img src="docs/media/agent_architecture.png" width="600">

### Key Features

1. **LangGraph Agent and latest features**: A customizable agent built using the
   LangGraph framework. Implements the latest LangGraph v0.3 features including
   human in the loop with `interrupt()`, and flow control with `Command`, and
   `langgraph-supervisor`.
2. **FastAPI Service**: Serves the agent with both streaming and non-streaming
   endpoints.
3. **Advanced Streaming**: A novel approach to support both token-based and
   message-based streaming.
4. **Streamlit Interface**: Provides a user-friendly chat interface for
   interacting with the agent.
5. **Multiple Agent Support**: Run multiple agents in the service and call by
   URL path. Available agents and models are described in `/info`
6. **Asynchronous Design**: Utilizes async/await for efficient handling of
   concurrent requests.
7. **Docker Support**: Includes Dockerfiles and a docker compose file for easy
   development and deployment.
8. **Testing**: Includes robust unit and integration tests for the full repo.
9. **Configurable Connection Pools**: Custom `PostgreSQL` connection pool
   settings for database performance tuning.
10. **Different Memory Options**: Supports different memory options including
    `PostgreSQL`, and `Sqlite`.
11. **Observability**: Integrates with `Langfuse` and `Langsmith` for
    observability and monitoring.
12. **Feedback Mechanism**: Includes a star-based feedback system integrated
    with observability platform.
13. **Prompt Management**: Supports prompt management integrated with
    observability platform and local file system.
14. **LiteLLM Integration**: Includes a LiteLLM proxy server for managing
    multiple LLM providers.

### Key Files

The repository is structured as follows:

- `langgraph_agent_toolkit/agents/blueprints/`: Defines several agents with
  different capabilities
- `langgraph_agent_toolkit/agents/agent_executor.py`: Defines the agent
  executor. Controls the agent execution and error handling.
- `langgraph_agent_toolkit/schema/`: Defines the protocol schema
- `langgraph_agent_toolkit/core/`: Core modules including LLM definition,
  memory, observability and settings
- `langgraph_agent_toolkit/service/service.py`: FastAPI service to serve the
  agents
- `langgraph_agent_toolkit/client/client.py`: Client to interact with the agent
  service
- `langgraph_agent_toolkit/streamlit_app.py`: Streamlit app providing a chat
  interface
- `docker`: Dockerfiles for the agent service and Streamlit app
- `tests/`: Unit and integration tests

## Setup and Usage

1. Clone the repository:

   ```sh
   git clone https://github.com/kryvokhyzha/langgraph-agent-toolkit
   cd langgraph-agent-toolkit
   ```

2. Set up environment variables: Create a `.env` file in the root directory. At
   least one LLM API key or configuration is required. See the
   [`.env.example`](./.env.example) file for a full list of available
   environment variables, including a variety different parameters.

3. You can now run the agent service and the Streamlit app locally, either with
   Docker or just using Python. The Docker setup is recommended for simpler
   environment setup and immediate reloading of the services when you make
   changes to your code.

### Building or customizing your own agent

To customize the agent for your own use case:

1. Add your new agent to the `langgraph_agent_toolkit/agents/blueprints`
   directory. You can copy an existing agent like those in the `blueprints`
   folder and modify it to change the agent's behavior and tools.
2. Add your new agent to the `AGENT_PATHS` list in
   `langgraph_agent_toolkit/core/settings.py`.
3. (optional) Adjust the Streamlit interface in
   `langgraph_agent_toolkit/streamlit_app.py` to match your agent's
   capabilities.

### Docker Setup

This project includes a Docker setup for easy development and deployment. The
`docker-compose.yaml` file defines the following services:

- `backend-agent-service`: The FastAPI service that serves the agent.
- `frontend-streamlit-app`: The Streamlit app that provides a chat interface for
  the agent.
- `postgres`: A PostgreSQL database for storing agent data and logs.
- `redis`: A Redis server for caching and message brokering.
- `minio`: A MinIO server for object storage.
- `clickhouse`: A ClickHouse server for analytics and data storage.
- `langfuse-web`: A Langfuse server for observability and monitoring.
- `langfuse-worker`: A worker for Langfuse to process events and logs.
- `litellm`: A LiteLLM proxy server for managing multiple LLM providers.

Each service is configured with appropriate resource limits and health checks to
ensure stability and performance.

For local development, we recommend using
[docker compose watch](https://docs.docker.com/compose/file-watch/). This
feature allows for a smoother development experience by automatically updating
your containers when changes are detected in your source code.

1. Make sure you have Docker and Docker Compose
   (>=[2.23.0](https://docs.docker.com/compose/release-notes/#2230)) installed
   on your system.

2. Build and launch the services in watch mode:

   ```sh
   docker compose watch
   ```

3. The services will now automatically update when you make changes to your
   code:

   - Changes in the relevant python files and directories will trigger updates
     for the relevant services.
   - NOTE: If you make changes to the `pyproject.toml` or `uv.lock` files, you
     will need to rebuild the services by running `docker compose up --build`.

4. Access the Streamlit app by navigating to `http://localhost:8501` in your web
   browser.

5. The agent service API will be available at `http://0.0.0.0:8080`. You can
   also use the OpenAPI docs at `http://0.0.0.0:8080/redoc`.

6. Other service endpoints:

   - Langfuse dashboard: `http://localhost:3000`
   - MinIO console: `http://localhost:9091`
   - LiteLLM API: `http://localhost:4000`
   - ClickHouse HTTP interface: `http://localhost:8123`

7. Use `docker compose down` to stop the services.

This setup allows you to develop and test your changes in real-time without
manually restarting the services.

### Building other apps on the AgentClient

The repo includes a generic `langgraph_agent_toolkit/client/client.AgentClient`
that can be used to interact with the agent service. This client is designed to
be flexible and can be used to build other apps on top of the agent. It supports
both synchronous and asynchronous invocations, and streaming and non-streaming
requests.

See the `langgraph_agent_toolkit/run_client.py` file for full examples of how to
use the `AgentClient`. A quick example:

```python
from client import AgentClient
client = AgentClient()

response = client.invoke("Tell me a brief joke?")
response.pretty_print()
# ================================== Ai Message ==================================
#
# A man walked into a library and asked the librarian, "Do you have any books on Pavlov's dogs and Schrödinger's cat?"
# The librarian replied, "It rings a bell, but I'm not sure if it's here or not."

```

### Development with LangGraph Studio

The agent supports
[LangGraph Studio](https://github.com/langchain-ai/langgraph-studio), a new IDE
for developing agents in LangGraph.

You can simply install LangGraph Studio, add your `.env` file to the root
directory as described above, and then launch LangGraph studio pointed at the
root directory. Customize `langgraph.json` as needed.

### Local development without Docker

You can also run the agent service and the Streamlit app locally without Docker,
just using a Python virtual environment.

1. Create a virtual environment and install dependencies:

   ```sh
   pip install uv
   uv sync --frozen
   source .venv/bin/activate
   ```

2. Set up environment variables: Create a `.env` file in the root directory. At
   least one LLM API key or configuration is required. See the
   [`.env.example`](./.env.example) file for a full list of available
   environment variables, including a variety different parameters.

3. Run the FastAPI server:

   ```sh
   python langgraph_agent_toolkit/run_service.py
   ```

4. In a separate terminal, run the Streamlit app:

   ```sh
   streamlit run langgraph_agent_toolkit/streamlit_app.py
   ```

5. Open your browser and navigate to the URL provided by Streamlit (usually
   `http://localhost:8501`).

## Useful resources

- [About graphs, states, schemas, multiple schemas, reducers, configuration, recursion limit, breakpoint, sub-graphs](https://langchain-ai.github.io/langgraph/concepts/low_level/#multiple-schemas)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Currently,
the tests need to be run using the local development without Docker setup. To
run the tests for the agent service:

1. Ensure you're in the project root directory and have activated your virtual
   environment.

2. Install the development dependencies and pre-commit hooks:

   ```sh
   pip install uv
   uv sync --frozen
   pre-commit install
   ```

3. Run the tests using pytest:

   ```sh
   pytest
   ```

## License

This project is licensed under the MIT License - see the LICENSE file for
details.
