# 🧰 Langgraph Agent Toolkit

[![build status](https://github.com/kryvokhyzha/langgraph-agent-toolkit/actions/workflows/test.yml/badge.svg)](https://github.com/kryvokhyzha/langgraph-agent-toolkit/actions/workflows/test.yml)
[![PyPI version](https://img.shields.io/pypi/v/langgraph-agent-toolkit.svg)](https://pypi.org/project/langgraph-agent-toolkit/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/langgraph-agent-toolkit.svg)](https://pypi.org/project/langgraph-agent-toolkit/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![codecov](https://codecov.io/gh/kryvokhyzha/langgraph-agent-toolkit/graph/badge.svg?token=OHSACTNSWZ)](https://codecov.io/gh/kryvokhyzha/langgraph-agent-toolkit)
[![Python Version](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fkryvokhyzha%2Flanggraph-agent-toolkit%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)](https://github.com/kryvokhyzha/langgraph-agent-toolkit/blob/main/pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/kryvokhyzha/langgraph-agent-toolkit/blob/main/LICENSE)

<!-- [![GitHub License](https://img.shields.io/github/license/kryvokhyzha/langgraph-agent-toolkit)](https://github.com/kryvokhyzha/langgraph-agent-toolkit/blob/main/LICENSE) -->

<!-- [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_red.svg)](https://langgraph-agent-toolkit.streamlit.app/) -->

## 📋 Introduction

A full toolkit for running an AI agent service built with:

- **[LangGraph](https://langchain-ai.github.io/langgraph/)** for agent creation
- **[FastAPI](https://fastapi.tiangolo.com/)** for API service
- **[Streamlit](https://streamlit.io/)** for user interface

Features include:

- Data structures and settings built with
  **[Pydantic](https://github.com/pydantic/pydantic)**
- **[LiteLLM](https://github.com/BerriAI/litellm)** proxy for multi-provider LLM
  support
- Complete template for building and deploying your own LangGraph-based agents

## 📑 Contents

- [Introduction](#-introduction)
- [Quickstart](#-quickstart)
- [Installation Options](#-installation-options)
- [Architecture](#architecture)
- [Key Features](#-key-features)
- [Environment Setup](#environment-setup)
  - [Creating Your `.env` File](#-creating-your-env-file)
  - [LiteLLM Configuration](#-litellm-configuration)
- [Project Structure](#-project-structure)
- [Setup and Usage](#setup-and-usage)
  - [Building Your Own Agent](#-building-your-own-agent)
  - [Docker Setup](#-docker-setup)
  - [Using the AgentClient](#-using-the-agentclient)
  - [Development with LangGraph Studio](#-development-with-langgraph-studio)
  - [Local Development Without Docker](#-local-development-without-docker)
- [Useful Resources](#-useful-resources)
- [Development and Contributing](#-development-and-contributing)
- [License](#-license)

## 🚀 Quickstart

1. Create a `.env` file based on [`.env.example`](./.env.example)

2. **Option 1: Run with Python from source**

   ```sh
   # Install dependencies
   pip install uv
   uv sync --frozen
   source .venv/bin/activate

   # Start the service
   python langgraph_agent_toolkit/run_service.py

   # In another terminal
   source .venv/bin/activate
   streamlit run langgraph_agent_toolkit/streamlit_app.py
   ```

3. **Option 2: Run with Python from PyPi repository**

   ```sh
   pip install langgraph-agent-toolkit
   ```

   ℹ️ You can check available extras in
   [Installation Options](#-installation-options) section or directly in
   [pyproject.toml](pyproject.toml) file.

4. **Option 3: Run with Docker**

   ```sh
   docker compose watch
   ```

## 📦 Installation Options

The toolkit supports multiple installation options using "extras" to include
just the dependencies you need:

### Available Extras

```sh
# LLM Provider Extras
pip install "langgraph-agent-toolkit[openai,uvicorn-backend,langfuse]"              # OpenAI, Uvicorn backend, and Langfuse observability
pip install "langgraph-agent-toolkit[anthropic,aws-backend,langsmith]"              # Anthropic, AWS Lambda backend, and Langsmith observability

# Also available: google-vertexai, aws, ollama, groq, deepseek
pip install "langgraph-agent-toolkit[all-llms,all-backends,all-observability]"      # All LLM providers, all backends, and all observability platforms

# Client-Only Installation
pip install "langgraph-agent-toolkit[client]"                                       # Just the client and Streamlit app
```

<a name="architecture"></a>

## 🏗️ Architecture

<!-- <img src="docs/media/agent_architecture.png" width="800"> -->
<img src="https://github.com/kryvokhyzha/langgraph-agent-toolkit/blob/main/docs/media/agent_architecture.png?raw=true" width="800">

## ✨ Key Features

1. **LangGraph Integration**

   - Latest LangGraph v0.3 features
   - Human-in-the-loop with `interrupt()`
   - Flow control with `Command` and `langgraph-supervisor`

2. **API Service**

   - FastAPI with streaming and non-streaming endpoints
   - Support for both token-based and message-based streaming
   - Multiple agent support with URL path routing
   - Available agents and models listed at `/info` endpoint
   - Supports different runners:
     - unicorn
     - gunicorn
     - ⚠️ mangum (AWS Lambda) - _experimental_
     - ⚠️ azure functions - _experimental_

3. **Developer Experience**

   - Asynchronous design with async/await
   - Docker configuration with live reloading
   - Comprehensive testing suite

4. **Enterprise Components**
   - Configurable PostgreSQL/SQLite connection pools
   - Observability via Langfuse and Langsmith
   - User feedback system
   - Prompt management system
   - LiteLLM proxy integration

<a name="environment-setup"></a>

## ⚙️ Environment Setup

### 📝 Creating Your `.env` File

1. Copy the example configuration:

   ```sh
   cp .env.example .env
   ```

2. Configure these essential sections:

   **🔑 LLM API Configuration**

   ```env
   # OpenAI Settings
   OPENAI_API_KEY=sk-xxxxxxxxxxxxx
   OPENAI_API_BASE_URL=http://litellm:4000/v1      # Can be OpenAI API or LiteLLM proxy
   OPENAI_API_VERSION=2025-01-01
   OPENAI_MODEL_NAME=gpt-4o-mini

   # Azure OpenAI Settings (Optional)
   AZURE_OPENAI_API_KEY=
   AZURE_OPENAI_ENDPOINT=
   AZURE_OPENAI_API_VERSION=
   AZURE_OPENAI_MODEL_NAME=
   AZURE_OPENAI_DEPLOYMENT_NAME=

   # Anthropic Settings (Optional)
   ANTHROPIC_API_KEY=
   ANTHROPIC_MODEL_NAME=

   # Other provider settings are also available:
   # - Google VertexAI
   # - Google GenAI
   # - AWS Bedrock
   # - DeepSeek
   # - Ollama
   ```

   **🔀 Multi-Provider Model Configuration**

   ```env
   # Configure multiple models from different providers in a single environment variable
   MODEL_CONFIGS={\
     "gpt4o": {\
       "provider": "azure_openai",\
       "name": "gpt-4o",\
       "api_key": "azure-key-123",\
       "endpoint": "https://your-resource.openai.azure.com/",\
       "api_version": "2023-05-15",\
       "deployment": "gpt4o-deployment",\
       "temperature": 0.7\
     },\
     "gpt4o-mini": {\
       "provider": "azure_openai",\
       "name": "gpt-4o-mini",\
       "api_key": "azure-key-123",\
       "endpoint": "https://your-resource.openai.azure.com/",\
       "api_version": "2023-05-15",\
       "deployment": "gpt4o-mini-deployment"\
     },\
     "gemini": {\
       "provider": "google_genai",\
       "name": "gemini-pro",\
       "api_key": "google-key-123",\
       "temperature": 0.7\
     }\
   }
   ```

   This configuration allows you to:

   - Define multiple models with different providers in one place
   - Reference them by logical names in your application
   - Set provider-specific parameters for each model
   - Switch between models without changing code

   **🗄️ Database Configuration**

   ```env
   POSTGRES_HOST=postgres
   POSTGRES_PORT=5432
   POSTGRES_USER=postgres
   POSTGRES_PASSWORD=postgres
   POSTGRES_DB=postgres
   POSTGRES_SCHEMA=public
   ```

   **📊 Observability Configuration**

   ```env
   # Option 1: Langfuse
   LANGFUSE_HOST=http://langfuse-web:3000
   LANGFUSE_PUBLIC_KEY=lf-pk-1234567890
   LANGFUSE_SECRET_KEY=lf-sk-1234567890
   ```

   ```env
   # Option 2: LangSmith
   LANGSMITH_TRACING=true
   LANGSMITH_API_KEY=api-key-xxxxxxxxcxxxxxx
   LANGSMITH_PROJECT=default
   LANGSMITH_ENDPOINT=https://api.smith.langchain.com
   ```

3. Customize other sections as needed (Redis, memory options, logging)

> [!WARNING]  
> The `.env` file contains sensitive credentials and should never be committed
> to version control. It's already included in `.gitignore`.

### 🔄 LiteLLM Configuration

1. Create your configuration:

   ```sh
   cp configs/litellm/config.example.yaml configs/litellm/config.yaml
   ```

2. Edit `configs/litellm/config.yaml` to include your models and credentials:

   **🤖 model_list**: Define available models

   ```yaml
   model_list:
     - model_name: gpt-4o-mini # This name is used when selecting the model in your app
       litellm_params:
         model: azure/gpt-4o-mini # Format for the underlying LiteLLM model
         litellm_credential_name: your_azure_credential # References credentials defined below
         rpm: 6 # Rate limit (requests per minute)
   ```

   **🔐 credential_list**: Store API keys and endpoints

   ```yaml
   credential_list:
     - credential_name: your_azure_credential
       credential_values:
         api_key: "your-api-key-here" # Best practice: use os.environ/AZURE_API_KEY
         api_base: "https://your-azure-endpoint.openai.azure.com/"
         api_version: "2025-01-01-preview"
   ```

   **🔀 router_settings**: Configure routing

   ```yaml
   router_settings:
     routing_strategy: simple-shuffle
     redis_host: redis
     redis_password: os.environ/REDIS_AUTH
   ```

3. Setup service environment files:

   Each service has its own environment file:

   ```sh
   cp configs/litellm/.litellm.env.example configs/litellm/.litellm.env
   cp configs/redis/.redis.env.example configs/redis/.redis.env
   cp configs/postgres/.postgres.env.example configs/postgres/.postgres.env
   cp configs/minio/.minio.env.example configs/minio/.minio.env
   cp configs/clickhouse/.clickhouse.env.example configs/clickhouse/.clickhouse.env
   cp configs/langfuse/.langfuse.env.example configs/langfuse/.langfuse.env
   ```

   For example, edit LiteLLM environment:

   ```env
   LITELLM_MASTER_KEY=sk-your-master-key  # Create a strong key here
   LITELLM_ENVIRONMENT=development
   DATABASE_URL="postgresql://postgres:postgres@postgres:5432/litellm"
   STORE_MODEL_IN_DB=True
   ```

> [!NOTE]  
> LiteLLM relies on Redis for request caching and rate limiting.

## 📂 Project Structure

The repository contains:

- `langgraph_agent_toolkit/agents/blueprints/`: Agent definitions
- `langgraph_agent_toolkit/agents/agent_executor.py`: Agent execution control
- `langgraph_agent_toolkit/schema/`: Protocol schema definitions
- `langgraph_agent_toolkit/core/`: Core modules (LLM, memory, settings)
- `langgraph_agent_toolkit/service/service.py`: FastAPI service
- `langgraph_agent_toolkit/client/client.py`: Service client
- `langgraph_agent_toolkit/streamlit_app.py`: Chat interface
- `docker/`: Docker configurations
- `tests/`: Test suite

<a name="setup-and-usage"></a>

## 🛠️ Setup and Usage

1. Clone the repository:

   ```sh
   git clone https://github.com/kryvokhyzha/langgraph-agent-toolkit
   cd langgraph-agent-toolkit
   ```

2. Set up your environment (see Environment Setup section)

3. Run the service (with Python or Docker)

### 🔧 Building Your Own Agent

To customize the agent:

1. Add your agent to `langgraph_agent_toolkit/agents/blueprints/`
2. Register it in `AGENT_PATHS` list in
   `langgraph_agent_toolkit/core/settings.py`
3. Optionally customize the Streamlit interface in `streamlit_app.py`

### 🐳 Docker Setup

The `docker-compose.yaml` defines these services with enhanced security:

- `backend-agent-service`: FastAPI service
- `frontend-streamlit-app`: Streamlit chat interface
- `postgres`: Database storage
- `redis`: Cache and message broker
- `minio`: Object storage
- `clickhouse`: Analytics database
- `langfuse-web` & `langfuse-worker`: Observability
- `litellm`: LLM proxy server

Using [docker compose watch](https://docs.docker.com/compose/file-watch/)
enables live reloading:

1. Ensure Docker and Docker Compose (>=2.23.0) are installed

2. Launch services:

   ```sh
   docker compose watch
   ```

3. Access endpoints:

   - 🖥️ Streamlit app: `http://0.0.0.0:8501`
   - 🔌 Agent API: `http://0.0.0.0:8080`
   - 📚 API docs: `http://0.0.0.0:8080/docs`
   - 📊 Langfuse dashboard: `http://0.0.0.0:3000`
   - 🤖 LiteLLM API: `http://0.0.0.0:4000` (accessible from any host)

4. Stop services:

   ```sh
   docker compose down
   ```

> [!NOTE]  
> If you modify `pyproject.toml` or `uv.lock`, rebuild with
> `docker compose up --build`

### 🔄 Using the AgentClient

The toolkit includes `AgentClient` for interacting with the agent service:

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

See `langgraph_agent_toolkit/run_client.py` for more examples.

### 💻 Development with LangGraph Studio

The project works with
[LangGraph Studio](https://github.com/langchain-ai/langgraph-studio):

1. Install LangGraph Studio
2. Add your `.env` file to the root directory
3. Launch LangGraph Studio pointing at the project root
4. Customize `langgraph.json` as needed

### 🧑‍💻 Local Development Without Docker

1. Set up a Python environment:

   ```sh
   pip install uv
   uv sync --frozen
   source .venv/bin/activate
   ```

2. Create and configure your `.env` file

3. Run the FastAPI server:

   ```sh
   python langgraph_agent_toolkit/run_service.py
   ```

4. Run the Streamlit app in another terminal:

   ```sh
   streamlit run langgraph_agent_toolkit/streamlit_app.py
   ```

5. Access the Streamlit interface (usually at `http://localhost:8501`)

## 📚 Useful Resources

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/concepts/low_level/#multiple-schemas)
- [LangGraph Memory Concept](https://langchain-ai.github.io/langgraph/concepts/memory/)
- [LangGraph Memory Persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/#memory)
- [How to edit graph state](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/edit-graph-state/)
- [How to create tools in Langchain](https://python.langchain.com/docs/how_to/custom_tools/)
- [Simple Serverless FastAPI with AWS Lambda](https://www.deadbear.io/simple-serverless-fastapi-with-aws-lambda/)

## 👥 Development and Contributing

Thank you for considering contributing to `Langgraph Agent Toolkit`! We
encourage the community to post Issues and Pull Requests.

Before you get started, please see our [Contribution Guide](CONTRIBUTING.md).

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for
details.
