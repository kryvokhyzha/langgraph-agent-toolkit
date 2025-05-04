<div align="center">
   <img alt="LangGraph Agent Toolkit Logo" src="https://github.com/kryvokhyzha/langgraph-agent-toolkit/blob/main/docs/media/logo.svg?raw=true" width="300">
</div>

---

# 🧰 LangGraph Agent Toolkit

|            |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CI/Testing | [![build status](https://github.com/kryvokhyzha/langgraph-agent-toolkit/actions/workflows/test.yml/badge.svg)](https://github.com/kryvokhyzha/langgraph-agent-toolkit/actions/workflows/test.yml) [![docs status](https://github.com/kryvokhyzha/langgraph-agent-toolkit/actions/workflows/sphinx.yml/badge.svg)](https://github.com/kryvokhyzha/langgraph-agent-toolkit/actions/workflows/sphinx.yml) [![codecov](https://codecov.io/gh/kryvokhyzha/langgraph-agent-toolkit/graph/badge.svg?token=OHSACTNSWZ)](https://codecov.io/gh/kryvokhyzha/langgraph-agent-toolkit) |
| Package    | [![PyPI version](https://img.shields.io/pypi/v/langgraph-agent-toolkit.svg)](https://pypi.org/project/langgraph-agent-toolkit/) [![PyPI Downloads](https://img.shields.io/pypi/dm/langgraph-agent-toolkit.svg)](https://pypi.org/project/langgraph-agent-toolkit/) [![Python Version](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fkryvokhyzha%2Flanggraph-agent-toolkit%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)](https://github.com/kryvokhyzha/langgraph-agent-toolkit/blob/main/pyproject.toml)           |
| Meta       | [![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff) [![GitHub License](https://img.shields.io/github/license/kryvokhyzha/langgraph-agent-toolkit)](https://github.com/kryvokhyzha/langgraph-agent-toolkit/blob/main/LICENSE)                                                                                                                                                                                                                                                                               |

<!-- [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_red.svg)](https://langgraph-agent-toolkit.streamlit.app/) -->

## 📋 Introduction

A comprehensive toolkit for building, deploying, and managing AI agents using
LangGraph, FastAPI, and Streamlit. It provides a production-ready framework for
creating conversational AI agents with features like multi-provider LLM support,
streaming responses, observability, memory and prompt management.

### What is langGraph-agent-toolkit?

The langgraph-agent-toolkit is a full-featured framework for developing and
deploying AI agent services. Built on the foundation of:

- **[LangGraph](https://langchain-ai.github.io/langgraph/)** for agent creation
  with advanced flows and human-in-the-loop capabilities
- **[FastAPI](https://fastapi.tiangolo.com/)** for robust, high-performance API
  services with streaming support
- **[Streamlit](https://streamlit.io/)** for intuitive user interfaces

Key components include:

- Data structures and settings built with
  **[Pydantic](https://github.com/pydantic/pydantic)**
- **[LiteLLM](https://github.com/BerriAI/litellm)** proxy for universal
  multi-provider LLM support
- Comprehensive memory management and persistence using PostgreSQL/SQLite
- Advanced observability tooling via Langfuse and Langsmith
- Modular architecture allowing customization while maintaining a consistent
  application structure

Whether you're building a simple chatbot or complex multi-agent system, this
toolkit provides the infrastructure to develop, test, and deploy your
LangGraph-based agents with confidence.

## 📑 Contents

- [Introduction](#-introduction)
- [Quickstart](#-quickstart)
- [Installation Options](#-installation-options)
- [Architecture](#architecture)
- [Key Features](#-key-features)
- [Environment Setup](#environment-setup)
- [Project Structure](#-project-structure)
- [Setup and Usage](#setup-and-usage)
- [Documentation](#-documentation)
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

   ℹ️ For more details on installation options, see the
   [Installation Documentation](docs/installation.rst).

4. **Option 3: Run with Docker**

   ```sh
   docker compose watch
   ```

<a name="installation-options"></a>

## 📦 Installation Options

The toolkit supports multiple installation options using "extras" to include
just the dependencies you need.

For detailed installation instructions and available extras, see the
[Installation Documentation](docs/installation.rst).

<a name="architecture"></a>

## 🏗️ Architecture

<img src="https://github.com/kryvokhyzha/langgraph-agent-toolkit/blob/main/docs/media/agent_architecture.png?raw=true" width="800">

<a name="key-features"></a>

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
   - Supports different runners (unicorn, gunicorn, mangum, azure functions)

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

For more details on features, see the [Usage Documentation](docs/usage.rst).

<a name="environment-setup"></a>

## ⚙️ Environment Setup

For detailed environment setup instructions, including creating your `.env` file
and configuring LiteLLM, see the
[Environment Setup Documentation](docs/environment_setup.rst).

<a name="project-structure"></a>

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

For detailed setup and usage instructions, including building your own agent,
Docker setup, using the AgentClient, and local development, see the
[Usage Documentation](docs/usage.rst).

<a name="documentation"></a>

## 📚 Documentation

Full documentation is available in the `docs/` directory and includes:

- [Installation Guide](docs/installation.rst)
- [Environment Setup](docs/environment_setup.rst)
- [Usage Guide](docs/usage.rst)

<a name="useful-resources"></a>

## 📚 Useful Resources

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/concepts/low_level/#multiple-schemas)
- [LangGraph Memory Concept](https://langchain-ai.github.io/langgraph/concepts/memory/)
- [LangGraph Memory Persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/#memory)
- [How to edit graph state](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/edit-graph-state/)
- [How to create tools in Langchain](https://python.langchain.com/docs/how_to/custom_tools/)
- [Simple Serverless FastAPI with AWS Lambda](https://www.deadbear.io/simple-serverless-fastapi-with-aws-lambda/)

<a name="development-and-contributing"></a>

## 👥 Development and Contributing

Thank you for considering contributing to `Langgraph Agent Toolkit`! We
encourage the community to post Issues and Pull Requests.

Before you get started, please see our [Contribution Guide](CONTRIBUTING.md).

<a name="license"></a>

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for
details.
