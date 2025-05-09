Usage Guide
==========

Setup and Usage
--------------

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/kryvokhyzha/langgraph-agent-toolkit
      cd langgraph-agent-toolkit

2. Set up your environment (see Environment Setup section)

3. Run the service (with Python or Docker)

Building Your Own Agent
----------------------

To customize the agent:

1. Add your agent to ``langgraph_agent_toolkit/agents/blueprints/``
2. Register it in ``AGENT_PATHS`` list in ``langgraph_agent_toolkit/core/settings.py``
3. Optionally customize the Streamlit interface in ``streamlit_app.py``

Docker Setup
-----------

The ``docker-compose.yaml`` defines these services with enhanced security:

- ``backend-agent-service``: FastAPI service
- ``frontend-streamlit-app``: Streamlit chat interface
- ``postgres``: Database storage
- ``redis``: Cache and message broker
- ``minio``: Object storage
- ``clickhouse``: Analytics database
- ``langfuse-web`` & ``langfuse-worker``: Observability
- ``litellm``: LLM proxy server

Using docker compose watch enables live reloading:

1. Ensure Docker and Docker Compose (>=2.23.0) are installed

2. Launch services:

   .. code-block:: bash

      docker compose watch

3. Access endpoints:

   - Streamlit app: ``http://0.0.0.0:8501``
   - Agent API: ``http://0.0.0.0:8080``
   - API docs: ``http://0.0.0.0:8080/docs``
   - Langfuse dashboard: ``http://0.0.0.0:3000``
   - LiteLLM API: ``http://0.0.0.0:4000`` (accessible from any host)

4. Stop services:

   .. code-block:: bash

      docker compose down

.. note::
   If you modify ``pyproject.toml`` or ``uv.lock``, rebuild with
   ``docker compose up --build``

Using the AgentClient
--------------------

The toolkit includes ``AgentClient`` for interacting with the agent service:

.. code-block:: python

   from client import AgentClient
   client = AgentClient()

   response = client.invoke("Tell me a brief joke?")
   response.pretty_print()
   # ================================== Ai Message ==================================
   #
   # A man walked into a library and asked the librarian, "Do you have any books on Pavlov's dogs and Schrödinger's cat?"
   # The librarian replied, "It rings a bell, but I'm not sure if it's here or not."

See ``langgraph_agent_toolkit/run_client.py`` for more examples.

Development with LangGraph Studio
-------------------------------

The project works with LangGraph Studio:

1. Install LangGraph Studio
2. Add your ``.env`` file to the root directory
3. Launch LangGraph Studio pointing at the project root
4. Customize ``langgraph.json`` as needed

Local Development Without Docker
-------------------------------

1. Set up a Python environment:

   .. code-block:: bash

      pip install uv
      uv sync --frozen
      source .venv/bin/activate

2. Create and configure your ``.env`` file

3. Run the FastAPI server:

   .. code-block:: bash

      python langgraph_agent_toolkit/run_service.py

4. Run the Streamlit app in another terminal:

   .. code-block:: bash

      streamlit run langgraph_agent_toolkit/streamlit_app.py

5. Access the Streamlit interface (usually at ``http://localhost:8501``)

Key Features
----------

**LangGraph Integration**

- Latest LangGraph v0.3 features
- Human-in-the-loop with ``interrupt()``
- Flow control with ``Command`` and ``langgraph-supervisor``

**API Service**

- FastAPI with streaming and non-streaming endpoints
- Support for both token-based and message-based streaming
- Multiple agent support with URL path routing
- Available agents and models listed at ``/info`` endpoint
- Supports different runners: unicorn, gunicorn, mangum (AWS Lambda), azure functions

**Developer Experience**

- Asynchronous design with async/await
- Docker configuration with live reloading
- Comprehensive testing suite

**Enterprise Components**

- Configurable PostgreSQL/SQLite connection pools
- Observability via Langfuse and Langsmith
- User feedback system
- Prompt management system
- LiteLLM proxy integration
