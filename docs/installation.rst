Installation Options
====================

Use Python 3.11, 3.12, 3.13, or 3.14. See :doc:`quickstart` for the first working
API example. It uses a fake model and needs no provider credentials.

Python 3.14 requires Pydantic 2.13.0 or later. This version also supports the
legacy models used by Langfuse SDK v2 and v3. The package metadata applies this
minimum only on Python 3.14. The ``studio`` extra also requires
``jsonschema-rs>=0.44.1`` on Python 3.14. Older locked versions cannot build on
this interpreter. Update the application lockfile when changing Python versions.
Docker images and the LangGraph deployment configuration still use Python 3.13.

For a published package installation, create an environment outside the source
checkout:

.. code-block:: bash

   uv venv --python 3.13
   uv pip install langgraph-agent-toolkit

The base installation contains the service, graph runtime, search tools, and
PostgreSQL and SQLite checkpoint adapters. These dependencies remain in the
base package to preserve existing service installations. It does not install a
model provider, Streamlit, or the LangGraph development server.

Provider and Backend Extras
---------------------------

Install the extras for the provider and deployment backend that you use.

.. code-block:: bash

   # OpenAI, Uvicorn, and Langfuse SDK v4
   uv pip install "langgraph-agent-toolkit[openai,uvicorn-backend,langfuse-v4]"

   # Anthropic, AWS Lambda, and LangSmith
   uv pip install "langgraph-agent-toolkit[anthropic,aws-backend,langsmith]"

The provider extras are ``openai``, ``anthropic``, ``aws``,
``google-vertexai``, and ``google-genai``. ``all-llms`` installs all of them.
The fake model works without a provider extra or an API key.

``openai-aiohttp`` adds the OpenAI SDK's optional asynchronous transport.
The API Docker image includes this extra and selects aiohttp by default.
``all-llms`` and ``all`` also include it. Python installations keep HTTPX as
their default so the base ``openai`` extra does not require aiohttp.
See :doc:`reliability` to select a transport and configure connection limits.

The backend extras are ``uvicorn-backend``, ``gunicorn-backend``,
``aws-backend``, and ``azure-backend``. ``all-backends`` installs all of them.
Uvicorn is already a base dependency. Its named extra remains available for
existing installation commands.
Use ``langfuse-v2``, ``langfuse-v3``, or ``langfuse-v4`` to select a Langfuse
Python SDK generation. Choose only one selector. These extras do not select or
install a Langfuse server. See :doc:`langfuse_compatibility` for the supported
SDK and server combinations.

The generic ``langfuse`` extra permits SDK versions from 2.60.10 to below 5.
It can combine with any one version selector. ``langsmith`` selects LangSmith,
and ``all-observability`` includes both generic ``langfuse`` and ``langsmith``.
Extra names use hyphens; underscore spellings such as ``langfuse_v2`` normalize
to the same name.

Agent Integrations
------------------

Install only the integrations that your agents use:

.. code-block:: bash

   uv pip install "langgraph-agent-toolkit[mcp]"
   uv pip install "langgraph-agent-toolkit[deepagents,openai]"

``mcp`` adds remote tool discovery and calls. ``deepagents`` adds the optional
Deep Agent blueprint. A real model still needs its provider extra and
configuration. See :doc:`mcp`, :doc:`deepagents`, and :doc:`integrations` for
their use cases and limits.

UI and Development Server
-------------------------

.. code-block:: bash

   # Install the Streamlit frontend.
   uv pip install "langgraph-agent-toolkit[ui]"

   # Install the LangGraph development server and CLI.
   uv pip install "langgraph-agent-toolkit[studio]"

   # Install all optional features with the default SDK selection.
   uv pip install "langgraph-agent-toolkit[all]"

   # Install the same features with Langfuse SDK v2.
   uv pip install "langgraph-agent-toolkit[all,langfuse-v2]"

Extras add dependencies. They cannot remove the graph and service dependencies
from the base wheel. There is no ``client`` package extra that produces a
smaller wheel installation. Langfuse SDK selectors are published extras in the
wheel. Local dependency groups are available only from a source checkout.

Minimal Frontend from Source
----------------------------

A source checkout has a separate dependency group for the SDK and Streamlit
frontend. This group does not install the package or its service dependencies.
Run these commands from the repository root:

.. code-block:: bash

   uv sync --frozen --only-group client
   uv run --no-sync streamlit run langgraph_agent_toolkit/run_app.py

Keep ``--no-sync`` on the run command. A normal project sync also installs the
base service dependencies.

Development and Deployment
--------------------------

Install the locked development environment with all optional features:

.. code-block:: bash

   uv sync --frozen --no-install-project --extra all
   uv run --no-sync pytest

Replace earlier ``--all-extras`` commands with ``--extra all``. Selecting every
extra now includes three incompatible Langfuse SDK selectors. The ``all`` extra
keeps the complete feature set and uses SDK v4 in the default lockfile selection.
It does not force the ``langfuse-v4`` selector.

Add one selector to use another SDK generation, when all dependencies permit it:

.. code-block:: bash

   uv sync --frozen --no-install-project --extra all --extra langfuse-v2

Use ``langfuse-v3`` or ``langfuse-v4`` in the same position. Keep the selected
SDK generation in the deployment's install command and lockfile.

The lockfile fixes the complete source environment. The wheel metadata limits
the graph and model framework APIs to the versions supported by this release.
Python 3.13 is required for the configured pre-commit hook environments.
See :doc:`contributing` for source changes and :doc:`testing` for opt-in process,
container, database, and live-provider tests.

The API Docker image includes OpenAI, both HTTP transports, MCP, Gunicorn, and
observability dependencies. Deep Agents requires the ``INSTALL_DEEPAGENTS=true``
build argument. Other model providers require a custom image with their extras.

See :doc:`deployment` for worker recovery, database sizing, and concurrent
conversation limits.
