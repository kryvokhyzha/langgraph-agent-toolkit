Run your first agent
====================

This guide starts one local API with a deterministic fake model and a SQLite
checkpointer. It needs no model key or external service. It checks API behavior
and conversation persistence. Model quality requires separate evaluation.

Prepare a fresh checkout
------------------------

Install ``uv`` and use Python 3.11, 3.12, 3.13, or 3.14. Run these commands in a
shell without existing toolkit or tracing overrides:

.. code-block:: bash

   git clone https://github.com/kryvokhyzha/langgraph-agent-toolkit.git
   cd langgraph-agent-toolkit
   uv sync --frozen --no-install-project --no-dev --extra uvicorn-backend

Create ``.env`` in the fresh checkout with these values:

.. code-block:: ini

   USE_FAKE_MODEL=true
   AUTH_MODE=trusted
   AUTH_SECRET=local-demo-token
   AGENT_PATHS=["langgraph_agent_toolkit.agents.blueprints.chatbot.agent:chatbot_agent"]
   DEFAULT_AGENT=chatbot-agent
   MEMORY_BACKEND=sqlite
   SQLITE_DB_PATH=quickstart.sqlite
   OBSERVABILITY_BACKEND=empty
   MCP_SERVERS={}
   MODEL_CONFIGS={}
   LANGSMITH_TRACING=false
   LANGCHAIN_TRACING_V2=false

Do not replace an existing deployment's ``.env`` with this demo configuration.
The repository's ``.env.example`` is a configuration reference for external
services and models. Its placeholders need configuration before use.

``AUTH_MODE=trusted`` lets the trusted API caller supply ``user_id``. Use the
demo token only on localhost. For a deployed application, authenticate end users
in your backend and supply their stable IDs. See :doc:`migration` for the token
mode and ownership rules.

Start and call the API
----------------------

.. code-block:: bash

   uv run --no-sync python -m langgraph_agent_toolkit.run_api --host 127.0.0.1 --port 8080

In another terminal, export the demo token and check readiness. Retry after
startup if the first check cannot connect:

.. code-block:: bash

   export AUTH_SECRET=local-demo-token

   curl --fail http://127.0.0.1:8080/health/ready

Send the first turn:

.. code-block:: bash

   curl --fail-with-body http://127.0.0.1:8080/chatbot-agent/invoke \
     -H "Authorization: Bearer ${AUTH_SECRET}" \
     -H 'Content-Type: application/json' \
     -d '{"input":{"message":"Hello"},"user_id":"demo-user","thread_id":"demo-thread"}'

The JSON response contains ``type=ai``, ``thread_id=demo-thread``, a ``run_id``,
and the content ``This is a test response from the fake model.``
The service's interactive API reference is at ``http://127.0.0.1:8080/docs``.

Stream and read history
-----------------------

Send a second turn to the same conversation:

.. code-block:: bash

   curl --fail-with-body -N http://127.0.0.1:8080/chatbot-agent/stream \
     -H "Authorization: Bearer ${AUTH_SECRET}" \
     -H 'Content-Type: application/json' \
     -d '{"input":{"message":"Another turn"},"user_id":"demo-user","thread_id":"demo-thread","stream_tokens":true}'

SSE events use ``data:`` lines and end with ``data: [DONE]``. The response can
contain token, message, or error events. Check the events for errors even when
the HTTP status is 200. Use ``/chatbot-agent/stream/jsonl`` for JSON Lines.
That protocol has no ``[DONE]`` marker.

Read the saved conversation:

.. code-block:: bash

   curl --fail-with-body --get http://127.0.0.1:8080/chatbot-agent/history \
     -H "Authorization: Bearer ${AUTH_SECRET}" \
     --data-urlencode 'user_id=demo-user' \
     --data-urlencode 'thread_id=demo-thread'

After both turns, history contains two human messages and two AI replies.
Stop the API with Ctrl+C and restart it with the same command. Repeat the
history request. The file ``quickstart.sqlite`` retains the conversation.

Use the Python client
---------------------

Run this example from the checkout while the API is running:

.. code-block:: python

   from langgraph_agent_toolkit.client import AgentClient

   with AgentClient(
       base_url="http://127.0.0.1:8080",
       agent="chatbot-agent",
       auth_secret="local-demo-token",
   ) as client:
       reply = client.invoke(
           {"message": "Hello from Python"},
           user_id="demo-user",
           thread_id="python-demo",
       )
       print(reply.content)
       print(client.get_history("python-demo", user_id="demo-user"))

The context manager closes the client's owned connections. See :doc:`usage`
for async clients, streaming, history import, and feedback.

Add the optional UI
-------------------

Stop the API before changing its environment's dependencies. Install the UI
extra and restart the API with the earlier command:

.. code-block:: bash

   uv sync --frozen --no-install-project --no-dev --extra uvicorn-backend --extra ui

In another terminal, use the same checkout and start Streamlit:

.. code-block:: bash

   AGENT_URL=http://127.0.0.1:8080 AUTH_SECRET=local-demo-token \
     uv run --no-sync streamlit run langgraph_agent_toolkit/run_app.py

Open ``http://localhost:8501``. The UI calls the running API. A UI session can
use its own user and thread IDs. It does not automatically open the curl
example's conversation.

Select a real model or another agent
------------------------------------

Install the chosen provider extra, set its model and credentials, and set
``USE_FAKE_MODEL=false``. Keep real credentials in your deployment's environment
or secret manager. The quickstart does not select or call a real provider.
See :doc:`installation` and :doc:`environment_setup` for provider configuration.

Use :doc:`integrations` to choose a tool agent, fixed graph, or Deep Agent.
Register the selected agent through ``AGENT_PATHS`` and ``DEFAULT_AGENT``.
See :doc:`usage` for a custom agent and :doc:`mcp` for remote tools.

Before deployment, configure authentication, persistent storage, worker
supervision, and request limits through :doc:`deployment` and :doc:`reliability`.
The API Docker image selects aiohttp by default. Python installations select
HTTPX unless you configure another transport.
