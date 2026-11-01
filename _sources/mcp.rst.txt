MCP tools
=========

Agents can call tools from Model Context Protocol (MCP) servers.
The toolkit uses ``langchain.mcp.MCPAdapter`` and FastMCP.
It discovers tools during service startup and adds them to selected agents.
The service API does not need a new endpoint.

Install
-------

Add the optional dependency to an application:

.. code-block:: bash

   uv add 'langgraph-agent-toolkit[mcp]'

For a source checkout, install the MCP extra with the required service extras:

.. code-block:: bash

   uv sync --frozen --no-install-project --extra mcp --extra uvicorn-backend --extra openai

The ``all`` extra and the backend Docker image include MCP support.
The base package does not install FastMCP.
No MCP connections start when the package or a blueprint is imported.

LangChain added its built-in MCP integration in version 1.4.0.
This namespace is in beta.
The toolkit limits LangChain to ``<1.5`` and FastMCP to ``<4.2``.
See the `LangChain announcement
<https://www.langchain.com/blog/mcp-in-langchain-stateless-protocol-elicitation-and-more>`_.

The older ``langchain-mcp-adapters`` package has a separate migration path.
Use the `official migration guide
<https://docs.langchain.com/oss/python/migrate/langchain-mcp-adapters>`_ for an
application that imports it.
The `adapter repository
<https://github.com/langchain-ai/langchain-mcp-adapters>`_ contains the old API.
The `langchain-mcp package on PyPI
<https://pypi.org/project/langchain-mcp/>`_ is a separate third-party package.
It is not required by this integration.

Configure an HTTP server
------------------------

Set these values in the service environment or its ``.env`` file:

.. code-block:: text

   MCP_SERVERS='{"documents":{"transport":"http","url":"https://documents.example.com/mcp","headers_env":{"Authorization":"DOCUMENTS_MCP_AUTH"},"tool_allowlist":["search"]}}'
   MCP_AGENT_SERVERS='{"create-agent":["documents"]}'
   MCP_DISCOVERY_TIMEOUT=30

Replace the example URL with your server URL.
Set ``DOCUMENTS_MCP_AUTH`` in the process environment through your secret manager.
Its value must contain the complete header value, such as ``Bearer <mcp-token>``.
``headers_env`` maps an HTTP header name to an environment variable name.
The toolkit does not add ``Bearer`` to the value.

``tool_allowlist`` contains the original tool names from the server.
This example exposes ``search`` to the agent as ``documents_search``.
Omit ``tool_allowlist`` to expose all tools from that server.
Use an empty list to expose no tools.
A missing tool name stops startup.

Each server has its own credentials.
The toolkit does not forward the API's ``AUTH_SECRET`` or ``user_id`` to MCP.
Conversation ownership and user memory scopes remain separate from MCP access.
Set the permissions for each MCP credential in the remote service.

Configure a local process
-------------------------

Use ``stdio`` for an MCP server that runs as a child process:

.. code-block:: text

   MCP_SERVERS='{"catalog":{"transport":"stdio","command":"python","args":["/app/mcp/catalog_server.py"],"env_env":{"CATALOG_API_KEY":"CATALOG_MCP_API_KEY"},"tool_allowlist":["lookup"]}}'
   MCP_AGENT_SERVERS='{"create-agent":["catalog"]}'

Install the command and script in the service environment or container.
The toolkit passes ``args`` as separate arguments.
It does not start a shell to interpret the command.
``env_env`` maps a child-process variable name to a service environment variable.
Direct values can use ``env``.

The transport uses ``keep_alive=False``.
It stops the child process after the last active connection context closes.
Concurrent calls can share a connection while their contexts overlap.
Use an HTTP server when repeated process startup is too expensive.
Store durable server state outside the child process.

Select agents and settings
--------------------------

``MCP_AGENT_SERVERS`` maps agent names to server aliases.
These built-in agents accept configured MCP tools:

* ``deep-agent`` (requires the ``deepagents`` extra; see :doc:`deepagents`)
* ``create-agent``
* ``create-agent-structured``
* ``hitl-agent``
* ``react-agent``

The selected agent must be loaded through ``AGENT_PATHS``.
When ``MCP_AGENT_SERVERS`` is empty, the resolved default agent receives all
configured servers.
An explicit mapping selects only the listed agents and servers.
An empty server list removes MCP tools from that selected agent.
Unknown agents, unknown servers, and duplicate assignments stop startup.

.. list-table:: Server configuration
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Behavior
   * - ``transport``
     - ``http`` or ``stdio``. The default is ``http``.
   * - ``url``
     - Required for HTTP. Use the Streamable HTTP MCP endpoint.
   * - ``command``, ``args``
     - The executable and argument list for stdio.
   * - ``headers``, ``headers_env``
     - Direct HTTP headers or references to environment variables.
   * - ``env``, ``env_env``
     - Direct stdio environment values or references to service variables.
   * - ``mode``
     - ``auto`` negotiates the protocol. ``legacy`` forces the older protocol.
   * - ``timeout``
     - Positive timeout in seconds for each client's requests and initialization.
       The default is 30.
   * - ``max_sse_event_size``
     - Positive integer limit in bytes for one incoming HTTP SSE event.
       Omit this field or use ``null`` to keep the SDK's 1 MiB default.
       This field does not apply to stdio or ordinary JSON responses.
   * - ``tool_allowlist``
     - Original tool names to expose. The default permits every discovered tool.

Server aliases must start with a letter.
They can contain letters, digits, underscores, and hyphens.
Their maximum length is 32 characters.
The final ``server_tool`` name must have at most 64 characters.
It can contain letters, digits, underscores, and hyphens.
Duplicate final tool names stop startup.

``MCP_DISCOVERY_TIMEOUT`` limits the complete discovery operation in seconds.
Its default is 30.
Workers discover only the servers assigned to agents.
Each worker builds its own graphs after discovery succeeds.
A connection error or timeout stops startup before readiness succeeds.

Tools retain their discovered schemas for the worker's lifetime.
Restart workers after a server changes its tool names or schemas.
Each new protocol connection loads the tool schemas before its first call.
This lets the MCP SDK send arguments marked with ``x-mcp-header`` as headers.
The FastMCP cache uses the server's cache policy and expiry time.
A server without a usable cache policy can still require catalog requests.
The toolkit does not force a cache expiry time on the server.

Each worker owns a separate HTTP connection pool for each configured server.
Tool calls close their protocol sessions after the last overlapping call exits.
The pool can reuse TCP connections between these sessions.
Worker shutdown and failed startup close the pools.
The pools use HTTPX2 defaults.
``LLM_HTTP_*`` settings apply to model-provider clients, not these MCP pools.

For HTTP tools with larger SSE responses, set ``max_sse_event_size`` on that
server. For example, ``"max_sse_event_size":2097152`` allows a 2 MiB event.
This limit includes the event's JSON data. A large text result can need more
space after JSON encoding. Keep the limit bounded to control per-call memory.
An oversized response fails the call. The toolkit does not repeat the operation.
The result can already be committed on the remote server.

The built-in retry middleware retries local tools only.
It does not retry a failed MCP write automatically.
A remote tool can finish a write before its connection fails.
Use server-side idempotency keys when a caller can retry that operation.

Resume a request for user input
-------------------------------

Modern MCP elicitation lets a tool request missing information.
The adapter converts this request to a LangGraph interrupt.
Keep ``CHECK_INTERRUPTS=true``.
Configure a checkpointer before using elicitation.
Use PostgreSQL or a persistent SQLite database when a run must survive a worker
restart.

The HTTP response keeps each interrupt in ``custom_data.interrupts``.
Each entry contains an ``id`` and its original ``value``.
For MCP elicitation, ``value.type`` is ``mcp_elicitation``.
``value.requests`` contains the question keys and form schemas or URLs.
The ``content`` field also contains text for display.
SSE and JSON Lines message events preserve the same ``custom_data`` object.

Recover questions after a lost response
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read ``/history`` or ``/{agent_id}/history`` with the paused conversation's
identity. The response includes ``pending_interrupts``:

.. code-block:: bash

   curl --get http://localhost:8080/create-agent/history \
     -H "Authorization: Bearer ${AUTH_SECRET}" \
     --data-urlencode 'thread_id=conversation-1' \
     --data-urlencode 'user_id=user-1'

.. code-block:: json

   {
     "messages": [],
     "next_offset": null,
     "total": 0,
     "pending_interrupts": [
       {
         "id": "actual-interrupt-id",
         "value": {
           "type": "mcp_elicitation",
           "tool_name": "save_report",
           "requests": [
             {
               "key": "destination",
               "mode": "form",
               "message": "Select a folder.",
               "requested_schema": {
                 "type": "object",
                 "properties": {"folder": {"type": "string"}},
                 "required": ["folder"]
               }
             }
           ]
         }
       }
     ]
   }

The example shows an empty message page with a pending question.
Every page includes all current pending interrupts.
Message pagination does not limit this field.
History reads use the saved state and do not call the remote tool.
Use the saved IDs to resume after a worker restart or a lost invoke response.
An empty list means that the saved conversation has no pending interrupts.
Recovery requires a saved checkpoint.
``NoOpSaver`` does not save pending questions.

Send structured answers
~~~~~~~~~~~~~~~~~~~~~~~

For one pending interrupt, put the answers in ``input.responses``:

.. code-block:: bash

   curl http://localhost:8080/create-agent/invoke \
     -H "Authorization: Bearer ${AUTH_SECRET}" \
     -H 'Content-Type: application/json' \
     -d '{
       "thread_id": "conversation-1",
       "user_id": "user-1",
       "input": {
         "responses": {
           "destination": {
             "action": "accept",
             "content": {"folder": "reports"}
           }
         }
       }
     }'

This example assumes ``AUTH_MODE=trusted``.
Set ``AUTH_SECRET`` in the caller's environment to the deployment's shared secret.
Use the same agent, ``thread_id``, ``user_id``, and authentication as the paused run.
Replace ``destination`` with the key from the actual question.
Match ``content`` to the requested form schema.
Answer every request key in that interrupt.

For multiple pending interrupts, put the answers in ``input.resume``.
Use every actual interrupt ID as a key:

.. code-block:: json

   {
     "thread_id": "conversation-1",
     "user_id": "user-1",
     "input": {
       "resume": {
         "interrupt-id-1": {
           "responses": {
             "destination": {
               "action": "accept",
               "content": {"folder": "reports"}
             }
           }
         },
         "interrupt-id-2": {
           "responses": {
             "confirmation": {"action": "decline"}
           }
         }
       }
     }
   }

Use either ``input.responses`` or ``input.resume`` in one request.
An answer's ``action`` can be ``accept``, ``decline``, or ``cancel``.
Send ``content`` only for an accepted form request.
For a URL request, complete the server's URL flow before sending ``accept``.
Do not send form content for a URL request.

Resuming elicitation repeats the remote tool call with the supplied answers.
The server must make any work before elicitation safe to repeat.
This is separate from retrying an ordinary transport failure.
See the elicitation example in the `LangChain announcement
<https://www.langchain.com/blog/mcp-in-langchain-stateless-protocol-elicitation-and-more>`_.

Legacy MCP tool calls are supported.
Legacy server-pushed elicitation is not supported by this integration.
Use the modern input-required protocol for interrupt and resume.

Use the Streamlit UI
~~~~~~~~~~~~~~~~~~~~

The included UI shows forms for MCP questions.
It reads pending questions from history when a conversation opens.
It also reads them from invoke, SSE, and JSON Lines responses.
Forms support basic strings, enums, booleans, numbers, and arrays of strings.
For advanced schemas, enter a JSON object and follow the displayed schema.
The API checks answer keys, actions, and MCP wire types before resume.
The MCP server must check the requested form schema and its constraints.

For a URL request, complete the linked operation before you select ``accept``.
Use ``decline`` or ``cancel`` when you do not want to supply the requested input.
The UI sends all answers with their request keys.
It uses actual interrupt IDs for parallel questions.
Ordinary chat input is unavailable while an MCP question needs an answer.
After a failed resume request, the UI reads history before it sends more input.

Read tool-result metadata
-------------------------

Tool messages include ``tool_status`` with ``success`` or ``error``.
MCP results with structured content also include ``tool_artifact``:

.. code-block:: json

   {
     "type": "tool",
     "content": "Report saved.",
     "tool_call_id": "call-1",
     "tool_status": "success",
     "tool_artifact": {
       "structured_content": {"report_id": "report-1"}
     }
   }

SSE, JSON Lines, and history preserve these fields.
The artifact contains only JSON-safe MCP ``structured_content`` objects.
Other Python artifacts remain outside the HTTP response.
Messages without a supported artifact use ``tool_artifact=null``.
An MCP tool error can use ``tool_status=error`` without failing the HTTP request.
Check this field when application code handles tool results directly.

Add MCP tools to a custom agent
-------------------------------

A custom ``Agent`` must provide ``graph_factory`` to use service configuration.
The factory receives the MCP tools assigned to that agent.
Combine them with local tools through ``merge_tools``.
It rejects duplicate names before the graph can replace a tool silently.

.. code-block:: python

   from collections.abc import Sequence

   from langchain.agents import create_agent
   from langchain_core.language_models import BaseChatModel
   from langchain_core.tools import BaseTool

   from langgraph_agent_toolkit.agents.agent import Agent
   from langgraph_agent_toolkit.agents.components.tools import add
   from langgraph_agent_toolkit.core.mcp import merge_tools


   def make_agent(model: BaseChatModel) -> Agent:
       def build_graph(extra_tools: Sequence[BaseTool] = ()):
           return create_agent(
               model=model,
               tools=merge_tools([add], extra_tools),
               checkpointer=None,
           )

       return Agent(
           name="custom-agent",
           description="An agent with local and MCP tools.",
           graph=build_graph(),
           graph_factory=build_graph,
       )

Call ``make_agent`` with your model.
Register the returned ``Agent`` object through ``AGENT_PATHS``.
The service attaches the configured checkpointer after it builds the graph.

For direct Python use, discover tools explicitly:

.. code-block:: python

   from langgraph_agent_toolkit.core.mcp import MCPServerConfig, load_mcp_tools


   async def get_document_tools():
       servers = {
           "documents": MCPServerConfig(
               url="https://documents.example.com/mcp",
               headers_env={"Authorization": "DOCUMENTS_MCP_AUTH"},
               tool_allowlist=["search"],
           )
       }
       tools_by_server = await load_mcp_tools(servers)
       return tools_by_server["documents"]

Call the resulting graph with ``ainvoke`` or ``astream``.
MCP tools use asynchronous operations.
Direct Python callers must supply a checkpointer for elicitation.

The example closes HTTP resources after each discovery or tool call.
Use ``MCPTransportManager`` to reuse HTTP connections across calls:

.. code-block:: python

   from langchain.agents import create_agent

   from langgraph_agent_toolkit.core.mcp import MCPTransportManager, load_mcp_tools


   async def run_agent(servers, model):
       async with MCPTransportManager() as manager:
           tools_by_server = await load_mcp_tools(servers, manager=manager)
           tools = [tool for group in tools_by_server.values() for tool in group]
           graph = create_agent(model=model, tools=tools)
           return await graph.ainvoke({"messages": [("user", "Find the report.")]})

Keep the manager open until all graph calls finish.
Use one manager in one event loop.
Do not call its tools after the manager closes.
The API service supplies this manager through its worker lifespan.

The service configuration does not start an interactive OAuth flow.
Use a custom ``fastmcp.Client`` and ``MCPAdapter`` in application code for advanced
authentication, custom transports, or client callbacks.
MCP resource and prompt APIs remain separate from tool discovery.
This integration does not add remote resources or prompts to an agent implicitly.

Verify a deployment
-------------------

Start the service with its MCP configuration.
Check ``/health/ready`` after startup.
Then send a request that needs an allowed MCP tool.
Check the remote service's result and the agent response.
Readiness confirms startup discovery; it does not poll remote MCP servers after
startup.

The MCP tests use local MCP servers and scripted models.
They do not call a production MCP server or a live model provider.
The Langfuse SDK matrix also installs the MCP extra.
Use a staging MCP server to verify deployment credentials and remote permissions.
