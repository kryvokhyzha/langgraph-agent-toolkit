Deep Agents
===========

`Deep Agents <https://github.com/langchain-ai/deepagents>`_ adds file tools,
delegation, and context management to LangChain's ``create_agent``. It uses
LangGraph for execution and checkpoints. The toolkit integration keeps the
existing HTTP API, authentication, and database lifecycle.

Use it for work that needs intermediate documents or separate research tasks.
A short support exchange usually needs fewer components. See
:doc:`integrations` to compare the available agent patterns.

Install
-------

Add the optional dependency to an application:

.. code-block:: bash

   uv add 'langgraph-agent-toolkit[deepagents]'

For a source checkout, install the example and service dependencies:

.. code-block:: bash

   uv sync --frozen --no-install-project --extra deepagents --extra uvicorn-backend --extra openai

The ``all`` extra includes Deep Agents. The default backend Docker image does
not include it. Build an image with this option:

.. code-block:: bash

   docker build -f docker/api/Dockerfile --build-arg INSTALL_DEEPAGENTS=true -t toolkit-deep-agent .

Run the example
---------------

Run this command from the repository root:

.. code-block:: bash

   uv run --no-sync python scripts/python/10-deep-agent.py

The default example needs no API key. A scripted model calls the real planning,
arithmetic, file, and delegation tools. It saves a delivery report in virtual
files and checks that the report reached the SQLite checkpoint. External
tracing is disabled for this offline run.

The default database is ``data/deep-agent-demo.sqlite``. The default
``thread_id`` is ``deep-agent-demo``, and the default ``user_id`` is
``demo-user``. Run it again with the same identifiers to retain earlier files.
Each run writes a report to a new virtual path.

Use a different database or conversation with these options:

.. code-block:: bash

   uv run --no-sync python scripts/python/10-deep-agent.py \
     --database=data/research.sqlite --thread_id=research-1 --user_id=user-1

For a live model, configure ``.local.env`` or the process environment first.
Set the selected provider's credentials and model name. Then run:

.. code-block:: bash

   uv run --no-sync python scripts/python/10-deep-agent.py --live=True \
     --thread_id=live-research-1 \
     --message='Compare two delivery plans and save your calculations in a report.'

The live run uses the blueprint's startup model selection described below.
It can call a model provider and incur charges. Its tool order and answer are
model decisions. The script has a 120-second execution limit.

Use the HTTP service
--------------------

Set these values in the service environment or its ``.env`` file:

.. code-block:: text

   AGENT_PATHS='["langgraph_agent_toolkit.agents.blueprints.deep_agent.agent:deep_agent"]'
   DEFAULT_AGENT=deep-agent
   MEMORY_BACKEND=sqlite
   SQLITE_DB_PATH=data/deep-agent-api.sqlite
   AUTH_MODE=trusted
   USE_FAKE_MODEL=true

Set ``AUTH_SECRET`` through your secret manager. The same shared secret can
serve a deployment for one client. With ``AUTH_MODE=trusted``, that client must
send the correct end-user identity in ``user_id``. See :doc:`migration`.

Create the database directory, then start the service:

.. code-block:: bash

   mkdir -p data
   uv run --no-sync python -m langgraph_agent_toolkit.run_api \
     --host=127.0.0.1 --port=8080 --reload=False

Use another terminal with the same ``AUTH_SECRET`` in its environment:

.. code-block:: bash

   curl --fail http://127.0.0.1:8080/health/ready
   curl --fail http://127.0.0.1:8080/deep-agent/invoke \
     -H "Authorization: Bearer ${AUTH_SECRET}" \
     -H 'Content-Type: application/json' \
     -d '{
       "thread_id": "report-1",
       "user_id": "user-1",
       "input": {"message": "Calculate six batches of seven items and save a report."}
     }'

``USE_FAKE_MODEL=true`` checks the HTTP path with a fixed response. Use the
offline script to exercise a complete planned tool sequence. For real tasks,
set ``USE_FAKE_MODEL=false`` and configure the model before restarting workers.

The model is selected once when each worker loads the blueprint:

1. ``USE_FAKE_MODEL=true`` selects the fake model.
2. Otherwise, ``MODEL_CONFIGS["deep_agent"]`` selects a full provider and model
   configuration, when present. Supply its provider credentials through that
   configuration or the provider's supported environment variables.
3. Otherwise, the blueprint uses ``OPENAI_MODEL_NAME``, ``OPENAI_API_KEY``, and
   ``OPENAI_API_BASE_URL``.

Install the extra for the selected model provider. Request fields such as
``model_name``, ``model_provider``, and ``model_config_key`` do not switch this
blueprint's model. Use a custom ``build_graph(model=...)`` for application-owned
model instances. See :doc:`environment_setup` for model configuration.

The existing invoke, streaming, history, clear, and interrupt-resume endpoints
also apply to this agent. History contains messages. It is not a file-download
API. The Python example uses ``graph.aget_state`` to inspect virtual files.

What it adds
------------

The upstream library provides virtual files, subagents with separate message
contexts, and summaries for long conversations. It can move large tool
results into files for later access. Skills and memory files are optional
sources of instructions. See the `upstream overview
<https://docs.langchain.com/oss/python/deepagents/overview>`_.

The toolkit selects Deep Agents 0.7.13. This version makes task planning
optional. The built-in blueprint explicitly enables ``TodoListMiddleware``
to provide ``write_todos``. It uses ``StateBackend`` for virtual files and
leaves the checkpointer for service startup to assign.

The default main agent and ``general-purpose`` subagent expose ``ls``,
``read_file``, ``write_file``, ``edit_file``, ``glob``, and ``grep`` for virtual
files. They also include ``add`` and ``multiply``. Their tool sets exclude
``execute`` and ``delete_file``. The default blueprint does not mount a host
directory or create a long-term memory store. Custom subagents follow the
configuration rules in the next section.

Add tools and subagents
-----------------------

Install the ``mcp`` extra to add tools from an MCP server. Use the configuration
in :doc:`mcp` and assign the server to this agent:

.. code-block:: text

   MCP_AGENT_SERVERS='{"deep-agent":["documents"]}'

``documents`` must name a configured ``MCP_SERVERS`` entry. Discovery runs at
service startup. The graph factory combines these tools with the blueprint's
local tools. Duplicate names and names reserved for Deep Agents tools stop
startup. Remote tool failures do not trigger automatic toolkit tool retries.

For Python customization, use ``build_graph``. This helper keeps local tools
and MCP injection in one service registration:

.. code-block:: python

   from collections.abc import Sequence

   from deepagents import CompiledSubAgent, SubAgent
   from langchain_core.language_models import BaseChatModel
   from langchain_core.tools import BaseTool

   from langgraph_agent_toolkit.agents.agent import Agent
   from langgraph_agent_toolkit.agents.blueprints.deep_agent.agent import build_graph


   def make_assistant(
       model: BaseChatModel,
       tools: Sequence[BaseTool] = (),
       subagents: Sequence[SubAgent | CompiledSubAgent] = (),
   ) -> Agent:
       def create_graph(extra_tools: Sequence[BaseTool] = ()):
           return build_graph(
               [*tools, *extra_tools], model=model, subagents=subagents
           )

       return Agent(
           name="report-assistant",
           description="Prepare reports with application tools and specialists.",
           graph=create_graph(),
           graph_factory=create_graph,
       )

Call ``make_assistant`` with your model and tools. Export the returned object
from an application module. Register that object through ``AGENT_PATHS``.
Assign MCP servers to ``report-assistant`` when required. The service attaches
its configured checkpointer after building the graph.

``subagents`` accepts upstream ``SubAgent`` definitions or compiled graphs.
Give each specialist the tools it needs. The default ``general-purpose``
subagent inherits the blueprint's file-tool selection. A custom isolated
``SubAgent`` uses upstream filesystem defaults unless its own ``middleware``
overrides ``FilesystemMiddleware``. An empty ``tools`` list only removes
ordinary tools; middleware can still add file tools. Compiled subagents and
custom tools retain the access that application code gives them. The offline
example shows a reviewer subagent with a separate scripted model.

Advantages and tradeoffs
------------------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Advantage
     - Cost or limit
   * - Files hold working notes and large results.
     - State files increase checkpoint size. Set retention rules and measure
       database growth on long tasks.
   * - Subagents keep detailed work out of the main message context.
     - They add model calls. Parallel execution can reduce elapsed time but
       increase simultaneous provider requests.
   * - Context summaries let work continue as history grows.
     - Summaries can omit details. Keep authoritative facts in source documents
       and verify the final result against them.
   * - Built-in tools and prompts reduce custom agent code.
     - Their behavior is part of the dependency contract. Test upgrades with
       the actual model and tools used by the application.

These are design tradeoffs, not measured speed or quality claims. The
`subagent guide <https://docs.langchain.com/oss/python/deepagents/subagents>`_
describes context isolation and focused tool sets. The
`context guide <https://docs.langchain.com/oss/python/deepagents/context-engineering>`_
describes summaries and tool-result storage.

Memory boundaries
-----------------

``thread_id`` identifies short-term state. The service scopes checkpoints to
the conversation owner and agent. Deep Agents files in ``StateBackend`` use
that same state. They persist across turns when a checkpointer is configured.
SQLite or PostgreSQL can preserve them across worker replacement.

Parent and subagent message contexts are separate. Their state-backed virtual
files can be shared within the run. Give concurrent subtasks different output
paths when they must not overwrite each other's results.

``user_id`` identifies long-term user memory. It remains independent of
``thread_id``. Passing it to the service does not create a ``StoreBackend`` or
a cross-thread filesystem. A custom application must provide a store and a
namespace based on the trusted user identity.

Clearing a conversation removes its checkpoints and state-backed files.
It does not delete files in an external backend or long-term user memory.

Add long-term files in an embedded application
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the upstream API when an application needs a custom backend, skills,
memory files, or filesystem permission rules. The toolkit's ``build_graph``
exposes a smaller set of options.

This separate Python factory routes ``/memories/`` to a store. Other files
remain in thread state. The store namespace comes from trusted runtime context:

.. code-block:: python

   from dataclasses import dataclass
   from hashlib import sha256

   from deepagents import create_deep_agent
   from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
   from langchain_core.language_models import BaseChatModel
   from langgraph.checkpoint.memory import InMemorySaver
   from langgraph.runtime import Runtime
   from langgraph.store.memory import InMemoryStore


   @dataclass(frozen=True)
   class MemoryContext:
       user_id: str


   def memory_namespace(runtime: Runtime[MemoryContext]) -> tuple[str, ...]:
       user_id = runtime.context.user_id
       if not isinstance(user_id, str) or not user_id:
           raise ValueError("user_id must be a nonempty string")
       digest = sha256(user_id.encode("utf-8")).hexdigest()
       return ("deep-agent", "users", digest)


   def make_memory_graph(model: BaseChatModel):
       return create_deep_agent(
           model=model,
           backend=CompositeBackend(
               default=StateBackend(),
               routes={
                   "/memories/": StoreBackend(namespace=memory_namespace)
               },
           ),
           context_schema=MemoryContext,
           store=InMemoryStore(),
           checkpointer=InMemorySaver(),
       )


   async def remember_preference(model: BaseChatModel):
       graph = make_memory_graph(model)
       context = MemoryContext(user_id="user-1")
       await graph.ainvoke(
           {"messages": [{"role": "user", "content":
               "Write /memories/preferences.md with this preference: concise reports."}]},
           config={"configurable": {"thread_id": "conversation-1"}},
           context=context,
       )
       return await graph.ainvoke(
           {"messages": [{"role": "user", "content":
               "Read /memories/preferences.md and state my report preference."}]},
           config={"configurable": {"thread_id": "conversation-2"}},
           context=context,
       )

Call ``remember_preference`` with a model that supports tool calls. Both turns
use one graph and one user namespace, but different threads. This example uses
in-memory storage. Its data is lost when the process exits. For production,
supply a durable store and saver. Open and close their connections in the
application lifecycle.

The namespace hashes the complete user ID without changing that identity.
This permits IDs with Unicode, spaces, or characters that ``StoreBackend``
rejects in a namespace component. The ``deep-agent`` prefix scopes this example
to one agent. Use a different prefix for another agent that shares the store.

Concurrent threads for one user can access the same long-term file. The
toolkit's conversation lock does not serialize those cross-thread edits.
Use unique artifact paths or application coordination for each user and file.
If the store supports compare-and-swap, use it to reject conflicting updates.
``StoreBackend`` does not guarantee atomic file edits.

This is an embedded application example. The toolkit HTTP service does not
initialize this store or pass ``MemoryContext`` automatically. A custom
application must derive that context from authenticated identity. It must also
scope its checkpoint keys. Do not derive either scope from model output.
See the `upstream backend guide
<https://docs.langchain.com/oss/python/deepagents/backends>`_.

Tools and permissions
---------------------

Add business tools through ``build_graph(extra_tools=...)`` or configure MCP
servers through the toolkit. Remote credentials must restrict access at the
target service. Deep Agents filesystem permissions apply to its built-in file
tools. They do not restrict custom tools, MCP tools, or shell execution.
Permission rules use the first match and allow unmatched paths. End a path
allowlist with a deny rule for the remaining paths. Keep tool permissions in
the tool service as well as in the agent configuration.
See `upstream permissions
<https://docs.langchain.com/oss/python/deepagents/permissions>`_.

Use ``build_graph(interrupt_on={"tool_name": True})`` to pause before a tool
that requires human approval. Replace ``tool_name`` with the actual tool name.
Keep ``CHECK_INTERRUPTS=True`` and configure a durable checkpointer. Place the
external write after approval. Repeating a failed request can repeat an
external side effect, so use idempotency at the target API.

For code execution, choose a sandbox with explicit resource and network
limits. ``LocalShellBackend`` executes on the host. A working directory is
not a security boundary. The default toolkit blueprint does not select it.
See `upstream backend guidance
<https://docs.langchain.com/oss/python/deepagents/backends>`_.

Operate long tasks
------------------

Set ``REQUEST_TIMEOUT`` for the longest supported request. Set model and tool
timeouts below that limit. Use ``DEFAULT_RECURSION_LIMIT`` or the request's
``recursion_limit`` to bound graph steps. A recursion limit is not a token or
cost budget; several model calls can occur inside delegated work.

The service coordinates requests for one conversation. Deep Agents does not
add a durable job queue or replay an HTTP request after a worker fails.
Use a separate job system when work must continue independently of a connected
HTTP client.

Record task completion, latency, model usage, tool failures, and checkpoint
size. Validate the final output on representative tasks. Offline examples
verify execution contracts; they do not estimate real-model quality or cost.
Version 0.7.13 uses LangGraph's ``DeltaChannel`` message storage. Its upstream
checkpoint format is still marked beta. Keep database backups and verify old
checkpoints and pending approvals before upgrading either library.
See :doc:`testing`, :doc:`deployment`, and :doc:`langfuse_compatibility`.
