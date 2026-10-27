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
serve a deployment for one client. With ``AUTH_MODE=trusted``, ``user_id`` is
optional. A trusted client can send it to separate end-user conversations.
Use the correct authenticated identity when you send it. See :doc:`migration`.

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

The toolkit selects Deep Agents 0.7.32. This version includes filesystem
permission fixes, media recovery, and async subagent updates.
See :doc:`dependency_releases_2026_10` for each intervening release.
Task planning remains optional.
The built-in blueprint explicitly enables ``TodoListMiddleware``
to provide ``write_todos``. It uses ``StateBackend`` for virtual files and
leaves the checkpointer for service startup to assign.

The default main agent and ``general-purpose`` subagent expose ``ls``,
``read_file``, ``write_file``, ``edit_file``, ``glob``, and ``grep`` for virtual
files. They also include ``add`` and ``multiply``. Their tool sets exclude
``execute`` and ``delete``. The default blueprint does not mount a host
directory or create a long-term memory store. Custom subagents follow the
configuration rules in the next section.

Read documents and media
------------------------

Use a model that supports each input format. Deep Agents checks the active
model before sending media. Its ``read_file`` tool can pass DOCX, PPTX, XLSX,
CSV, JSON, and Markdown files as native file inputs to OpenAI Responses.
Support depends on the provider, selected model, API, and MIME type.
See the `upstream multimodal guide
<https://docs.langchain.com/oss/python/deepagents/multimodal>`_.

The toolkit UI accepts these document extensions in addition to its existing
image, PDF, audio, and video extensions. It preserves the upload's filename
in the standard content block. The REST API uses the existing message format:

.. code-block:: json

   {
     "thread_id": "document-1",
     "input": {
       "message": [
         {"type": "text", "text": "Summarize this document."},
         {
           "type": "file",
           "base64": "<base64-encoded-document>",
           "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
           "extras": {"filename": "report.docx"}
         }
       ]
     }
   }

Replace the base64 placeholder with the document content. A filename does not
prove the content's format. Keep request-size limits and validate files at the
application boundary.

The blueprint passes a configured model instance to Deep Agents. Upstream's
OpenAI model-string defaults do not change that instance. Enable Responses
explicitly for a deployment that supports it:

.. code-block:: text

   MODEL_CONFIGS='{"deep_agent":{"provider":"openai","name":"gpt-4.1","use_responses_api":true}}'

Use the actual model or deployment name. Supply credentials through the
provider environment or secret mechanism. Restart the workers after changing
the configuration. The example does not change the toolkit's global defaults.
For a gateway that supports only Chat Completions, set
``use_responses_api=false`` and use formats that it accepts.
Test the actual gateway route before enabling native document inputs.
See `OpenAI model configuration
<https://docs.langchain.com/oss/python/deepagents/models#openai-and-the-responses-api>`_.

If a newly read file causes an invalid model request, upstream retries that
model call once with unsupported file content removed. This recovery does not
add retries for remote tool writes. Model-switching middleware also uses the
active model's content capabilities. Custom compiled subagents retain their
own middleware and model settings.

Offload binary content
----------------------

``build_graph`` accepts ``backend``, ``store``, and
``offload_binary_content``. The default is ``False``.
The default ``StateBackend`` stores files inside checkpoints.
It cannot reduce checkpoint size through binary offloading.
The helper rejects offloading when the blob route uses ``StateBackend``.

Use a backend that stores blobs outside graph state. One option is
``CompositeBackend`` with a ``/blobs/`` route to ``StoreBackend``.
For this route, also pass the store through ``build_graph(store=...)``.
This is required even if ``StoreBackend`` has an explicit store.
Use one durable store that all API replicas can access.
Keep its connections open for the full graph lifetime.
An in-memory store or a worker-local directory cannot provide restart recovery
across replicas.

The following embedded example uses the toolkit's PostgreSQL store and saver.
Configure the PostgreSQL settings first. Pass trusted identities and a model
that supports the input content:

.. code-block:: python

   from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
   from langchain_core.language_models import BaseChatModel
   from langgraph.runtime import Runtime

   from langgraph_agent_toolkit.agents.blueprints.deep_agent.agent import build_graph
   from langgraph_agent_toolkit.core.memory.factory import MemoryFactory
   from langgraph_agent_toolkit.core.memory.types import MemoryBackends
   from langgraph_agent_toolkit.service.auth import storage_thread_id


   def blob_namespace(runtime: Runtime) -> tuple[str, ...]:
       thread_id = runtime.execution_info.thread_id
       if not thread_id:
           raise ValueError("A scoped thread_id is required")
       return ("document-assistant", "blobs", thread_id)


   async def read_document(
       model: BaseChatModel,
       content: list[dict],
       user_id: str,
       thread_id: str,
   ):
       memory = MemoryFactory.create(MemoryBackends.POSTGRES)
       async with memory.get_memory_store() as store, memory.get_checkpoint_saver() as saver:
           await store.setup()
           await saver.setup()
           backend = CompositeBackend(
               default=StateBackend(),
               routes={"/blobs/": StoreBackend(namespace=blob_namespace)},
           )
           graph = build_graph(
               model=model,
               backend=backend,
               store=store,
               offload_binary_content=True,
           )
           graph.checkpointer = saver
           scoped_id = storage_thread_id(user_id, "document-assistant", thread_id)
           return await graph.ainvoke(
               {"messages": [{"role": "user", "content": content}]},
               config={"configurable": {"thread_id": scoped_id, "user_id": user_id}},
               durability="sync",
           )

The namespace uses the checkpoint key scoped by owner, agent, and public
thread ID. It keeps blobs in short-term conversation storage.
For HTTP runs, the service supplies that scoped key in
``runtime.execution_info.thread_id``. Do not use an unscoped public thread ID
as the only access boundary in a shared store.
Keep long-term files in a separate namespace based on trusted ``user_id``.
Use a separate store or schema for each client deployment.
If deployments share a store, add a trusted deployment ID to the namespace.

This example closes both database pools after the run. A service application
must keep the store open through its lifespan. The default HTTP service opens
its checkpointer only. ``MEMORY_BACKEND=postgres`` does not automatically
attach a blob store. A custom service must initialize the store and provide
it to ``build_graph(store=store, ...)`` through its graph factory.
The caller owns the store's connections. Include this pool in the total
database connection budget.

The middleware stores content under ``/blobs/<sha256>`` and keeps a small
reference in message history. It restores media before each model call.
Content hashes identify blobs. They do not enforce access control.
In this version, ``artifacts_root`` belongs to ``CompositeBackend``.
It is not a ``FilesystemMiddleware`` argument.
See `upstream binary offloading
<https://docs.langchain.com/oss/python/deepagents/backends#offload-binary-content>`_.

Keep these limits in the application design:

* Earlier input checkpoints can still contain the original inline upload.
  Offloading reduces later checkpoint payloads.
* A failed upload keeps media inline. A missing or invalid blob becomes a text
  notice, and the model call continues. Verify required media before treating
  the answer as complete.
* History returns saved references. It does not restore or download blobs.
* Clearing checkpoints does not delete external blobs. Keep blobs while any
  retained checkpoint can reference them. Add explicit retention and cleanup.
* Restored media still goes to the model. Request-size, provider, token, and
  traffic limits still apply.
* The default ``general-purpose`` subagent inherits the file middleware.
  A custom isolated subagent needs its own offloading configuration.
  A compiled subagent retains its own implementation.

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

   from deepagents import AsyncSubAgent, CompiledSubAgent, SubAgent
   from langchain_core.language_models import BaseChatModel
   from langchain_core.tools import BaseTool

   from langgraph_agent_toolkit.agents.agent import Agent
   from langgraph_agent_toolkit.agents.blueprints.deep_agent.agent import build_graph


   def make_assistant(
       model: BaseChatModel,
       tools: Sequence[BaseTool] = (),
       subagents: Sequence[SubAgent | CompiledSubAgent | AsyncSubAgent] = (),
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

``subagents`` accepts upstream ``SubAgent``, ``CompiledSubAgent``, and
``AsyncSubAgent`` definitions.
Give each specialist the tools it needs. The default ``general-purpose``
subagent inherits the blueprint's file-tool selection. A custom isolated
``SubAgent`` uses upstream filesystem defaults unless its own ``middleware``
overrides ``FilesystemMiddleware``. An empty ``tools`` list only removes
ordinary tools; middleware can still add file tools. Compiled subagents and
custom tools retain the access that application code gives them. The offline
example shows a reviewer subagent with a separate scripted model.

Configure skills, memory, and file permissions
----------------------------------------------

``build_graph`` accepts ``skills``, ``memory``, and ``permissions``.
All three default to ``None``. They add no sources or rules unless configured.
Paths refer to the selected backend. ``StateBackend`` reads files from graph
state. It does not read files from the host.

Use ``FilesystemBackend`` when an application needs a directory on disk.
Prepare these trusted files in that directory before building the graph:

* ``AGENTS.md`` contains persistent context, such as the report currency.
* ``skills/report-check/SKILL.md`` contains the report workflow.
  Its YAML frontmatter must include ``name`` and ``description``.

For example, use this content in ``skills/report-check/SKILL.md``:

.. code-block:: markdown

   ---
   name: report-check
   description: Check report totals before delivery.
   ---
   Read the report and check each total.

This factory loads the configured context. It also denies built-in file tools
access to ``/private/``:

.. code-block:: python

   from pathlib import Path

   from deepagents import FilesystemPermission
   from deepagents.backends import FilesystemBackend
   from langchain_core.language_models import BaseChatModel

   from langgraph_agent_toolkit.agents.blueprints.deep_agent.agent import build_graph


   def make_report_graph(model: BaseChatModel, workspace: Path):
       return build_graph(
           model=model,
           backend=FilesystemBackend(root_dir=workspace, virtual_mode=True),
           skills=["/skills/"],
           memory=["/AGENTS.md"],
           permissions=[
               FilesystemPermission(
                   operations=["read", "write"],
                   paths=["/private/**"],
                   mode="deny",
               )
           ],
       )


   async def prepare_report(model: BaseChatModel, workspace: Path):
       graph = make_report_graph(model, workspace)
       return await graph.ainvoke(
           {"messages": [{"role": "user", "content": "Prepare the report."}]}
       )

Supply a configured model and the prepared directory to ``prepare_report``.
Attach a durable checkpointer when the graph must retain conversation state.
The directory remains available to the process. ``virtual_mode=True`` gives
file paths a virtual root; it does not isolate the process or custom tools.

Skills and memory use direct backend reads. File-tool permissions do not
restrict those reads. Configure only trusted source paths. A ``memory`` path
loads instructions; it does not create a user-memory store. Use the namespace
rules in `Memory boundaries`_ for per-user data. Give each client deployment
its own directory or a backend that enforces its scope.

Permissions apply to the blueprint's built-in file tools. The default
``general-purpose`` subagent inherits them. A custom ``SubAgent`` inherits
them unless it supplies its own ``permissions``. That field replaces the
parent's rules. A ``CompiledSubAgent`` keeps its compiled behavior.
Custom tools, MCP tools, and direct backend calls keep their own access rules.

Run background subagents
------------------------

An ``AsyncSubAgent`` runs on a separate Agent Protocol server through the
LangGraph SDK. The toolkit HTTP API is not an Agent Protocol server.
Configure a compatible LangGraph deployment before using this option.
Supply its URL and authentication headers from application configuration.

.. code-block:: python

   from deepagents import AsyncSubAgent
   from langchain_core.language_models import BaseChatModel

   from langgraph_agent_toolkit.agents.blueprints.deep_agent.agent import build_graph


   def make_background_graph(
       model: BaseChatModel,
       agent_server_url: str,
       headers: dict[str, str],
   ):
       reviewer: AsyncSubAgent = {
           "name": "remote-review",
           "description": "Review a report in the background.",
           "graph_id": "report-review",
           "url": agent_server_url,
           "headers": headers,
           "mode": "isolated",
           "update_strategy": "enqueue",
           "notify_parent": False,
       }
       return build_graph(model=model, subagents=[reviewer])

Replace ``report-review`` with a graph or assistant ID on that server.
Building this graph does not start remote work. The model can use
``start_async_task``, ``check_async_task``, ``update_async_task``,
``cancel_async_task``, and ``list_async_tasks`` during execution.
The toolkit reserves these tool names.

``update_strategy="enqueue"`` queues a follow-up behind active remote work.
The upstream default is ``"interrupt"``, which replaces the active run.
A queued follow-up can start after the earlier run asks for human input.
In that case, the earlier request for input is dropped.
Do not enqueue a follow-up that must preserve that pending decision.

``notify_parent=False`` disables automatic parent wake-ups after completion.
The running parent can still check task status. The application must arrange
later checks or handle results itself. An optional ``webhook`` URL can receive
remote completion notifications. The application must provide that endpoint;
the toolkit does not install one.

Upstream marks queueing and parent notifications as beta.
A remote task with status ``waiting`` needs human input on its remote thread.
Resume that thread on the Agent Protocol server. A reply to the toolkit
conversation does not automatically resume it.

``mode="isolated"`` sends the delegated task without parent history.
The experimental ``mode="fork"`` also sends a parent-history snapshot.
It does not copy the remote graph's model, tools, or prompt configuration.
Matching those settings does not guarantee a provider prompt-cache hit.
The old ``"handoff"`` alias was removed. Use ``"isolated"`` or ``"fork"``.
See the `upstream async subagent guide
<https://docs.langchain.com/oss/python/deepagents/async-subagents>`_.

Offline tests verify graph registration without a remote connection.
Test the actual Agent Protocol deployment, authentication, remote approvals,
and result delivery before production use.

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

The toolkit's ``build_graph`` supports custom backends, skills, memory files,
and filesystem permissions. Use the upstream API when an application also
needs options such as ``context_schema`` or custom middleware.

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

Permission checks include the requested path and its symlink-resolved target.
This applies to deny rules and conditional approvals on ``FilesystemBackend``.
``permissions=[FilesystemPermission(..., mode="interrupt")]`` pauses matching
file operations for approval. Keep a durable checkpointer for these pauses.
Explicit ``interrupt_on`` entries take priority for the same tool.
Do not set an entry to ``False`` when its permission approval must remain active.

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
The selected version uses LangGraph's ``DeltaChannel`` message storage.
Its upstream checkpoint format is still marked beta. Keep database backups and verify old
checkpoints and pending approvals before upgrading either library.
See :doc:`testing`, :doc:`deployment`, and :doc:`langfuse_compatibility`.
