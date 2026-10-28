Optional tool selection
========================

Tool selection limits the tool definitions that a model sees for one call.
Use it when an agent has a large catalog of tools. Keep it disabled for a small
catalog unless a representative test shows a benefit.

The toolkit reuses LangChain's ``LLMToolSelectorMiddleware`` and
``ProviderToolSearchMiddleware``. It does not implement the complete
`Less-is-More research method <https://arxiv.org/abs/2411.15399>`_. That method
also uses an ideal-tool recommendation call, embedding indexes, and offline
tool clustering.

Select a mode
--------------

.. list-table:: Tool selection modes
   :header-rows: 1
   :widths: 15 40 45

   * - Mode
     - Behavior
     - Use and limits
   * - ``off``
     - Keep the existing tool catalog. This is the default.
     - Use for small catalogs and as the benchmark baseline.
   * - ``llm``
     - Ask a model to select tools before the main model call.
     - Works with a selector model that supports LangChain structured output.
       Adds model requests when the candidate count exceeds ``max_tools``.
   * - ``provider``
     - Mark candidate tools for the provider's native tool search.
     - Requires a supported provider, model, and API route. It does not make a
       separate toolkit selector-model call.

Provider mode currently uses the upstream Anthropic and OpenAI integrations.
Provider detection alone does not prove that a deployed model or gateway
supports tool search. Test the actual model and API route before deployment.
An unsupported provider raises an error when searchable tools are present.
For OpenAI, configure the model instance with ``use_responses_api=True``.
The route must support the Responses API and tool search. A gateway that only
supports Chat Completions is not sufficient. Check the current `OpenAI guide
<https://developers.openai.com/api/docs/guides/tools-tool-search>`_ for model
support.
See the `provider search reference
<https://reference.langchain.com/python/langchain/agents/middleware/provider_tool_search/ProviderToolSearchMiddleware>`_.

Provider search controls model-context loading. The request can still contain
the deferred tool definitions. Do not assume that it reduces HTTP payload
bytes. MCP catalog discovery still occurs at worker startup.

Configure an API worker
-----------------------

``TOOL_SELECTION`` is one JSON setting. No change to a REST request body is
required. Set it before the service imports its agents:

.. code-block:: bash

   TOOL_SELECTION='{"mode":"llm","max_tools":8}'

The built-in ``create_agent``, ``create_agent_structured``, ``hitl_agent``,
and Deep Agents builders use this setting.
The custom ``create_react_agent`` creator does not
use this integration. Restart workers after a configuration change.

.. list-table:: ``ToolSelectionConfig`` fields
   :header-rows: 1
   :widths: 28 18 54

   * - Field
     - Default
     - Purpose
   * - ``mode``
     - ``off``
     - Choose ``off``, ``llm``, or ``provider``.
   * - ``max_tools``
     - ``8``
     - Positive limit on normally selected tools in ``llm`` mode. Required
       tools and provider-specific tool dictionaries do not use this limit.
   * - ``always_include``
     - ``[]``
     - Exact tool names to keep available. Names must be unique and present
       in the current model request.
   * - ``model_config_key``
     - Unset
     - Name of a selector model in ``MODEL_CONFIGS``. Applies to ``llm`` mode.
       An unset key uses the main model.
   * - ``max_retries``
     - ``1``
     - Additional selection attempts after a malformed structured response.
       Accepts values from ``0`` to ``3``. Applies to ``llm`` mode.
   * - ``on_parsing_failure``
     - ``error``
     - Choose ``error``, ``none``, or ``all`` after malformed-response retries.
       Applies to ``llm`` mode.

The bridge skips LLM selection when there are no tools or when the eligible
candidate count is at most ``max_tools``. Required tools remain available.
``max_tools`` is not a limit on total tool schemas, model calls, or tool calls.
It has no effect on provider mode.

Use the final registered names for MCP tools. A server prefix can change a
tool's name. Inspect the configured catalog before setting ``always_include``.
An unknown required name produces an error. See :doc:`mcp` for tool naming and
server allowlists.

Use a separate selector model
-----------------------------

The default LLM mode uses the main model for selection. A separate model can
reduce selection cost, but it must select the tools required by the task.
For example:

.. code-block:: bash

   MODEL_CONFIGS='{"selector":{"provider":"openai","name":"your-selector-model"}}'
   TOOL_SELECTION='{"mode":"llm","max_tools":8,"model_config_key":"selector"}'

Replace ``your-selector-model`` with a model available to the deployment.
Install its provider extra. Supply credentials through the process environment
or the secret manager. Do not put credentials in committed configuration.

The toolkit constructs this model through ``CompletionModelFactory``.
Managed HTTP connections, credentials, timeouts, and transport retries still
apply. A missing ``MODEL_CONFIGS`` key fails graph setup. Provider transport
retries are separate from malformed-response retries.

Use the Python builders
-----------------------

Pass a configuration to one graph to override the worker setting:

.. code-block:: python

   from langgraph_agent_toolkit.agents.blueprints.create_agent._shared import (
       build_tool_graph,
       create_model,
   )
   from langgraph_agent_toolkit.core.tool_selection import ToolSelectionConfig

   graph = build_tool_graph(
       create_model(),
       tool_selection=ToolSelectionConfig(mode="llm", max_tools=4),
   )

Pass ``selector_model=...`` to use an application-owned ``BaseChatModel``
instance. Pass either that instance or ``model_config_key``. Do not pass both.
The caller owns the lifecycle of an injected model and its connections.

The Deep Agents builder accepts the same selection arguments:

.. code-block:: python

   from langgraph_agent_toolkit.agents.blueprints.deep_agent.agent import build_graph
   from langgraph_agent_toolkit.core.tool_selection import ToolSelectionConfig

   graph = build_graph(
       tool_selection=ToolSelectionConfig(mode="llm", max_tools=4),
   )

Both builders accept ``extra_middleware=()`` for application middleware.
The toolkit applies this middleware before selection. An access filter can
remove tools before the selector sees them. In the native builder, the
direct-answer budget can remove all tools before selection runs.
Do not add another selector through ``extra_middleware`` when the configured
selector is active unless the application deliberately composes both.

``tool_selection=None`` uses the worker setting. To disable selection for one
graph even when that setting enables it, pass
``ToolSelectionConfig(mode="off")`` explicitly.

Keep Deep Agents control tools available
----------------------------------------

The integration keeps existing Deep Agents planning, delegation, file, and
asynchronous task tools available. It protects only tools present in that
request. These tools do not use ``max_tools`` and remain outside provider
search. The resulting catalog can therefore be larger than ``max_tools``.

Selection applies to the main graph built by the toolkit. Do not assume that
it also applies to a supplied subagent or to the upstream general-purpose
subagent. Configure those agents' own middleware when they need a selection
policy. Compiled subagents keep their existing graph behavior.

Selection does not change filesystem permissions, store namespaces, or the
checkpoint backend. It changes the model's view of tools for the current call.
See :doc:`deepagents` for backend and subagent access rules.

Handle failures and permissions
-------------------------------

The default parsing policy raises an error after the allowed attempts.
``none`` gives the main model no selected candidate tools. Required tools can
remain available. This can cause an incomplete answer.
``all`` restores the candidate catalog for that call. This fallback bypasses
``max_tools`` and can increase prompt size.

These policies handle returned values without the expected ``tools`` list.
Exceptions from structured-output parsing, provider errors, timeouts, invalid
tool names, and tool exceptions do not use this fallback policy.
The integration does not retry an executed tool when selection fails.
Keep approval and idempotency controls for external writes.
Human approval in ``hitl_agent`` still applies to a selected write tool.

Tool visibility is not authorization. Registered tools can still exist in the
graph's execution registry. Apply access checks in the tool or remote server.
Select only from the authorized catalog. A fallback must not restore tools
that an earlier access filter removed. ``always_include`` does not grant access.

The bridge creates native selector middleware for each model request.
Concurrent requests do not change a shared graph's tool list or checkpoint
history. MCP connections still use their normal worker lifecycle.

Evaluate multi-turn behavior and cost
-------------------------------------

The installed LangChain LLM selector bases its choice on the latest
``HumanMessage``. It does not use all earlier messages or intermediate tool
results to select the next tools. Test follow-ups such as ``do that`` and
workflows whose next tool depends on a prior result. Keep required workflow
tools available when that is appropriate. See the `LLM selector reference
<https://reference.langchain.com/python/langchain/agents/middleware/tool_selection/LLMToolSelectorMiddleware>`_.

A selected catalog can reduce the main prompt while increasing total model
requests. Selection can run again before a later main model call.
Changing tool subsets can also reduce provider prompt-cache reuse.
Measure selector calls, main calls, total tokens, cost, cache hits, task
completion, and p50/p95 latency together. Use the configured observability
backend to separate selector work from main-model work.

Run the offline example
-----------------------

The example uses fixed responses and real LangChain middleware. It does not
read ``.env``, contact a model provider, or enable remote tracing.
Run these commands from the repository root:

.. code-block:: bash

   uv run --no-sync python scripts/python/11-tool-selection.py --mode=off
   uv run --no-sync python scripts/python/11-tool-selection.py --mode=llm
   uv run --no-sync python scripts/python/11-tool-selection.py --builder=deep --mode=llm

Install the ``deepagents`` extra for the last command.
The output reports actual main-model tool names and selector calls.
The Deep Agents run also writes a virtual result file.
The example checks middleware integration. It does not measure real-model
tool-choice quality or provider latency.

With its fixed task and twelve unrelated tools, the native run changes each
main-model catalog from fifteen tools to one. It adds two selector calls.
The Deep Agents run changes its catalog from twenty-two tools to nine. It adds
three selector calls and keeps the control tools available.
These counts describe this example, not a recommended production limit.

Run the API benchmark
---------------------

The process tests compare ``off`` and ``llm`` with the same tasks and tool
catalogs. They run real API processes, HTTP requests, SQLite checkpoints, and
tools. Scripted models make the results repeatable.

.. code-block:: bash

   uv sync --frozen --no-install-project --extra all
   uv run --no-sync pytest tests/e2e/test_tool_selection_api.py \
     --run-e2e -q --basetemp=data/tool-selection-benchmark

Each builder writes a ``benchmark.json`` report under its test directory.
Set ``LAT_TEST_TOOL_SELECTION_REPORT_DIR`` to copy reports to one directory.
Use a separate empty directory for each comparison run.
The copied report names are ``native.json`` and ``deepagents.json``.

The report records serialized tool-schema UTF-8 bytes and tool counts.
These are not provider token counts or HTTP request bytes.
It also records selector and main-model calls, task success, side effects,
and local API p50/p95 latency. The suite checks streaming, saved history,
restart, concurrent requests, and bounded selection failures.

The comparison has 8, 32, and 128 unrelated tools.
Each variant uses one invoke request, one SSE request, and one JSONL
request. Its latency summary combines those three protocols. Read the raw
request samples before comparing protocols. Three samples do not establish
production tail latency or capacity.

.. list-table:: Deterministic comparison
   :header-rows: 1
   :widths: 22 25 23 30

   * - Builder
     - Main tools in ``off`` mode
     - Main tools in ``llm`` mode
     - Total model calls per task
   * - Native
     - 15 / 39 / 135
     - 3
     - 3 in ``off``; 6 in ``llm``
   * - Deep Agents
     - 22 / 46 / 142
     - 11
     - 9 in ``off``; 16 in ``llm``

The tasks complete in both modes. Deep Agents call counts include its
specialist. Main-model tool counts include tools kept available by policy.

Synthetic model timings measure local framework overhead. They do not predict
real-model speed, token cost, selection quality, or provider capacity.
Provider mode needs a separate check against the actual provider and gateway.
Before enabling selection in production, repeat the comparison with a
representative task set and the deployed models. Keep selection disabled for
agents where that comparison does not show a useful result.
