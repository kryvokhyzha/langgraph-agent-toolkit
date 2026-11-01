October 2026 dependency releases
================================

Review date: 2026-10-27.

This review starts from the previous 0.10.2 candidate lockfile.
It lists stable releases up to the current baseline.
It omits development-only changes from the summaries.
Each version links to the full upstream release notes.
See :doc:`dependency_updates` for the selected versions and deployment steps.

Deep Agents
-----------

* `0.7.24 <https://github.com/langchain-ai/deepagents/releases/tag/deepagents%3D%3D0.7.24>`_: Add fork mode for asynchronous subagents. Recover when summarization exceeds the context limit. Preserve the parent rubric during parallel calls.
* `0.7.25 <https://github.com/langchain-ai/deepagents/releases/tag/deepagents%3D%3D0.7.25>`_: Check requested and resolved paths against filesystem permissions and approval rules. Read skill-tool support from model profiles. Prefer ripgrep for sandbox searches and skip ``.git``.
* `0.7.26 <https://github.com/langchain-ai/deepagents/releases/tag/deepagents%3D%3D0.7.26>`_: Report subagents that wait for human input. Connect subagent traces to their parent. Expose the summarizer and preserve resolved tool-result paths.
* `0.7.27 <https://github.com/langchain-ai/deepagents/releases/tag/deepagents%3D%3D0.7.27>`_: Send asynchronous subagent status updates to the parent. Pass subagent names to their runs.
* `0.7.28 <https://github.com/langchain-ai/deepagents/releases/tag/deepagents%3D%3D0.7.28>`_: Apply the ``path`` argument in ``ContextHubBackend.glob``.
* `0.7.29 <https://github.com/langchain-ai/deepagents/releases/tag/deepagents%3D%3D0.7.29>`_: Accept composite routes without a final slash. Reject text edits to binary files. Skip binary files in in-memory ``grep``.
* `0.7.30 <https://github.com/langchain-ai/deepagents/releases/tag/deepagents%3D%3D0.7.30>`_: Queue subagent follow-up messages. Add ``notify_parent`` and ``webhook``. Report remote cancellation and retry failed thread reads. Remove replaced reasoning blocks during summarization. Measure shell output limits in UTF-8 bytes.
* `0.7.31 <https://github.com/langchain-ai/deepagents/releases/tag/deepagents%3D%3D0.7.31>`_: Trace execution output retention. Omit asynchronous task IDs from replies by default. Preserve permissions in middleware overrides. Remove the ``handoff`` mode alias; use ``isolated`` or ``fork``.
* `0.7.32 <https://github.com/langchain-ai/deepagents/releases/tag/deepagents%3D%3D0.7.32>`_: Defer Anthropic and OpenAI imports until the SDK needs them. This reduces import work.

LangChain
---------

* `1.4.5 <https://github.com/langchain-ai/langchain/releases/tag/langchain%3D%3D1.4.5>`_: Accept a custom ``summarizer`` in ``SummarizationMiddleware``.
* `1.4.6 <https://github.com/langchain-ai/langchain/releases/tag/langchain%3D%3D1.4.6>`_: Add optional approval interrupts for each tool call in ``HumanInTheLoopMiddleware``. The default interrupt format stays unchanged.
* `1.4.7 <https://github.com/langchain-ai/langchain/releases/tag/langchain%3D%3D1.4.7>`_: Read ``tool-call-delta`` events in stream consumers. Accept an agent address in ``LangChainTracer``. Raise the Core and LangGraph minimum versions.

LangChain Core
--------------

* `1.6.10 <https://github.com/langchain-ai/langchain/releases/tag/langchain-core%3D%3D1.6.10>`_: Declare support for mid-conversation changes in model profiles.
* `1.6.11 <https://github.com/langchain-ai/langchain/releases/tag/langchain-core%3D%3D1.6.11>`_: Accept an agent address in ``LangChainTracer``.
* `1.6.12 <https://github.com/langchain-ai/langchain/releases/tag/langchain-core%3D%3D1.6.12>`_: Preserve ``extras`` in content-block start events.
* `1.6.13 <https://github.com/langchain-ai/langchain/releases/tag/langchain-core%3D%3D1.6.13>`_: Read ``tool-call-delta`` events. Iteration over ``ChatModelStream.tool_calls`` now returns new argument slices.
* `1.7.0 <https://github.com/langchain-ai/langchain/releases/tag/langchain-core%3D%3D1.7.0>`_: Emit ``tool-call-delta`` argument slices instead of accumulated ``block-delta`` arguments. The accumulated ``.tool_calls.get()`` and ``.output`` contracts stay unchanged.

OpenAI integration
------------------

* `1.6.3 <https://github.com/langchain-ai/langchain/releases/tag/langchain-openai%3D%3D1.6.3>`_: Expose inferred Responses API selection at model creation. Apply GPT-6 request constraints.
* `1.6.4 <https://github.com/langchain-ai/langchain/releases/tag/langchain-openai%3D%3D1.6.4>`_: Refresh OpenAI model profiles.
* `1.6.5 <https://github.com/langchain-ai/langchain/releases/tag/langchain-openai%3D%3D1.6.5>`_: Support mid-conversation tool changes on ``SystemMessage``. Update model profiles.
* `1.6.6 <https://github.com/langchain-ai/langchain/releases/tag/langchain-openai%3D%3D1.6.6>`_: Raise an error when a provider sends an error event in a stream.
* `1.6.7 <https://github.com/langchain-ai/langchain/releases/tag/langchain-openai%3D%3D1.6.7>`_: Discover Azure workload identity. Refresh model profiles.
* `1.7.0 <https://github.com/langchain-ai/langchain/releases/tag/langchain-openai%3D%3D1.7.0>`_: Add Decisions API support.
* `1.7.1 <https://github.com/langchain-ai/langchain/releases/tag/langchain-openai%3D%3D1.7.1>`_: Declare mid-conversation support in model profiles.
* `1.7.2 <https://github.com/langchain-ai/langchain/releases/tag/langchain-openai%3D%3D1.7.2>`_: Keep one content block when Responses API events change an item's ``output_index``.
* `1.7.3 <https://github.com/langchain-ai/langchain/releases/tag/langchain-openai%3D%3D1.7.3>`_: Refresh model profiles and dependency requirements. Require Core 1.7.0 or later.

Anthropic integration
---------------------

* `1.7.6 <https://github.com/langchain-ai/langchain/releases/tag/langchain-anthropic%3D%3D1.7.6>`_: Refresh model profiles and declare mid-conversation support. Require Core 1.6.10 or later.

AWS integration
---------------

* `1.6.0 <https://github.com/langchain-ai/langchain-aws/releases/tag/langchain-aws%3D%3D1.6.0>`_: Report cache-write usage by cache TTL. Preserve non-ASCII JSON and omit unsupported sampling parameters. When Bedrock returns ``cacheDetails``, cache-write usage moves from ``cache_creation`` to TTL-specific keys.
* `1.6.1 <https://github.com/langchain-ai/langchain-aws/releases/tag/langchain-aws%3D%3D1.6.1>`_: Limit JSON Schema union variants. Preserve checkpoint namespaces for AgentCore code tools. Separate Nova reasoning from user text. Accept the ``streaming`` initialization argument.
* `1.6.2 <https://github.com/langchain-ai/langchain-aws/releases/tag/langchain-aws%3D%3D1.6.2>`_: Support Claude Fable and Sonnet 5. Allow thinking with forced tools where the model supports it.
* `1.6.3 <https://github.com/langchain-ai/langchain-aws/releases/tag/langchain-aws%3D%3D1.6.3>`_: Fix thinking and streaming for new Claude models. Support ``search_result`` blocks and managed knowledge bases. Improve model-provider detection.
* `1.6.4 <https://github.com/langchain-ai/langchain-aws/releases/tag/langchain-aws%3D%3D1.6.4>`_: Add Bedrock guardrails and per-request guardrail settings to ``ChatAnthropicBedrock``. Fix Opus 5 thinking and streaming.
* `1.7.0 <https://github.com/langchain-ai/langchain-aws/releases/tag/langchain-aws%3D%3D1.7.0>`_: Add ``ChatOpenAIMantle`` through the optional ``openai`` extra. Support ``reasoning_effort`` and simpler Anthropic prompt caching in Converse. Include the model name in tracing parameters.
* `1.7.1 <https://github.com/langchain-ai/langchain-aws/releases/tag/langchain-aws%3D%3D1.7.1>`_: Add ``DynamoDBVectorStore``. Support Kimi streaming in Converse.
* `1.7.2 <https://github.com/langchain-ai/langchain-aws/releases/tag/langchain-aws%3D%3D1.7.2>`_: Derive Mantle credentials from AWS credentials. Add ``ChatAnthropicMantle``.
* `1.7.3 <https://github.com/langchain-ai/langchain-aws/releases/tag/langchain-aws%3D%3D1.7.3>`_: Normalize assistant blocks from other providers before Converse serialization.
* `1.7.4 <https://github.com/langchain-ai/langchain-aws/releases/tag/langchain-aws%3D%3D1.7.4>`_: Reject unsupported Mantle guardrail headers. Fix Mantle authentication precedence and model-specific reasoning blocks.
* `1.7.5 <https://github.com/langchain-ai/langchain-aws/releases/tag/langchain-aws%3D%3D1.7.5>`_: Support Grok streaming and forced tools. Preserve zero temperature in traces. Add Mantle ``auth_mode``. Reduce prompt-cache trace inputs. Update the optional Nova Sonic SDK.
* `1.7.6 <https://github.com/langchain-ai/langchain-aws/releases/tag/langchain-aws%3D%3D1.7.6>`_: Repair structured-output arguments and JSON Schema handling. Update model profiles, reasoning effort, and tool choice for new models.
* `1.7.7 <https://github.com/langchain-ai/langchain-aws/releases/tag/langchain-aws%3D%3D1.7.7>`_: Support ``LANGSMITH_GATEWAY`` in Converse. Preserve invalid tool calls for replay. Reject incomplete streams. Skip cache points after reasoning.
* `1.7.8 <https://github.com/langchain-ai/langchain-aws/releases/tag/langchain-aws%3D%3D1.7.8>`_: Add a readable ``model`` property to Bedrock chat models. Apply explicit regions to supplied AWS sessions in the AWS checkpoint package.
* `1.7.9 <https://github.com/langchain-ai/langchain-aws/releases/tag/langchain-aws%3D%3D1.7.9>`_: Handle redacted-only reasoning. Escape MemoryDB filter values. Add AgentCore web search and a Nova Sonic model property.
* `1.8.0 <https://github.com/langchain-ai/langchain-aws/releases/tag/langchain-aws%3D%3D1.8.0>`_: Update sampling, thinking, and routing for current models. Prefer Responses for supported Mantle models. Move unsupported tool-result images. Reject query syntax in MemoryDB field names.
* `1.8.1 <https://github.com/langchain-ai/langchain-aws/releases/tag/langchain-aws%3D%3D1.8.1>`_: Raise ``ContextOverflowError`` for context-limit failures. Filter reasoning blocks from other providers.
* `1.8.2 <https://github.com/langchain-ai/langchain-aws/releases/tag/langchain-aws%3D%3D1.8.2>`_: Preserve invalid streamed tool arguments for replay. Add Haiku 5.5 support and mid-conversation model profiles. Align AgentCore code tools with Deep Agents 0.7.
* `1.9.0 <https://github.com/langchain-ai/langchain-aws/releases/tag/langchain-aws%3D%3D1.9.0>`_: Accept plain-text MIME blocks. Correct reasoning and cache-token usage. Fix nested actor lookup in the AWS checkpoint package.
* `1.9.1 <https://github.com/langchain-ai/langchain-aws/releases/tag/langchain-aws%3D%3D1.9.1>`_: Support Z.ai models and ``requestMetadata``. Use deterministic names for unnamed files. Correct model capability profiles. Fix AgentCore store round trips in the AWS checkpoint package.

Google integrations
-------------------

* `langchain-google-genai 4.4.1 <https://github.com/langchain-ai/langchain-google/releases/tag/libs%2Fgenai%2Fv4.4.1>`_: Omit Bedrock cache points from Gemini prompts. Treat Gemini 4 and later as current model families.
* `langchain-google-genai 4.4.2 <https://github.com/langchain-ai/langchain-google/releases/tag/libs%2Fgenai%2Fv4.4.2>`_: Normalize model names. Fix Gemini 3.5 cache paths and streaming usage. Refresh model profiles.
* `langchain-google-vertexai 3.2.4 <https://github.com/langchain-ai/langchain-google/releases/tag/libs%2Fvertexai%2Fv3.2.4>`_: Reflect per-call model overrides in traces. Refresh model profiles and dependency requirements.
* `langchain-google-vertexai 3.2.5 <https://github.com/langchain-ai/langchain-google/releases/tag/libs%2Fvertexai%2Fv3.2.5>`_: Remove the unused PyArrow dependency. Preserve citations from metadata-only stream chunks. Convert file blocks to Anthropic documents. Correct Anthropic profiles, Gemini cache paths, streaming usage, and request labels.

LangSmith SDK
-------------

* `0.14.8 <https://github.com/langchain-ai/langsmith-sdk/releases/tag/v0.14.8>`_: Send ``session_id`` with run patches. Reach the multipart-to-batch fallback after a 404 response. Update sandbox authentication and encrypted mounts.
* `0.14.9 <https://github.com/langchain-ai/langsmith-sdk/releases/tag/v0.14.9>`_: Send OTLP traces to an agent environment. Use ``TypedDict`` for ``AgentAddress``. Correct comparative evaluator scores. Reduce sandbox command latency.
* `0.14.10 <https://github.com/langchain-ai/langsmith-python/releases/tag/v0.14.10>`_: Move Python SDK releases to ``langchain-ai/langsmith-python``. The PyPI package and Python imports stay unchanged.
* `0.14.11 <https://github.com/langchain-ai/langsmith-python/releases/tag/v0.14.11>`_: Remove the temporary release-version pin. No toolkit API change is required.
* `0.14.12 <https://github.com/langchain-ai/langsmith-python/releases/tag/v0.14.12>`_: Preserve incomplete and failed Responses output in tracing. Add agent registry APIs and enforced feedback configuration. Correct example reference outputs.
* `0.14.13 <https://github.com/langchain-ai/langsmith-python/releases/tag/v0.14.13>`_: Wait for buffered run operations in ``Client.flush()``. Stop failed experiment iteration from hanging. Add sandbox volumes and destination resolution. Sandbox schema changes require review for direct sandbox consumers.
* `0.14.14 <https://github.com/langchain-ai/langsmith-python/releases/tag/v0.14.14>`_: Stream v2 run, trace, and thread queries. Accept addresses in wrappers and integrations. Include subtree usage in run queries.

FastMCP
-------

* `4.1.0 <https://github.com/PrefectHQ/fastmcp/releases/tag/v4.1.0>`_: Tighten filesystem paths, OpenAPI path parameters, regex search, and MultiAuth identities. Restore the SDK's 30-minute HTTP session idle timeout. Fix elicitation validation, schema pagination, result status, and cancellation.
* `4.1.1 <https://github.com/PrefectHQ/fastmcp/releases/tag/v4.1.1>`_: Bound regex search by pattern length and time. Restore linked skill-directory discovery. Fix JWKS refresh locking and proxy-setting handling.
* `4.1.2 <https://github.com/PrefectHQ/fastmcp/releases/tag/v4.1.2>`_: Check authorization after component transforms. Prevent denied-resource fallback. Bind OAuth client credentials to a configured issuer. Fix stdio UTF-8 mode, timeouts, schemas, and shared dependency scopes. Add experimental Skills discovery.
* `4.1.3 <https://github.com/PrefectHQ/fastmcp/releases/tag/v4.1.3>`_: Require current certifi bundles after the CA incident. Fix client-credential defaults, generator transforms, server timeouts, and timeout classification.
* `4.1.4 <https://github.com/PrefectHQ/fastmcp/releases/tag/v4.1.4>`_: Prevent stale client contexts from closing new sessions. Forward ``max_sse_event_size``. Fix proxy header precedence, mounted visibility, and generated commands. Support rotating task credentials and exclude Redis 8.1.0.

MCP SDK
-------

* `2.3.0 <https://github.com/modelcontextprotocol/python-sdk/releases/tag/v2.3.0>`_:
  Require ``httpx2>=2.10.0``. Add ``max_sse_event_size`` for HTTP clients.
  Keep its 1 MiB default. Refresh schemas and retry once after a
  ``HeaderMismatch`` rejection. Omit empty legacy metadata and parameters.
  Validate header annotations at server registration. Look up tool schemas
  without an extra list request. Exclude interactive OAuth login from request
  deadlines. Validate explicit null structured results.

Integration decisions
---------------------

The service keeps its current SSE and JSON Lines contracts.
Its executor still consumes ``AIMessageChunk`` values from graph streams.
The Core 1.7.0 change affects direct consumers of the v3 content-block protocol.
Applications that use that protocol must concatenate ``tool-call-delta`` slices.
Use ``.tool_calls.get()`` or ``.output`` for the complete result.

The model factory passes provider configuration to the selected native model.
For an OpenAI-compatible gateway that requires Chat Completions, set
``use_responses_api=false`` in its ``MODEL_CONFIGS`` entry.
For document inputs that need Responses, set ``use_responses_api=true``.
The dependency update does not change these explicit settings.

The blueprint supports native Deep Agents permissions, skills, memory sources,
and asynchronous subagent definitions. Existing defaults stay unchanged.
Asynchronous subagents need an external Agent Protocol service.
The toolkit REST service does not provide that protocol.
Review resolved filesystem paths when you use symlinks and permission rules.
Treat configured skill and memory source paths as trusted input.

FastMCP 4.1 changes server-side security checks and session ownership.
Finish pending MultiAuth tasks before you upgrade a separate FastMCP server.
For direct ``ClientCredentialsOAuthProvider`` or
``PrivateKeyJWTOAuthProvider`` use, configure ``issuer`` to enable issuer binding.
The toolkit's shared-header client configuration does not use these providers.
The toolkit closes protocol sessions after calls and reuses only HTTP pools.
It does not keep an idle protocol session for 30 minutes.
Use the per-server ``max_sse_event_size`` setting for larger HTTP SSE results.
Its default preserves the SDK's 1 MiB limit. See :doc:`mcp`.
FastMCP 4.2.0a1 is a pre-release. This update excludes it.

Rebuild images and update operating-system and custom certificate stores.
Updating certifi does not update those stores.
See the `FastMCP CA security notice
<https://github.com/PrefectHQ/fastmcp/blob/main/SECURITY.md#certificate-authority-incident-october-2026>`_.

``langchain-community==0.4.2`` remains the latest stable release.
The update does not replace the toolkit's authentication or memory scopes.
``user_id`` remains the long-term user scope.
``thread_id`` remains the short-term conversation scope.

LangGraph
---------

* `1.2.12 <https://github.com/langchain-ai/langgraph/releases/tag/1.2.12>`_: Add ``response_schema`` to ``interrupt()``. Detect subgraphs from bytecode when source is unavailable. Correct v3 stream projection types.
* `1.2.13 <https://github.com/langchain-ai/langgraph/releases/tag/1.2.13>`_: Remove answered interrupts from saved state responses. Isolate updates and replay on older checkpoint branches. Preserve delta counters and use the selected saver for subgraphs.
* `1.2.14 <https://github.com/langchain-ai/langgraph/releases/tag/1.2.14>`_: Publish the next patch version. The release notes list no additional behavior change.
* `1.2.15 <https://github.com/langchain-ai/langgraph/releases/tag/1.2.15>`_: Preserve control and interrupt events in v1/v2 streams. Save cached-node writes and state-update writes on the correct checkpoint. Correct delta replay order, message IDs, fork snapshots, and failed-write handling. Propagate invalid tool resume values.
* `1.2.16 <https://github.com/langchain-ai/langgraph/releases/tag/1.2.16>`_: Apply a resumed ``Command.update`` once. Keep completed subgraph tasks from running again. Save pending exit-mode writes once. Let other tasks finish when a resume value fails validation.
* `1.2.17 <https://github.com/langchain-ai/langgraph/releases/tag/1.2.17>`_: Apply an explicit recursion limit when it equals the default. Preserve the caller namespace in remote subgraph streams.
* `1.2.18 <https://github.com/langchain-ai/langgraph/releases/tag/1.2.18>`_: Run error handlers on all supported paths. Stream failed task results and the final checkpoint after a breakpoint or recursion limit. Prevent repeated subgraph tags. Change the default delta snapshot interval from 1,000 updates to 100 updates.

Checkpoint packages
-------------------

* `langgraph-checkpoint 4.3.0 <https://github.com/langchain-ai/langgraph/releases/tag/checkpoint%3D%3D4.3.0>`_: Validate store namespaces before dispatch. Preserve serialized data when an object cannot be restored. Correct in-memory delta history and replay order. Accept mapping values in stores.
* `langgraph-checkpoint 4.3.1 <https://github.com/langchain-ai/langgraph/releases/tag/checkpoint%3D%3D4.3.1>`_: Reject NUL bytes in store namespace labels.
* `langgraph-checkpoint-postgres 3.1.3 <https://github.com/langchain-ai/langgraph/releases/tag/checkpointpostgres%3D%3D3.1.3>`_: Start delta reads at the requested checkpoint. Treat an empty checkpoint ID as the latest checkpoint. Validate namespaces and correct delta replay order.
* `langgraph-checkpoint-postgres 3.2.0 <https://github.com/langchain-ai/langgraph/releases/tag/checkpointpostgres%3D%3D3.2.0>`_: Add native ``prune`` and ``aprune`` methods. Keep the latest checkpoint by default. Support explicit thread deletion.
* `langgraph-checkpoint-sqlite 3.1.2 <https://github.com/langchain-ai/langgraph/releases/tag/checkpointsqlite%3D%3D3.1.2>`_: Add task paths to saved writes. Follow parent pointers when reading delta history. Correct write order and empty checkpoint IDs. Validate store namespaces.
* `langgraph-prebuilt 1.1.1 <https://github.com/langchain-ai/langgraph/releases/tag/prebuilt%3D%3D1.1.1>`_: Propagate invalid resume values from ``ToolNode``. Keep the graph available for a later valid answer.

LangGraph SDK
-------------

* `0.4.5 <https://github.com/langchain-ai/langgraph/releases/tag/sdk%3D%3D0.4.5>`_: Add typed interrupt response schemas to SDK data models. Update dependencies.
* `0.4.6 <https://github.com/langchain-ai/langgraph/releases/tag/sdk%3D%3D0.4.6>`_: Encode thread and assistant IDs in stream request URLs.
* `0.4.7 <https://github.com/langchain-ai/langgraph/releases/tag/sdk%3D%3D0.4.7>`_: End a thread stream only on a root run lifecycle event.
* `0.4.8 <https://github.com/langchain-ai/langgraph/releases/tag/sdk%3D%3D0.4.8>`_: Report thread stream errors. Stop retries after a rejected stream request.
* `0.4.9 <https://github.com/langchain-ai/langgraph/releases/tag/sdk%3D%3D0.4.9>`_: Add thread priority settings. Relax third-party upper bounds.
* `0.4.10 <https://github.com/langchain-ai/langgraph/releases/tag/sdk%3D%3D0.4.10>`_: Retry ``runs.wait`` with a bodyless GET. Raise errors from synchronous wait calls. Correct multi-line SSE data, empty IDs, and a leading byte order mark. Accept durability, resumable streaming, and context in typed overloads.

LangGraph CLI
-------------

* `0.4.30 <https://github.com/langchain-ai/langgraph/releases/tag/cli%3D%3D0.4.30>`_: Accept compatible API version ranges.
* `0.4.31 <https://github.com/langchain-ai/langgraph/releases/tag/cli%3D%3D0.4.31>`_: Permit API versions below 1.0.0. Deploy images from a previous build.
* `0.4.32 <https://github.com/langchain-ai/langgraph/releases/tag/cli%3D%3D0.4.32>`_: Select listeners and agent environments for deployments. Accept an image URI for self-hosted deployments. Clarify missing deployment configuration.
* `0.4.33 <https://github.com/langchain-ai/langgraph/releases/tag/cli%3D%3D0.4.33>`_: Deploy an image that is already in a registry. List deployment listeners. Reject Git dependency URLs that contain credentials.
* `0.4.34 <https://github.com/langchain-ai/langgraph/releases/tag/cli%3D%3D0.4.34>`_: Return listener lists as JSON with ``--json``.
* `0.4.35 <https://github.com/langchain-ai/langgraph/releases/tag/cli%3D%3D0.4.35>`_: Accept private registry image pull secrets. Use the returned registry username. Keep build logs after a deployment failure.
* `0.4.36 <https://github.com/langchain-ai/langgraph/releases/tag/cli%3D%3D0.4.36>`_: Accept deployment type aliases. Push images through the active Docker context.
* `0.4.37 <https://github.com/langchain-ai/langgraph/releases/tag/cli%3D%3D0.4.37>`_: Use serverless deployments by default. Exclude subprocess standard input from verbose logs.
* `0.4.38 <https://github.com/langchain-ai/langgraph/releases/tag/cli%3D%3D0.4.38>`_: Accept priority configuration in ``langgraph.json``.
* `0.4.39 <https://github.com/langchain-ai/langgraph/releases/tag/cli%3D%3D0.4.39>`_: Return deployment lists as JSON with ``--json``.
* `0.4.40 <https://github.com/langchain-ai/langgraph/releases/tag/cli%3D%3D0.4.40>`_: Send upload headers from the deployment control plane.

The stable ``langgraph-supervisor`` version remains 0.0.31.
This update keeps the existing supervisor interface.

Checkpoint upgrade operations
-----------------------------

Back up the database before this update.
Stop active requests before you update SQLite workers.
SQLite saver setup adds ``writes.task_path`` to databases from version 3.1.1.
The migration keeps existing rows.
Do not mix old and new SQLite saver versions during this change.
The regression fixture uses a database from the released 3.1.1 saver.
It verifies a pending interrupt and a completed parallel branch after upgrade.

The PostgreSQL saver migration list is unchanged from 3.1.2 to 3.2.0.
The toolkit still runs saver setup under its schema lock.
Run checkpoint migrations and resume checks before you replace all workers.

Store namespace labels cannot contain periods, NUL bytes, or empty strings.
Validation now also covers reads, searches, and direct batch operations.
Encode user identifiers before you use them as namespace labels.
The toolkit's scoped checkpoint keys already use safe labels.
Keep long-term user memory separate from conversation checkpoints.

The new PostgreSQL ``aprune`` default is ``strategy="keep_latest"``.
It keeps current state and pending interrupts for supported threads.
It refuses ``DeltaChannel`` threads because they need older writes.
Deep Agents uses these channels.
The toolkit does not run this maintenance method automatically.
Do not replace checkpoint deletion with pruning for Deep Agents threads.
External media blobs still need their own retention policy.

LangGraph 1.2.18 takes a delta snapshot after 100 updates by default.
This can reduce restore time and increase database storage.
Set ``snapshot_frequency=1000`` on a custom ``DeltaChannel`` to keep its old interval.
An explicit interval on an upstream channel still takes precedence.

Typed interrupts use native schema validation.
The toolkit returns ``response_schema`` in live and recovered interrupt records.
Send answers as ``input.resume`` values keyed by interrupt ID.
The order of parallel interrupt records can differ between live and saved responses.
Use IDs to match questions and answers.
A Python response type enables native validation.
A JSON Schema dictionary describes the response without enabling that validation.

Legacy chains and text splitters
--------------------------------

* `langchain-classic 1.0.8 <https://github.com/langchain-ai/langchain/releases/tag/langchain-classic%3D%3D1.0.8>`_:
  Accept content-block tokens in streaming callbacks. Correct runnable type
  annotations. The toolkit does not need an interface change.
* `langchain-text-splitters 1.1.3 <https://github.com/langchain-ai/langchain/releases/tag/langchain-text-splitters%3D%3D1.1.3>`_:
  Restore lazy imports for large optional dependencies. Reject invalid inputs
  to ``RecursiveJsonSplitter`` with ``TypeError``. Correct separators for
  Kotlin, Rust, Haskell, C#, and Elixir.

Optional Studio server and runtime
----------------------------------

The ``studio`` extra installs ``langgraph-cli[inmem]``.
It now selects ``langgraph-api==0.16.7`` and
``langgraph-runtime-inmem==0.36.7``.
The previous versions were 0.10.3 and 0.30.3.
These packages run the separate Studio development server.
They do not replace the toolkit's FastAPI server, checkpointer, or request queue.

The `official Agent Server changelog
<https://docs.langchain.com/langsmith/agent-server-changelog>`_ documents the
API release families. The summaries below include fixes added to each family.

* `0.11 <https://docs.langchain.com/langsmith/agent-server-changelog#v0-11>`_:
  Add delta-aware pruning and Prometheus metrics. Correct durable interrupt
  resume and conditional-thread authorization. Metrics names and units change.
* `0.12 <https://docs.langchain.com/langsmith/agent-server-changelog#v0-12>`_:
  Add search cost limits and gRPC stores. Fix store authorization filters,
  resume races, subgraph events, and namespace matching.
* `0.13 <https://docs.langchain.com/langsmith/agent-server-changelog#v0-13>`_:
  Support Python 3.14 and A2A file output. Preserve cancellation across worker
  failures. Fix thread authorization and queue port conflicts.
* `0.14 <https://docs.langchain.com/langsmith/agent-server-changelog#v0-14>`_:
  Support custom encryption with gRPC checkpoints. Fix scheduler duplication,
  lock contention, delta restoration, auth checks, and cancellation.
* `0.15 <https://docs.langchain.com/langsmith/agent-server-changelog#v0-15>`_:
  Add A2UI and task-only A2A history. Fix false interrupt results, HTTPS
  webhooks, gRPC replay, startup failures, and tracing dependencies.
* `0.16 <https://docs.langchain.com/langsmith/agent-server-changelog#v0-16>`_:
  Add typed interrupts, storage diagnostics, and dependency health details.
  Preserve pending questions and retry killed in-memory runs after restart.
  Correct delta history and transient-error streaming.
  Version 0.16.7 fixes factory-graph state operations with gRPC checkpoints.

The API metadata selects the matching in-memory runtime family:

.. list-table:: Studio API and runtime families
   :header-rows: 1
   :widths: 25 25 50

   * - API family
     - Runtime family
     - Official metadata
   * - 0.11.x
     - 0.31.x
     - `API 0.11.0 <https://pypi.org/project/langgraph-api/0.11.0/>`_
   * - 0.12.x
     - 0.32.x
     - `API 0.12.0 <https://pypi.org/project/langgraph-api/0.12.0/>`_
   * - 0.13.x
     - 0.33.x
     - `API 0.13.0 <https://pypi.org/project/langgraph-api/0.13.0/>`_
   * - 0.14.x
     - 0.34.x
     - `API 0.14.0 <https://pypi.org/project/langgraph-api/0.14.0/>`_
   * - 0.15.x
     - 0.35.x
     - `API 0.15.0 <https://pypi.org/project/langgraph-api/0.15.0/>`_
   * - 0.16.x
     - 0.36.x
     - `API 0.16.7 <https://pypi.org/project/langgraph-api/0.16.7/>`_

The `runtime 0.36.7 metadata
<https://pypi.org/project/langgraph-runtime-inmem/0.36.7/>`_ has no separate
patch changelog. Its review uses the API changelog and published requirements.
It does not establish a patch-by-patch runtime change list.
The selected API requires ``grpcio-tools==1.81.1``, compatible gRPC 1.81
packages, and ``protobuf>=6.32.1,<7.0.0``.
It also requires ``anyio>=4.14.2`` and ``starlette>=1.3.1``.
The lockfile resolves these requirements together.
Do not independently pin the runtime to an incompatible API family.

The verification covers dependency resolution, CLI help, and SDK contracts.
It does not verify a running Studio server or a remote Agent Server deployment.
Test graph loading, persisted local state, and pending questions in Studio
before you replace an existing Studio environment.
The 0.17 API family remains a release candidate at this review date.
Its optional run-drain handoff is excluded from this stable update.
