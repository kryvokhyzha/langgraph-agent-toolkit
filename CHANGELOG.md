# Changelog

This file records all notable project changes.

This file uses the [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
format. The project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.10.1]

### Changed

- Match managed LLM connection limits, phase timeouts, and retries to OpenAI SDK
  3.13.0 defaults. Allow 1,000 connections and 100 idle connections per pool.
  Use a 5-second idle expiry and connect timeout. Use 600-second read, write,
  and pool timeouts. Keep the separate API request deadline.
- Default `AUTH_MODE` to `trusted` to preserve 0.9.2 shared-token
  authentication. Keep `user_id` optional. Accept a supplied user ID with
  `AUTH_SECRET`. Keep strict identity binding available through
  `AUTH_MODE=token`.
- Reduce Dependabot version-update PR limits and group all hook revisions. Check
  Actions and hooks monthly. Keep Python and Docker checks weekly. Keep security
  updates separate from the routine schedule and cooldown.
- Run configured pre-commit hooks on pull requests. Exercise Python hooks when
  only hook configuration changes. Prepare feedback routes before testing a
  short request timeout under coverage.
- Add `CHECKPOINT_DURABILITY` with the default `sync`. Finish each checkpoint
  before the next graph step starts. Use `async` to overlap saves and graph
  steps when the larger crash window is acceptable. Both modes use async
  database methods and propagate checkpoint write errors.
- Delegate SQLite lock polling to `filelock.AsyncFileLock`. Separate PostgreSQL
  lock acquisition, monitoring, and cleanup. Preserve cancellation received
  during SQLite rollback after a database error.

### Fixed

- Apply zero PostgreSQL timeout values explicitly to disable inherited limits.
  Report the actual minimum and maximum sizes for each database pool.
- Preserve prior messages when importing into a graph without a message reducer.
  Report unsupported streamed output instead of silently skipping it.
- Preserve interleaved progress and tool-result events in the UI. Keep failed
  input separate from saved history and reload checkpoints after a client error.

### Documentation

- Explain the `AUTH_MODE` default and shared-token migration settings.
- Explain checkpoint durability modes and their recovery limits.
- Separate workload tuning from advanced settings. Clarify SQL timeout scope,
  connection budgets across replicas, and request shutdown limits.

## [0.10.0]

### Changed

- Add Python 3.14 support and CI coverage. Require Pydantic 2.13.0 or later on
  this interpreter. Use a compatible native JSON schema dependency for Studio.
- Run process tests with package coverage. Check the lockfile once, retain test
  reports, and validate releases before publication. Add native uv, Docker, and
  pre-commit updates to Dependabot. Send one complete report to Codecov.
- Update `actions/checkout` and `actions/setup-python` to v7 in all workflows.
- Reduce the default Uvicorn worker heartbeat allowance to 10 seconds after
  deferring launcher imports. Keep an explicit override for slower startup.

- Bound active HTTP requests before buffering bodies. Return 503 when worker
  capacity is full. Apply one request deadline through streaming and cleanup.
- Reuse service-owned OpenAI and Azure HTTP connections with explicit pool and
  timeout settings. Support `DefaultAioHttpClient` through `openai-aiohttp`. See
  [the reliability guide](docs/reliability.rst) for limits and tuning.

- Permit the provider and LangSmith versions required by Deep Agents 0.7.13.
  Keep the existing LangChain, Core, and LangGraph version ranges.
- Share native agent setup and client request parsing. Consolidate repeated
  dependency constraints and remove the unused `duckduckgo-search` dependency.
- Replace duplicate and import-only assertions with tests of graph execution,
  exported telemetry, and HTTP behavior. See
  [the testing guide](docs/testing.rst) for test layers and commands.
- Bind API conversations to the authenticated user, agent, and public thread ID.
  Use one `AUTH_SECRET` with `AUTH_MODE=trusted` for each client deployment
  whose trusted application supplies `user_id`. Individual user tokens remain
  optional. See [the migration guide](docs/migration.rst) before upgrading
  stored conversations.
- Serialize conversation operations across workers with PostgreSQL advisory
  locks or SQLite file locks. Bound queue waiting, request duration, and body
  size.
- Use the configured service checkpointer for built-in agents. Fail worker
  startup when required resources cannot initialize. Readiness checks database
  connectivity.
- Paginate history. Clear-history now deletes all checkpoints for the
  conversation.
- Reuse client HTTP connections and expose stream failures as
  `AgentClientError`. Close owned clients with context managers or `close()` /
  `aclose()`.
- Move Streamlit and LangGraph Studio to `ui` and `studio` extras. Remove unused
  direct dependencies and narrow core compatibility to tested API versions.
- Update LangGraph to 1.2.11, LangChain to 1.4.0, and the default Langfuse SDK
  to 4.15.2. Update their checkpoint and OpenAI integrations. See
  [the dependency review](docs/dependency_updates.rst) for release changes.

### Fixed

- Bind token-user feedback to the authenticated user, agent, and run with
  server-signed response proof. Require a separate `FEEDBACK_SIGNING_SECRET` for
  this mode. Keep trusted backend feedback compatible. See
  [the migration guide](docs/migration.rst) for client changes.
- Restore Streamlit conversations through the URL-selected agent. Update the URL
  after new chats and agent changes. Build resume links with the public
  Streamlit URL API and preserve the deployment path. Load all history pages
  before displaying a saved conversation. Stop chat input if any page fails.
- Extract text from content blocks before knowledge-base retrieval. Preserve
  Bedrock source locations and nested document titles in the augmented prompt.
- Preserve HTTP status, service error codes, and `Retry-After` in
  `AgentClientError`. Bound rejected-stream body reads. Disable proxy buffering
  and caching for SSE responses.
- Preserve streamed OpenAI and Azure refusal text and provider token usage in
  API messages and imported history. Reject incomplete SSE responses in the
  Python client, including clean EOF without the completion marker.
- Return a fixed 503 error for rejected model-provider credentials. Keep
  provider authentication error details out of API responses and toolkit logs in
  all environment modes.
- Return explicit 429, 503, and 504 responses for provider rate limits,
  connection failures, and timeouts. Normalize interrupted OpenAI and Azure
  streams at the model boundary without retrying received tokens.

- Stop abandoned invoke requests and close stream generators in their original
  context. Keep conversation locks until graph and checkpoint cleanup finish.
  Mark health probes unhealthy when cancellation cleanup stalls.
- Bound feedback threads after HTTP cancellation and bound owned HTTP-client
  shutdown without leaving default-executor threads running.
- Roll back cancelled SQLite writes. Remove SQLite iterator and PostgreSQL
  startup migration deadlocks. Include health checks in pool checkout deadlines,
  close uncertain sessions on cancellation, and permit concurrent pooled reads.
- Consume invoke state incrementally instead of retaining every full graph
  state.

- Keep internal summaries and retained history out of the answer stream for Deep
  Agents and native LangChain summarization middleware.
- Keep client and UI imports available without optional agent dependencies.
- Omit unset history query parameters. Accept SSE comments and keepalive frames
  without ending the stream. Report malformed stream events as
  `AgentClientError`.
- Read the complete saved history of Functional API agents. Append imported
  messages without replacing earlier turns or other saved fields. Preserve
  existing checkpoints and emit only the new answer during streaming.
- Require `thread_id` in client history operations. Keep short-term checkpoint
  history separate from long-term stores addressed by a stable `user_id`.
- Reject nested coordinated calls before they can wait for their parent. Release
  checkpoint iterator cursors before yielding. Discard uncertain PostgreSQL lock
  sessions after cancellation, and release locks when streams close in another
  context.
- Bound telemetry shutdown, so a stalled flush cannot prevent worker exit.
- Support Langfuse SDK v2 with current LangChain callbacks. Bind SDK v3 and v4
  callbacks to the configured project. Preserve trace identities, root output,
  and feedback links across concurrent runs. See
  [the Langfuse compatibility guide](docs/langfuse_compatibility.rst) to select
  an SDK for each server version.
- Isolate model defaults and credentials between factory calls. Validate
  settings overrides and preserve constructor and dotenv configuration values.
- Repair history add/clear, tool calls in the custom creator, standalone
  executor inputs, async feedback, and the Azure Functions request/lifespan
  adapter.
- Preserve configured prompts during fallback, restore sandboxed Jinja
  rendering, honor pinned prompt versions, and clear stale retrieval context on
  misses.
- Coordinate prompt refreshes and move blocking prompt/feedback work off the
  event loop.
- Escape PostgreSQL credentials and bind checkpoint writes to the
  conversation-lock session, so a lost connection cannot continue writes through
  a replacement session.

### Added

- Add a verified local quickstart with a fake model and SQLite. Update the
  README, setup guides, and API usage examples. Generate the complete Python API
  reference and fail documentation builds on warnings.
- Install and select `openai-aiohttp` by default in the API Docker image. Keep
  HTTPX available through `LLM_HTTP_ASYNC_TRANSPORT=httpx`. Python installations
  outside the image retain HTTPX as their default.
- Add opt-in real OpenAI journeys for HTTPX and aiohttp, with fixed request and
  output limits. Check tools, concurrent streams, token usage, disconnects, and
  recovery. See [the live model guide](docs/live_llm_testing.rst).
- Add repeatable API load tests with a local model simulator, real database
  persistence, worker replacement, slow readers, fault injection, and soak
  measurements. See [the load-testing guide](docs/load_testing.rst).

- Published `langfuse-v2`, `langfuse-v3`, and `langfuse-v4` extras to select one
  Python SDK major. Keep the existing `langfuse` extra for compatibility. Use
  `uv sync --extra all` instead of `--all-extras`, because the version selectors
  cannot be installed together.
- Optional `deepagents` integration with planning, virtual files, subagents,
  approvals, MCP tools, and the configured service checkpointer. Add an offline
  SQLite example, integration guide, and a comparison of supported approaches.
- Optional Deep Agents backend image build with `INSTALL_DEEPAGENTS=true`.
- API process journeys for authentication, user memory, thread history,
  concurrent turns, durable restarts, and interrupt resume.
- Real Langfuse SDK export tests for v2, v3, and v4. Add an opt-in live server
  matrix that checks stored prompts, native and Deep Agent model/tool traces,
  subagent parent links, and feedback.
- Container tests for API history and frontend health. Include subprocesses in
  whole-package coverage and enforce a 77% combined line and branch minimum.
- Optional MCP tools through `langchain[mcp]` and FastMCP 4. Configure HTTP or
  stdio servers, credentials, tool allowlists, and agent assignments. Discover
  tools before readiness and close connections after use. See
  [the MCP guide](docs/mcp.rst).
- MCP human-input requests through invoke and streaming responses. Preserve all
  interrupt IDs and validate single or parallel resume answers.
- Gitleaks v8.30.1 pre-commit hook. Redact detected credentials in its output.
- Explicit, transactional checkpoint ownership migration and inactive-thread
  retention commands. Both default to dry runs.
- Regression tests for real HTTP/schema contracts, durable restarts, concurrent
  turns, cancellation, database reconnection, and worker replacement.

## [0.9.2]

### Fixed

- `TrimMessagesMiddleware` / `TokenTrimMiddleware` now keep the model's view
  when the **latest turn does not fit the budget**. `trim_messages`
  (`strategy="last"`, `start_on="human"`) returns `[]` when the current turn
  exceeds the budget. This can occur when one user message exceeds the token
  budget. It can also occur when a turn contains more tool calls than the
  message budget permits. Previously, this result invoked the model with only
  the system prompt. The model did not receive the user's question. Both
  middlewares now keep the latest turn unchanged. The turn can exceed the
  budget, but the model can answer it. Regression tests cover an oversized user
  message, an oversized tool result, and a burst of 6+ tool calls.

### Added

- `TokenTrimMiddleware` limits the model's message view by a **token** budget.
  It complements `TrimMessagesMiddleware`, which limits the view by message
  count. It changes only the view. The full history stays in state, and the
  system prompt stays in the view. Configure the token counter with
  `token_counter`. The default counter is `count_tokens_approximately`, which
  adds no dependency. For an exact count, pass a `tiktoken`-backed counter or
  the model. The new `DEFAULT_MAX_TOKENS_HISTORY_LENGTH` setting defines the
  default budget. This setting has no default value because the correct budget
  depends on the model. Put `TokenTrimMiddleware` after `TrimMessagesMiddleware`
  to limit the message count and token count.

## [0.9.1]

### Added

- Reusable `create_agent` middleware adds the custom `create_react_agent`
  history features to the native LangChain `create_agent`. It uses only the
  public API. `ImmediateGenerationMiddleware` handles the model-call budget. On
  the last permitted call, it removes tools and asks the model for a direct
  answer. This action prevents a stall. `SanitizeHistoryMiddleware` repairs
  broken tool-call and result pairs before the model call.
  `ClearIntermediateToolCallsMiddleware` reduces token use. It keeps only the
  most recent results for each tool from earlier turns. Configure it with the
  constructor or the `CLEAR_INTERMEDIATE_TOOL_CALLS*` settings. By default, the
  safe `name_args` deduplication key combines only identical repeated calls.
  `TrimMessagesMiddleware` limits the view. It keeps the last
  `DEFAULT_MAX_MESSAGE_HISTORY_LENGTH` messages in the model input and keeps the
  full state. The `create_agent` blueprint shows these features.
- The `hitl_agent` blueprint adds human-in-the-loop tool approval to the native
  `create_agent` through `HumanInTheLoopMiddleware`. The executor uses
  `build_resume_command` to resume. It maps replies of `approve`,
  `reject: <reason>`, or free text to the middleware decision format. Custom
  `interrupt()` blueprints do not change.
- `UserComplexInput.message` accepts multimodal LangChain content blocks. It
  accepts text, image, file, audio, and video blocks through a URL or base64.
  `MULTIMODAL_MAX_ATTACHMENTS` sets the attachment limit for each message. The
  Streamlit chat supports file and image uploads.

### Changed

- The Dockerfiles (`docker/api`, `docker/app`) now use multiple stages, uv cache
  mounts, and dependency-before-source layers. Code changes no longer reinstall
  dependencies, so rebuilds are faster. The changes also make images smaller and
  reduce the build context.
- Renamed and removed duplicate ReAct example blueprints. Renamed `react_new` to
  `create_agent` and `react_so` to `create_agent_structured`. Removed
  `react_old`, which duplicated `react`, and removed `react_so_old`. The
  examples are now grouped by builder: toolkit `create_react_agent` or native
  `create_agent`.
- `sanitize_chat_history` now operates in both directions. It removes unanswered
  tool calls. It also removes orphaned ToolMessages when the related tool call
  is gone, such as after trimming or summarization. This change applies to the
  custom `create_react_agent` and `SanitizeHistoryMiddleware`.
- Moved the Streamlit UI from one `run_app.py` file to a `ui/` package. The
  package contains `main_page`, `components/`, and `utils/`. The `run_app.py`
  file is now a small entry point.

### Fixed

- Human-in-the-loop interrupts now resume by default because `CHECK_INTERRUPTS`
  defaults to `True`. Previously, the executor did not resume them.
- Streaming now returns an agent's `structured_response` (`response_format`).
  This behavior now matches `invoke`.

## [0.9.0]

### Added

- Added the JSON Lines (NDJSON) streaming endpoints `/stream/jsonl` and
  `/{agent_id}/stream/jsonl`. They provide a typed alternative to SSE and use a
  `StreamChunk` response schema.
- Added the `AgentClient.stream_jsonl` / `astream_jsonl` SDK methods.
- Added a Streamlit option to select SSE or JSON Lines as the streaming
  protocol.
- Added idiomatic OpenAPI operation IDs and grouped `openapi_tags`. Documented
  the 401/404/422/429/500/503 error responses with an `ErrorResponse` schema.

### Changed

- Migrated the Langfuse integration to the v4 SDK. It remains compatible with
  v3.
- Upgraded to the LangChain 1.x / LangGraph 1.x APIs. These APIs include
  `create_agent`, `context_schema`, the new `SummarizationMiddleware`, and the
  supervisor APIs.
- Production error responses no longer contain internal exception details.
  `ENV_MODE` controls this behavior.
- The supervisor blueprint uses `output_mode="full_history"`. Sub-agent messages
  now remain after history reloads.
- In `docker-compose`, the backend now waits for a healthy Langfuse service
  before it starts.

### Fixed

- Langfuse now records trace-level output. The v4 callback change caused this
  regression.
- The Kubernetes startup probe now passes during a degraded start and prevents a
  CrashLoop. Shutdown now clears the DB pool and readiness flags.
- `/health/db` no longer reports an error with the SQLite backend.
- `401` responses now include a `WWW-Authenticate: Bearer` header.
- Fixed the `app_prefix` TypeError in `PostgresMemoryBackend.get_store`.
- Fixed various LangGraph / Starlette deprecation warnings.

## [0.8.15]

### Fixed

- Fixed the `add_messages` and `clear_history` endpoints.
- Fixed `[-1]` handling.

## [0.8.14]

### Added

- Added Cors settings.

## [0.8.13]

### Added

- Added new default settings.

### Updated

- Updated callback creation.
- Updated the project structure.

## [0.8.12]

### Added

- Added new API healthchecks.

### Update

- Updated the main logger.

## [0.8.11]

### Fixed

- Fixed handling for an AIMessage that has tool calls without a ToolMessage.

## [0.8.10]

### Updated

- Updated the default Postgres settings.

## [0.8.9]

### Updated

- Refactored the observability class.
- Refactored the prompt manager class.
- Fixed the uvicorn setup.

## [0.8.8]

### Fixed

- Fixed a bug to support `langfuse < 2.70.0`.

## [0.8.7]

### Fixed

- Fixed a bug to support `langfuse < 2.70.0`.

## [0.8.6]

### Fixed

- Fixed a bug to support `langfuse < 2.70.0`.

## [0.8.5]

### Updated

- Updated Core Dependencies to support `langchain < 1.0.0`.

## [0.8.4]

### Added

- Added new postgres settings.
- Added a DB healthcheck API.

### Fixed

- Fixed support for the old langfuse sdk.

## [0.8.3]

### Updated

- Updated the configuration setting structure.

## [0.8.2]

### Added

- Added a factory for embedding models.

## [0.8.1]

### Fixed

- Changed langfuse `score` to `create_score`.

### Updated

- Updated Core Dependencies.

## [0.8.0]

### Added

- Added a `create_agent` example.

### Updated

- Updated Core Dependencies.

### Fixed

- Fixed Langfuse.
- Fixed stream mode in the `invoke` method.

## [0.7.23]

### Updated

- Updated the default logging configuration.

## [0.7.22]

### Added

- Added NoOpSaver for checkpointing.

### Updated

- Updated the logging configuration.

## [0.7.21]

### Updated

- Removed the caching decorator from the create method.

## [0.7.20]

### Updated

- Improved the logging configuration.
- Improved message parsing.

## [0.7.19]

### Updated

- Improved model parameter values and factory model creation logic.

## [0.7.18]

### Added

- Added the `SKIP_REDIRECTION_LOGGING` environment variable and improved the
  logging middleware.

## [0.7.17]

### Fixed

- Fixed error handling.
- Fixed tests.

## [0.7.16]

### Fixed

- Fixed the type of the `content` field in the `ChatMessage` model.

### Added

- Added new tests.

## [0.7.15]

### Added

- Added `remote first` logic for the observability platform.

### Updated

- Updated the UI.
- Added support to read some default values from env vars.

## [0.7.14]

### Updated

- Made the `message` field optional.
- Updated the default value for `MEMORY_BACKEND`.
- Refactored the `lifespan` function.

## [0.7.13]

### Added

- Added `DB_CONFIGS` initialization.

## [0.7.12]

### Updated

- Updated dependencies.

## [0.7.11]

### Updated

- Updated dependencies.

## [0.7.10]

### Fixed

- Fixed error handling in `message_generator`.
- Fixed the max_messages type.

## [0.7.9]

### Fixed

- Fixed the graph type (create_react_agent).

## [0.7.8]

### Added

- Added a prompt manager.
- Added Utils functions.

### Updated

- Updated dependencies.

## [0.7.7]

### Fixed

- Fixed error handling.

## [0.7.6]

### Fixed

- Added a schema for the postgres db.

## [0.7.5]

### Fixed

- Downgraded langfuse.

## [0.7.4]

### Updated

- Fixed the Langfuse Callback import.

## [0.7.3]

### Added

- Added a new env variable for the model config (base64).

## [0.7.2]

### Added

- Added a new env variable for the model config (file).

## [0.7.1]

### Fixed

- Updated the agent executor test mock assertion for the additional
  `environment` and `tags` parameters.
- Added the missing `@pytest.mark.asyncio` decorator to asynchronous test
  methods in the prompt tests.

### Updated

- Updated dependencies to the latest versions.
- Updated the agent executor `get_callback_handler` method to pass these
  parameters:
  - Added the `environment` parameter from settings.ENV_MODE.
  - Added the `tags` parameter with the agent name for observability tracking.

### Improved

- Improved Langfuse observability prompt hash detection with a fallback:
  - Updated the `push_prompt` method to use tags when commit_message is empty.
  - Added `hasattr` checks for the `commit_message` and `tags` attributes.
  - Updated logging to show the old and new hash values.

## [0.7.0]

### Fixed

- Fixed the React Agent with SO.

### Updated

- Updated dependencies.
- Updated the names of the default configurable parameters.

## [0.6.0]

### Fixed

- Fixed the React Agent with SO.

### Added

- Added complex input.

### Updated

- Updated dependencies.
- Updated error handling.
- Updated tests.

## [0.5.0]

### Fixed

- Fixed Streamlit UI bugs.
- Fixed a Windows compatibility issue.
- Improved message handling in `pre_hook_model`.
- Added a prompt hash to the Langfuse observability class.
- Renamed some parameters.

## [0.4.5]

### Fixed

- Fixed structured output and the model factory.

## [0.4.4]

### Updated

- Updated project dependencies.

### Fixed

- Fixed structured output and the model factory.

## [0.4.3]

### Updated

- Updated project dependencies.

## [0.4.2]

### Fixed

- Fixed a streaming bug.
- Fixed the Streamlit welcome message display.
- Fixed client error handling.
- Fixed package dependencies.

## [0.4.1]

### Updated

- Aligned the Client API with the server endpoints.
- Added parameters to the invoke and stream methods.

### Added

- Added message management methods to the client (add_messages, aadd_messages).
- Added chat history retrieval methods (get_history, aget_history).
- Added history clearing methods (clear_history, aclear_history).
- Added the synchronous feedback creation method (create_feedback).
- Added support for the model_config_key parameter.
- Added support for the recursion_limit parameter.

### Fixed

- Updated client tests to mock API endpoints correctly.
- Fixed parameter handling in the stream and invoke methods.

## [0.4.0]

### Updated

- Updated endpoints.

### Added

- Added an endpoint to clear history.
- Added an endpoint to add a message to the history.

### Fixed

- Made minor fixes and refactored code.

## [0.3.1]

### Added

- Added the ability to pass parameters to the service runner.
- Added an argument to select the service runner.

### Updated

- Updated the service Dockerfile.

## [0.3.0]

### Added

- Added `MODEL_CONFIGS` to unify LLM env variables.
- Added a blueprint with AWS KB.

### Fixed

- Fixed streaming message handling.
- Refactored the code structure to make maintenance easier.
- Fixed optional dependencies.
- Fixed API exception handling.

## [0.2.0]

### Fixed

- Refactored the code structure to make maintenance easier.
- Refactored the Model factory.

### Removed

- Removed AllModels. Added environment variables for different providers.

## [0.1.2]

### Added

- Added the `user_id` parameter.
- Added the `store` creator to memory classes.

### Fixed

- Improved error handling and testing in the Streamlit app.
- Added a new chat button.
- Fixed variable names.
- Fixed type hints.

### Removed

- Removed print statements.

## [0.1.1]

### Added

- Added the `get_default_agent` and `set_default_agent` functions.

### Fixed

- Made minor fixes.
- Refactored code.
- Updated dependencies.
- Updated README.

## [0.1.0]

### Changed

- Updated the project structure.
- Updated the code style.
- Updated the agent blueprints.

### Added

- Added support for the `Langfuse` observability platform.
- Added an agent executor.
- Added a prompt manager.
- Added a custom implementation of React Agent.
- Added these service runners: standard, aws lambda, and azure functions.

### Fixed

- Made minor fixes.

### Removed

- Removed support for twelve LLM providers. Replaced them with
  `openai-compatible`. Use `LiteLLM` as a proxy for any LLM provider.
