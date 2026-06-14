# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.2]

### Fixed

- `TrimMessagesMiddleware` / `TokenTrimMiddleware` no longer silently destroy
  the model's view when the **latest turn does not fit the budget**.
  `trim_messages` (`strategy="last"`, `start_on="human"`) returns `[]` whenever
  the current turn exceeds the budget — e.g. a single user message larger than
  the token budget, or a turn with more tool calls than the message budget —
  which invoked the model with the system prompt only and made it answer with
  the user's question dropped. Both middlewares now floor to the latest turn
  (kept verbatim, over budget but answerable) rather than emptying it.
  Regression tests cover an oversize user message, an oversize tool result, and
  a 6+ tool-call burst.

### Added

- `TokenTrimMiddleware` — bounds the model's message view by a **token** budget
  (the token-counting companion to `TrimMessagesMiddleware`, which bounds by
  message count). View-only, so the full history stays in state and the system
  prompt is preserved. The token counter is configurable (`token_counter`,
  default `count_tokens_approximately` — no extra dependency; pass a
  `tiktoken`-backed counter or the model for an exact count), and the budget
  defaults to the new `DEFAULT_MAX_TOKENS_HISTORY_LENGTH` setting (unset by
  default, since a sensible budget is model-specific). Compose it after
  `TrimMessagesMiddleware` to cap both message count and token size.

## [0.9.1]

### Added

- Reusable `create_agent` middleware that brings the custom `create_react_agent`
  history features to native LangChain `create_agent`, using only public API:
  `ImmediateGenerationMiddleware` (graceful model-call budget — on the last
  allowed call it strips tools and asks the model to synthesize a direct answer
  instead of stalling), `SanitizeHistoryMiddleware` (repair broken
  tool-call/result pairing before the model call), and
  `ClearIntermediateToolCallsMiddleware` (token reduction — keep only the most
  recent result(s) per tool from earlier turns; configurable via constructor or
  `CLEAR_INTERMEDIATE_TOOL_CALLS*` settings, with a safe `name_args` dedup key
  that collapses only identical repeat calls by default), and
  `TrimMessagesMiddleware` (view-only window bound — trims the model's input to
  the last `DEFAULT_MAX_MESSAGE_HISTORY_LENGTH` messages while keeping full
  state). Demonstrated in the `create_agent` blueprint
- `hitl_agent` blueprint: human-in-the-loop tool approval on native
  `create_agent` via `HumanInTheLoopMiddleware`. The executor bridges the resume
  (`build_resume_command`): replying `approve` / `reject: <reason>` / free text
  maps to the middleware's decision format, with custom `interrupt()` blueprints
  unchanged
- Multimodal input: `UserComplexInput.message` accepts LangChain content blocks
  (text / image / file / audio / video, via URL or base64), with a configurable
  per-message attachment cap (`MULTIMODAL_MAX_ATTACHMENTS`), and the Streamlit
  chat supports file/image uploads

### Changed

- Dockerfiles (`docker/api`, `docker/app`) are now multi-stage with uv cache
  mounts and dependency-before-source layering: faster rebuilds (a code change
  no longer reinstalls dependencies), smaller images, and a trimmed build
  context
- Renamed and de-duplicated the ReAct example blueprints: `react_new` →
  `create_agent`, `react_so` → `create_agent_structured`; removed `react_old` (a
  duplicate of `react`) and `react_so_old`. Examples are now grouped by builder
  (toolkit `create_react_agent` vs native `create_agent`)
- `sanitize_chat_history` is now bidirectional: in addition to stripping
  unanswered tool calls, it drops orphaned ToolMessages (a tool result whose
  requesting tool call is gone, e.g. after trimming/summarization). Both the
  custom `create_react_agent` and `SanitizeHistoryMiddleware` benefit
- Refactored the Streamlit UI from a single `run_app.py` into a `ui/` package
  (`main_page`, `components/`, `utils/`); `run_app.py` is now a thin entry point

### Fixed

- Human-in-the-loop interrupts now resume by default (`CHECK_INTERRUPTS`
  defaults to `True`); previously the resume was silently skipped
- Streaming now surfaces an agent's `structured_response` (`response_format`),
  matching `invoke`

## [0.9.0]

### Added

- JSON Lines (NDJSON) streaming endpoint `/stream/jsonl` (and
  `/{agent_id}/stream/jsonl`) as a typed alternative to SSE, with a
  `StreamChunk` response schema
- `AgentClient.stream_jsonl` / `astream_jsonl` SDK methods
- Streamlit option to choose the streaming protocol (SSE or JSON Lines)
- Idiomatic OpenAPI operation IDs, documented error responses
  (401/404/422/429/500/503) with an `ErrorResponse` schema, and grouped
  `openapi_tags`

### Changed

- Migrated the Langfuse integration to the v4 SDK (still v3-compatible)
- Upgraded to LangChain 1.x / LangGraph 1.x APIs (`create_agent`,
  `context_schema`, the new `SummarizationMiddleware` and supervisor APIs)
- Error responses no longer expose internal exception details in production
  (gated by `ENV_MODE`)
- Supervisor blueprint uses `output_mode="full_history"` so sub-agent messages
  survive history reloads
- `docker-compose`: the backend now waits for Langfuse to be healthy before
  starting

### Fixed

- Trace-level output is now recorded in Langfuse (regression introduced by the
  v4 callback change)
- Kubernetes degraded boot: the startup probe now passes (no CrashLoop) and the
  DB pool / readiness flags are cleared on shutdown
- `/health/db` no longer errors when using the SQLite backend
- `401` responses include a `WWW-Authenticate: Bearer` header
- `PostgresMemoryBackend.get_store` `app_prefix` TypeError
- Various LangGraph / Starlette deprecation warnings

## [0.8.15]

### Fixed

- `add_messages` and `clear_history` endpoints
- `[-1]` handling

## [0.8.14]

### Added

- Cors settings

## [0.8.13]

### Added

- New default settings

### Updated

- Callback creation
- Project structure

## [0.8.12]

### Added

- New API healthchecks

### Update

- Main logger

## [0.8.11]

### Fixed

- Handling AIMessage that has tool calls without ToolMessage

## [0.8.10]

### Updated

- Default Postgres settings

## [0.8.9]

### Updated

- Refactoring of observability class
- Refactoring of prompt manager class
- Fix uvicorn setup

## [0.8.8]

### Fixed

- Fixed bug to support `langfuse < 2.70.0`

## [0.8.7]

### Fixed

- Fixed bug to support `langfuse < 2.70.0`

## [0.8.6]

### Fixed

- Fixed bug to support `langfuse < 2.70.0`

## [0.8.5]

### Updated

- Core Dependencies to support `langchain < 1.0.0`

## [0.8.4]

### Added

- New postgres settings
- DB healthcheck API

### Fixed

- Problem with support of old langfuse sdk

## [0.8.3]

### Updated

- Structure of configuration setting

## [0.8.2]

### Added

- Factory for embedding models

## [0.8.1]

### Fixed

- langfuse `score` -> `create_score`

### Updated

- Core Dependencies

## [0.8.0]

### Added

- `create_agent` example

### Updated

- Core Dependencies

### Fixed

- Fix Langfuse
- Stream mode inside `invoke` method

## [0.7.23]

### Updated

- Default logging configuration

## [0.7.22]

### Added

- NoOpSaver for checkpointing

### Updated

- Logging configuration

## [0.7.21]

### Updated

- Remove caching decorator from create method

## [0.7.20]

### Updated

- Enhance logging configuration
- Improve message parsing

## [0.7.19]

### Updated

- Enhance model parameter values and improve factory model creation logic

## [0.7.18]

### Added

- `SKIP_REDIRECTION_LOGGING` environment variable and enhance logging middleware

## [0.7.17]

### Fixed

- Error handling
- Tests

## [0.7.16]

### Fixed

- Type of `content` field for `ChatMessage` model

### Added

- New tests

## [0.7.15]

### Added

- `remote first` logic for observability platform

### Updated

- minor updates on UI
- you can read some default values from env vars

## [0.7.14]

### Updated

- Make `message` field optional
- Default value for `MEMORY_BACKEND`
- Refactor `lifespan` function

## [0.7.13]

### Added

- `DB_CONFIGS` initialization

## [0.7.12]

### Updated

- Dependencies

## [0.7.11]

### Updated

- Dependencies

## [0.7.10]

### Fixed

- error handling in `message_generator`
- max_messages type

## [0.7.9]

### Fixed

- type of graph (create_react_agent)

## [0.7.8]

### Added

- Prompt manager
- Utils functions

### Updated

- Dependencies

## [0.7.7]

### Fixed

- Error handling

## [0.7.6]

### Fixed

- Added schema for postgres db

## [0.7.5]

### Fixed

- Downgrade langfuse

## [0.7.4]

### Updated

- Langfuse Callback import fix

## [0.7.3]

### Added

- New env variable for model config (base64)

## [0.7.2]

### Added

- New env variable for model config (file)

## [0.7.1]

### Fixed

- Agent executor test mock assertion to include additional `environment` and
  `tags` parameters
- Async test methods missing `@pytest.mark.asyncio` decorator in prompts tests

### Updated

- Dependencies to latest versions
- Agent executor `get_callback_handler` method to pass additional parameters:
  - Added `environment` parameter from settings.ENV_MODE
  - Added `tags` parameter with agent name for better observability tracking

### Improved

- Langfuse observability prompt hash detection with fallback mechanism:
  - Enhanced `push_prompt` method to use tags as fallback when commit_message is
    empty
  - Added robust `hasattr` checks for both `commit_message` and `tags`
    attributes
  - Improved logging to show old vs new hash values for better debugging

## [0.7.0]

### Fixed

- React Agent with SO

### Updated

- Dependencies
- Name of default configurable parameters

## [0.6.0]

### Fixed

- React Agent with SO

### Added

- Complex input

### Updated

- Dependencies
- Error handling
- Tests

## [0.5.0]

### Fixed

- Streamlit UI bugs
- Windows compatibility issue
- Enhance message handling inside `pre_hook_model`
- Add prompt hash to Langfuse observability class
- Rename few parameters

## [0.4.5]

### Fixed

- Strucuted output and model factory

## [0.4.4]

### Updated

- Project dependencies

### Fixed

- Strucuted output and model factory

## [0.4.3]

### Updated

- Project dependencies

## [0.4.2]

### Fixed

- Streaming bug
- Steamlit welcom message display
- Client handling error
- Package dependencies

## [0.4.1]

### Updated

- Client API to fully align with server endpoints
- Extended invoke, stream methods with additional parameters

### Added

- Message management methods in the client (add_messages, aadd_messages)
- Chat history retrieval methods (get_history, aget_history)
- History clearing methods (clear_history, aclear_history)
- Synchronous feedback creation method (create_feedback)
- Support for model_config_key parameter
- Support for recursion_limit parameter

### Fixed

- Client tests to properly mock API endpoints
- Parameter handling in stream and invoke methods

## [0.4.0]

### Updated

- Endpoints

### Added

- Endpoint to clear history
- Add new message to the history

### Fixed

- Minor fixes and refactoring

## [0.3.1]

### Added

- Ability to pass parameters to service runner
- Argument to select service runner

### Updated

- Service Dockerfile

## [0.3.0]

### Added

- `MODEL_CONFIGS` to unify LLM env variables
- New blueprint with AWS KB

### Fixed

- Streaming messages handling
- Refactored code structure for better maintainability
- Optional dependencies
- API exception handling

## [0.2.0]

### Fixed

- Refactored code structure for better maintainability
- Refactored Model factory

### Removed

- Removed AllModels and added environment variables for different providers

## [0.1.2]

### Added

- `user_id` parameter
- `store` creator to memory classes

### Fixed

- enhance error handling and testing in Streamlit app
- add new chat button
- variable names
- type hints

### Removed

- print statements

## [0.1.1]

### Added

- `get_default_agent` and `set_default_agent` functions

### Fixed

- Minor fixes
- Refactoring
- Update dependencies
- Update README

## [0.1.0]

### Changed

- Project structure
- Code style
- Agent blueprints

### Added

- Support of `Langfuse` observability platform.
- Agent executor
- Prompt manager
- Custom implementation of React Agent
- Service runners: standard, aws lambda, azure functions

### Fixed

- Minor fixes

### Removed

- Support of dozen LLM providers. They were replaced by a single one -
  `openai-compatible`. We can use `LiteLLM` as proxy for any LLM provider.
