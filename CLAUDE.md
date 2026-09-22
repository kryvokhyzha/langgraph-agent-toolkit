# CLAUDE.md

Use this guidance when you work in this repository. Keep responses concise.

## Project overview

`langgraph-agent-toolkit` is a production framework for AI agents. It builds,
deploys, and runs agents with **LangGraph** (agent graphs), **FastAPI**
(streaming HTTP service), and **Streamlit** (chat UI). It is a published PyPI
library, not a monorepo. It has one installable package
(`langgraph_agent_toolkit/`) and Docker/CI infrastructure.

Key components include a **LiteLLM** proxy for multi-provider LLMs, PostgreSQL /
SQLite checkpointers for memory, and **Langfuse / LangSmith** for observability
and prompt management. Almost all subsystems use a **factory +
Pydantic-settings** pattern. See
[Architecture & core patterns](#architecture--core-patterns).

Use these entry points:

- **`langgraph_agent_toolkit/run_api.py`** — FastAPI service for uvicorn /
  gunicorn / AWS Lambda / Azure Functions. Its default bind is `0.0.0.0:8080`.
  It uses the `fire` CLI.
- **`langgraph_agent_toolkit/run_app.py`** — Streamlit chat UI. Its default bind
  is `:8501`. It uses the `AgentClient` SDK to call the service.
- **`langgraph.json`** — LangGraph CLI / Studio deployment configuration. It
  exposes the `chatbot` graph.

## Tech stack and key conventions

- **Python**: `>=3.11,<3.15`. The Ruff target is **`py311`**. Docker images and
  the LangGraph deployment configuration use **3.13**. Write code that runs on
  3.11–3.14.
- **Package manager**: Use `uv`. Do not use `pip`/`poetry`/`conda` directly. The
  lockfile is `uv.lock`. The project installs **deps-only** with
  `--no-install-project`. Imports use `pythonpath`, not an editable install.
- **Build backend**: `hatchling`. The package is top-level
  (`langgraph_agent_toolkit/`). Do not put it under `src/`.
- **Configuration**: **Pydantic Settings** (`core/settings.py`,
  `core/_base_settings.py`) loads configuration from env / `.env`. Do not add a
  Hydra/argparse configuration system. See
  [Configuration & settings](#configuration--settings).
- **Logging**: Use Loguru through `helper/logging.py`. `LoggerConfig` is a
  singleton through `SingletonMeta`. Use `ENV_MODE`, `LOG_LEVEL`, `JSON_LOGS`,
  and `COLORIZE` environment variables. Custom levels are `WARNONCE` and
  `DEPRECATED`. Use `warn_once()` / `log_deprecated()` for deduplicated logs.
  `InterceptHandler` sends stdlib/FastAPI logs to loguru.
- **CLI**: `fire` (used by `run_api.py`).
- **Path/root resolution**: Use `rootutils` and the `.project-root` marker.
  Example scripts use them. Prefer them to paths calculated from `__file__`.
- **Data models**: Use `pydantic` for settings, request/response schema, and
  exceptions.

## Repository layout

```
langgraph_agent_toolkit/      # The package (all importable code)
  run_api.py                  # FastAPI service entry (fire CLI)
  run_app.py                  # Streamlit UI entry
  agents/
    agent.py                  # Agent dataclass (wraps a compiled LangGraph Pregel)
    agent_executor.py         # Loads agents from "module:obj" strings; invoke/stream
    components/               # checkpoint/, creators/ (custom create_react_agent), middlewares/, tools, utils
    blueprints/               # Example agent graphs: react, create_agent*, chatbot, supervisor, etc.
  service/                    # FastAPI HTTP layer (factory, routes, handler, middleware, auth)
  client/client.py            # AgentClient SDK (httpx; sync + async)
  core/                       # Cross-cutting factories + settings
    settings.py / _base_settings.py   # Pydantic Settings singleton
    models/ memory/ observability/ prompts/   # Factory-backed subsystems
  schema/                     # Shared Pydantic models (ChatMessage, UserInput, ...)
  helper/                     # logging, exceptions, constants, types, utils
configs/                      # Per-service env + config (litellm/, postgres/, redis/, ...)
docker/                       # api/Dockerfile (backend), app/Dockerfile (frontend)
docker-compose.yaml           # Full local stack (backend, frontend, litellm, langfuse, ...)
scripts/
  postgres-init/              # DB bootstrap SQL (create_databases.sql, create_schema.sql)
  python/                     # 01–09 runnable usage examples (proxy, prompts, history)
tests/                        # Focused tests, component integration, and process journeys
docs/                         # Sphinx docs (published to GitHub Pages)
.github/workflows/            # CI/CD: test.yml, release.yml, sphinx.yml
pyproject.toml                # Project + tool config (ruff, pytest, extras)
Makefile                      # uv / pre-commit / docker shortcuts
langgraph.json                # LangGraph CLI deploy config
.env.example                  # Committed template for root env vars; copy to .env
.project-root                 # rootutils marker — do not delete
```

Git ignores these temporary or generated files: `data/`, `dist/`,
`.langgraph_api/`, `.venv/`, and caches.

## Common commands

Prefer `make` targets when they are available. Always run Python with `uv run …`
to use the locked environment.

| Task                          | Command                                                                   |
| ----------------------------- | ------------------------------------------------------------------------- |
| Install deps (all features)   | `make uv_install_deps`                                                    |
| Update deps (frozen)          | `make uv_update_deps`                                                     |
| Refresh lockfile              | `make uv_get_lock`                                                        |
| Show installed                | `make uv_show_deps` / `make uv_show_deps_tree`                            |
| Build wheel                   | `make uv_build_wheel`                                                     |
| Install pre-commit hooks      | `make pre_commit_install`                                                 |
| Run pre-commit on all files   | `make pre_commit_run`                                                     |
| Run tests                     | `uv run pytest`                                                           |
| Run API process journeys      | `uv run pytest tests/e2e tests/service/test_worker_recovery.py --run-e2e` |
| Run Docker checks             | `uv run pytest tests/integration/test_docker_e2e.py --run-docker`         |
| Lint / format (manual)        | `uv run ruff check --fix .` / `uv run ruff format .`                      |
| Run the API (uvicorn default) | `uv run python langgraph_agent_toolkit/run_api.py`                        |
| Run the Streamlit UI          | `uv run streamlit run langgraph_agent_toolkit/run_app.py`                 |
| Run an example script         | `uv run python scripts/python/<name>.py`                                  |
| Full local stack (Docker)     | `docker compose up` (or `docker compose watch`)                           |
| Rebuild one service           | `make rebuild_api` / `make rebuild_app`                                   |
| Tag release from version      | `make push_new_tag`                                                       |

## Code style

Ruff enforces these rules in `pyproject.toml`:

- Line length **120**, target `py311`.
- Rules: `E, F, W, I, D` (pycodestyle, pyflakes, isort, pydocstyle).
- isort: 2 blank lines after imports.
- Docstrings: Missing-docstring rules are disabled (`D100`–`D107` ignored). Use
  **Google-style** docstrings. Sphinx napoleon parses them.
- Per-file ignores: `scripts/python/*` and `run_agent.py` allow `E402`;
  `service/factory.py` allows `F821`.

## Technical writing

Write comments, docstrings, and documentation in ASD-STE100 Simplified Technical
English.

- Use short, direct sentences in active voice. State one instruction or fact in
  each sentence. Use simple, consistent words. Do not use idioms, ambiguous
  terms, or unnecessary synonyms.
- Preserve each technical identifier exactly. This includes code symbols, API
  names, file paths, commands, options, environment variables, configuration
  keys, URLs, literal values, casing, and punctuation.
- You may use `gpt-5.6-sol`, `gpt-5.6-terra`, or `gpt-5.6-luna` to rewrite
  existing comments and documentation in this style. Change prose only. Do not
  change technical meaning or program behavior.

Also follow these conventions:

- Prefer `pathlib.Path` over `os.path`.
- Use the configured loguru `logger` (`helper/logging.py`). Do not call
  `logging.getLogger`.
- Use `pydantic` for data models and configuration validation.
- Add type hints to new code. Tests can use fewer type hints.

## Configuration & settings

Use one global settings singleton:
`from langgraph_agent_toolkit.core.settings import settings`. It is a
`pydantic_settings.BaseSettings` in `_base_settings.py`. It uses
`extra="ignore"` and loads the nearest `.env`.

- Add new configuration values to the `Settings` class in
  `core/_base_settings.py`. Give each value a sensible default. Also add the
  matching entry to `.env.example`.
- `settings.setup()` runs at import. It validates `LANGGRAPH_<NAME>` environment
  overrides against the declared Pydantic field types. It loads `MODEL_CONFIGS`
  / `DB_CONFIGS` from inline JSON, base64, or a file path.
- Use `MODEL_CONFIGS` for multi-model setups. It contains named provider/param
  dictionaries that `model_config_key` selects.
- Langfuse uses validated `Settings` values and direct environment variables. It
  passes credentials and client options to the SDK explicitly.
  `LANGFUSE_BASE_URL` takes precedence over `LANGFUSE_HOST`. LangSmith reads
  `LANGSMITH_*` directly from `os.environ`. Entry points load `.env` for these
  direct environment reads.
- Notable defaults: `ENV_MODE=production`, `HOST/PORT=0.0.0.0/8080`,
  `MEMORY_BACKEND` unset (→ no persistence), `OBSERVABILITY_BACKEND` unset (→
  `EMPTY` at runtime), `DEFAULT_AGENT=create-agent`, `CHECK_INTERRUPTS=True`,
  `CORS_ENABLED=False`, `AUTH_MODE=trusted`. Production requires authentication.
  One `AUTH_SECRET` preserves shared-token access with optional `user_id`. Set
  `AUTH_MODE=token` to require token-bound user identities.
- `MCP_SERVERS` configures optional MCP tools. `MCP_AGENT_SERVERS` assigns
  servers to agents. An empty assignment selects the default agent. MCP imports
  stay lazy, and tool discovery runs during service startup. See `docs/mcp.rst`.

## Architecture & core patterns

**Factory pattern (used by every `core/` subsystem).** Use this pattern when you
add a backend:

1. `types.py` — a `StrEnum` of backends (lowercase `auto()` values).
2. `base.py` — an abstract `Base*` interface.
3. one module per concrete backend.
4. `factory.py` — `*Factory.create(...)` converts input to an enum and uses
   `match`. The default case raises `ValueError`.

This pattern applies to **models** (`CompletionModelFactory`/
`EmbeddingModelFactory`; OpenAI uses `ChatOpenAIPatched`; other models use
LangChain `init_chat_model`; `FAKE` uses `FakeToolModel`). It also applies to
**memory** (`POSTGRES`/`SQLITE`; SQLite has no store), **observability**
(`LANGFUSE`/`LANGSMITH`/`EMPTY`; lazy imports keep optional dependencies
optional), and **prompts** (`PromptManager` + `ObservabilityChatPromptTemplate`;
files are the source of truth and Jinja2 renders them).

**Register agents with import strings, not filesystem scans.** `AgentExecutor`
reads `"module.path:object"` strings from `settings.AGENT_PATHS` and imports
them dynamically. An imported object can be a compiled graph or an `Agent`
dataclass. Required import failures stop startup. To add an agent, create
`agents/blueprints/<name>/agent.py` that exposes a compiled graph or `Agent`.
Then add its path to `AGENT_PATHS`. The default list is in `_base_settings.py`.
`run_api.py` uses this configured list. Set `AGENT_PATHS` when a deployment must
load different agents.

**ReAct example blueprints — two builders.** The toolkit provides two ways to
build a tool-calling ReAct agent:

- **`react`** — Uses the toolkit's **custom `create_react_agent`**
  (`components/creators/`). It forks LangGraph's prebuilt agent. It adds an
  `immediate_generation` router, `sanitize_chat_history`, and an always-on
  `pre_model_hook`. Its default prompt is local. Import does not write remote
  prompts or require a Langfuse service.
- **`create_agent`** (flagship) and **`create_agent_structured`** — Use
  **LangChain's native `create_agent`** with middleware. The
  `components/middlewares/` reproduce custom-agent features on the supported
  path. All use view-only `wrap_model_call`, so the full history stays in state.
  They are `ImmediateGenerationMiddleware`, `SanitizeHistoryMiddleware`,
  `ClearIntermediateToolCallsMiddleware` (token reduction; tune it with
  `CLEAR_INTERMEDIATE_TOOL_CALLS*` settings), and `TrimMessagesMiddleware`
  (view-only window limit; reuses `DEFAULT_MAX_MESSAGE_HISTORY_LENGTH`). The
  flagship stack also uses the built-in `ContextEditingMiddleware` (window) and
  `ToolRetryMiddleware` (resilience). `create_agent_structured` adds a
  `response_format`. Use this pattern for new agents. `create-agent` is the
  default (`DEFAULT_AGENT=create-agent`).
- **`hitl_agent`** — Provides human-in-the-loop tool approval on `create_agent`
  through `HumanInTheLoopMiddleware`. It resumes through
  `agent_executor.build_resume_command`. This maps `approve` /
  `reject: <reason>` / free-text replies to the middleware decision format. Raw
  `interrupt()` blueprints (`interrupt_agent`) remain unchanged. They receive
  `Command(resume=<input>)`.

**Service.** `service/handler.py:create_app()` is the FastAPI factory. Its async
`lifespan` adds observability, the memory checkpointer, and `AgentExecutor` to
`app.state`. It sets readiness only after agents initialize. There is a public
router (health, `/`) and a `verify_bearer`-gated private router. Most agent
endpoints have `/{agent_id}/…` and a default-agent alias.

**MCP tools.** Install the `mcp` extra. Tool-enabled blueprints expose
`build_graph(extra_tools=())` through `Agent.graph_factory`. Use `merge_tools`
from `core/mcp.py` to reject duplicate tool names. Keep network access out of
module imports. The service builds selected graphs before it assigns the
checkpointer. MCP tools are asynchronous. Do not apply automatic tool retries to
remote writes. MCP elicitation uses `custom_data.interrupts` and structured
resume answers. Keep `CHECK_INTERRUPTS=True` for this feature.

**Streaming = SSE.** Use this wire contract:
`data: {"type": "token"|"message"|"error", "content": …}` lines terminated by
`data: [DONE]`. Produced in `service/utils.py: message_generator`, parsed in
`client/client.py:_parse_stream_line`. Keep both ends in sync if you change it.

## Testing

- Framework: `pytest` + `pytest-asyncio` + `pytest-cov` + `pytest-env` in
  `[dependency-groups] tests`.
- `pyproject.toml` sets `pythonpath = ["langgraph_agent_toolkit", "."]`.
- **`asyncio_mode=auto`** runs asynchronous tests. New tests can also use
  `@pytest.mark.asyncio` to make their execution mode explicit.
- Focused tests check validation and error boundaries. Component tests run real
  graphs, SQLite, MCP protocols, and Langfuse SDK serialization. Use mocks for
  external dependencies, not to replace the contract under test. The fake LLM is
  `ModelProvider.FAKE` → `FakeToolModel`. Its canonical response is
  `"This is a test response from the fake model."`.
- `uv run pytest` needs no external service. It includes in-process HTTP and
  local MCP stdio tests. PostgreSQL tests also run when `LAT_TEST_POSTGRES_DSN`
  is set. `--run-postgres` requires that DSN and fails when it is absent.
- **Process journeys** (`tests/e2e/`) require `--run-e2e`. They call the real
  API through TCP and test durable restart and resume. Worker replacement tests
  use the same flag. CI runs them on each supported Python version.
- **Real OpenAI tests** also require `--run-llm`, `LAT_TEST_OPENAI_API_KEY`, and
  `LAT_TEST_OPENAI_MODEL`. They use synthetic prompts and at most twelve model
  requests. Normal tests skip them. See `docs/live_llm_testing.rst`.
- **Docker checks** require `--run-docker`. They check API behavior, frontend
  health, and HTML. They do not test browser interaction.
- **Langfuse SDK tests** run SDK 2.60.10, 3.15.0, and 4.15.2 on Python
  3.11–3.14. Select the matching `langfuse-v2`, `langfuse-v3`, or `langfuse-v4`
  extra and pin the exact SDK baseline in each test environment. Live server
  tests require `--run-langfuse` and a dedicated project configured through
  `LAT_TEST_LANGFUSE_*`. Missing explicit configuration causes failure.
- **Coverage** includes the whole package, including blueprints, UI, and legacy
  creators. `.coveragerc` enables branches and subprocess collection. The 77%
  floor measures combined line and branch coverage. It is not a pure branch
  percentage. Do not omit difficult package files to increase this result.
- Add tests for new features in the matching `tests/<area>/` directory.
- See `docs/testing.rst` for commands, layer boundaries, and live-test setup.

## Adding dependencies

- **Runtime**: Add dependencies to `[project].dependencies` in `pyproject.toml`.
  Then run `make uv_install_deps` (`uv sync --extra all --no-install-project`).
- **Optional features**: Use `[project.optional-dependencies]` extras. LLM
  providers are `openai`, `anthropic`, `google-vertexai`, `google-genai`, `aws`,
  and `all-llms`. Service backends are `uvicorn-backend`, `gunicorn-backend`,
  `aws-backend`, and `azure-backend`. Observability extras are `langfuse`,
  `langfuse-v2`, `langfuse-v3`, `langfuse-v4`, `langsmith`, and
  `all-observability`. The selectors require SDK `>=2.60.10,<3`, `>=3.15.0,<4`,
  and `>=4.15.2,<5`, respectively. Use only one selector. Generic `langfuse`
  permits `>=2.60.10,<5` and can combine with any one selector. These extras
  select the SDK, not the server version. Their canonical names use hyphens;
  package tools normalize underscore spellings such as `langfuse_v2`.
- **All features**: Replace `--all-extras` with `--extra all`. The old flag
  selects all three incompatible SDK selectors. The `all` extra includes every
  feature through the existing umbrella extras and does not force a selector.
  The default lockfile selection uses SDK v4. Use
  `--extra all --extra langfuse-v2` for the same features with SDK v2 when all
  requirements are compatible. `all-observability` still includes generic
  `langfuse` and `langsmith`.
- **Dev / lint / test / docs**: Use the matching `[dependency-groups]` group.
  Groups are local source-checkout tools, not published wheel extras. The
  `client` group is the minimal Streamlit-only set.
- Update `uv.lock` after every change. The pre-commit `uv-lock` hook enforces
  this. Prefer `~=` for closely tracked libraries. Prefer `>=,<` for broad
  ranges.

## Environment variables (`.env` / `.env.example`)

- `.env.example` is the committed source of truth for root (backend/app)
  environment variables. `.env` is gitignored. Never commit secrets.
- When code reads a new environment variable, add it to `.env.example` with a
  placeholder.
- Per-service container configuration is in `configs/<svc>/.<svc>.env`. Each has
  a committed `.example`. `configs/litellm/config.yaml` is gitignored and stores
  a real local key. Do not commit it.
- `scripts/python/` examples use `.local.env` with `load_dotenv`. It is separate
  from the service `.env`.

## Docker / local stack

The API image installs `openai-aiohttp` and defaults to aiohttp for managed
asynchronous OpenAI and Azure calls. Python installations default to HTTPX. Use
`LLM_HTTP_ASYNC_TRANSPORT=httpx` to override the image default. The example
`.env` leaves this selection unset so it does not override the environment.

`docker-compose.yaml` starts the full stack. It contains `lat-agent-backend`
(:8080), `lat-agent-frontend` (:8501), `litellm` (:4000), and a self-hosted
Langfuse stack. The Langfuse stack contains postgres, redis, clickhouse, minio,
langfuse-web :3000, and langfuse-worker. Postgres creates three DBs (`agents`,
`litellm`, `langfuse`) and a `checkpoints` schema through
`scripts/postgres-init/`. Both application images use multi-stage
`python:3.13-slim` builds. The builder uses `uv`; the runtime contains its
virtual environment and application source. Both runtime images install `curl`
for health probes. Use `make rebuild_api` / `make rebuild_app` to rebuild one
service.

## Pre-commit hooks (what runs on commit)

The hooks run Gitleaks with redacted output and `pre-commit-hooks` basics,
including `detect-private-key`, `detect-aws-credentials`, and
`check-added-large-files --maxkb=1024`. They run `ruff` (fix + format),
`codespell`, `prettier` (md/yaml/toml/json/dockerfile/shell,
`--print-width=80 --prose-wrap=always`), and `uv-lock`. They use `python3.13`.
Do not bypass them with `--no-verify` unless explicitly asked.

## Finishing a task (verify before reporting done)

Always run an appropriate verification before you report a task as complete. Do
not rely on a diff that looks correct. Use the lightest command that tests the
change:

- **Code touching the package** → Run `uv run pytest`. If the suite is slow, run
  `uv run pytest tests/<area>/test_x.py::test_y`.
- **Style / imports** → Run `uv run ruff check --fix .` and
  `uv run ruff format .`.
- **Just before a commit** → Run `make pre_commit_run`.
- **Service / endpoint changes** → Start the service with
  `uv run python langgraph_agent_toolkit/run_api.py` and `USE_FAKE_MODEL=true`.
  Call `/health/ready`, or run the in-process e2e
  (`tests/service/test_stream_integration.py`).
- **New example script** → Run it with a minimal configuration to verify that it
  starts.

If you cannot run verification because it needs a DB, real LLM keys, or an
external service, say so explicitly. Do not imply success. Do not suppress
errors to make a command pass. Fix the root cause.
