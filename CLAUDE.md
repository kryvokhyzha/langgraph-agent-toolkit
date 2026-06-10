# CLAUDE.md

Guidance for Claude Code when working in this repository. Keep responses concise
and follow the conventions below.

## Project overview

`langgraph-agent-toolkit` is a production framework for building, deploying, and
running AI agents built on **LangGraph** (agent graphs), **FastAPI** (streaming
HTTP service), and **Streamlit** (chat UI). It is a published PyPI library, not
a monorepo: one installable package (`langgraph_agent_toolkit/`) plus Docker/CI
infrastructure around it.

Key building blocks: a **LiteLLM** proxy for multi-provider LLMs, PostgreSQL /
SQLite checkpointers for memory, and **Langfuse / LangSmith** for observability
and prompt management. Almost every subsystem is wired through a **factory +
Pydantic-settings** pattern (see
[Architecture & core patterns](#architecture--core-patterns)).

Entry points:

- **`langgraph_agent_toolkit/run_api.py`** — FastAPI service (uvicorn / gunicorn
  / AWS Lambda / Azure Functions), default bind `0.0.0.0:8080`. `fire` CLI.
- **`langgraph_agent_toolkit/run_app.py`** — Streamlit chat UI (default
  `:8501`), drives the service via the `AgentClient` SDK.
- **`langgraph.json`** — LangGraph CLI / Studio deploy config (exposes the
  `chatbot` graph).

## Tech stack and key conventions

- **Python**: `>=3.11,<3.14`; Ruff target is **`py311`**. Docker images and the
  LangGraph deploy config use **3.13**, so write code that runs on 3.11–3.13.
- **Package manager**: `uv` (never `pip`/`poetry`/`conda` directly here).
  Lockfile is `uv.lock`. Note the project installs **deps-only**
  (`--no-install-project`); imports resolve via `pythonpath`, not an editable
  install.
- **Build backend**: `hatchling`; the package is **top-level**
  (`langgraph_agent_toolkit/`), **not** under `src/`.
- **Configuration**: **Pydantic Settings** (`core/settings.py`,
  `core/_base_settings.py`) loaded from env / `.env`. **There is no
  Hydra/argparse config system** — do not add one. See
  [Configuration & settings](#configuration--settings).
- **Logging**: Loguru via `helper/logging.py` (singleton `LoggerConfig` through
  `SingletonMeta`). Env vars: `ENV_MODE`, `LOG_LEVEL`, `JSON_LOGS`, `COLORIZE`.
  Custom levels: `WARNONCE`, `DEPRECATED` (via `warn_once()` /
  `log_deprecated()`, deduped). An `InterceptHandler` routes stdlib/FastAPI
  logging into loguru.
- **CLI**: `fire` (used by `run_api.py`).
- **Path/root resolution**: `rootutils` + the `.project-root` marker (used by
  the example scripts; prefer it over computing paths from `__file__`).
- **Data models**: `pydantic` everywhere (settings, request/response schema,
  exceptions).

## Repository layout

```
langgraph_agent_toolkit/      # The package (all importable code)
  run_api.py                  # FastAPI service entry (fire CLI)
  run_app.py                  # Streamlit UI entry
  agents/
    agent.py                  # Agent dataclass (wraps a compiled LangGraph Pregel)
    agent_executor.py         # Loads agents from "module:obj" strings; invoke/stream
    components/               # checkpoint/, creators/ (custom create_react_agent), tools, utils
    blueprints/               # Example agent graphs: chatbot, react*, supervisor, etc.
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
tests/                        # pytest suite (unit / in-process e2e / docker e2e)
docs/                         # Sphinx docs (published to GitHub Pages)
.github/workflows/            # CI/CD: test.yml, release.yml, sphinx.yml
pyproject.toml                # Project + tool config (ruff, pytest, extras)
Makefile                      # uv / pre-commit / docker shortcuts
langgraph.json                # LangGraph CLI deploy config
.env.example                  # Committed template for root env vars; copy to .env
.project-root                 # rootutils marker — do not delete
```

Temp / generated, safe to ignore (gitignored): `data/`, `dist/`,
`.langgraph_api/`, `.venv/`, caches.

## Common commands

Prefer `make` targets when available; always run Python via `uv run …` so the
locked environment is used.

| Task                          | Command                                                   |
| ----------------------------- | --------------------------------------------------------- |
| Install deps (all extras)     | `make uv_install_deps`                                    |
| Update deps (frozen)          | `make uv_update_deps`                                     |
| Refresh lockfile              | `make uv_get_lock`                                        |
| Show installed                | `make uv_show_deps` / `make uv_show_deps_tree`            |
| Build wheel                   | `make uv_build_wheel`                                     |
| Install pre-commit hooks      | `make pre_commit_install`                                 |
| Run pre-commit on all files   | `make pre_commit_run`                                     |
| Run tests                     | `uv run pytest`                                           |
| Run docker e2e tests          | `uv run pytest tests/integration --run-docker`            |
| Lint / format (manual)        | `uv run ruff check --fix .` / `uv run ruff format .`      |
| Run the API (uvicorn default) | `uv run python langgraph_agent_toolkit/run_api.py`        |
| Run the Streamlit UI          | `uv run streamlit run langgraph_agent_toolkit/run_app.py` |
| Run an example script         | `uv run python scripts/python/<name>.py`                  |
| Full local stack (Docker)     | `docker compose up` (or `docker compose watch`)           |
| Rebuild one service           | `make rebuild_api` / `make rebuild_app`                   |
| Tag release from version      | `make push_new_tag`                                       |

## Code style

Enforced by Ruff (`pyproject.toml`):

- Line length **120**, target `py311`.
- Rules: `E, F, W, I, D` (pycodestyle, pyflakes, isort, pydocstyle).
- isort: 2 blank lines after imports.
- Docstrings: missing-docstring rules are **disabled** (D100–D107 ignored). Use
  **Google-style** docstrings (Sphinx napoleon parses them).
- Per-file ignores: `scripts/python/*` and `run_agent.py` allow `E402`;
  `service/factory.py` allows `F821`.

Additional conventions:

- Prefer `pathlib.Path` over `os.path`.
- Use the configured loguru `logger` (`helper/logging.py`) — do **not** call
  `logging.getLogger`.
- Use `pydantic` for data models / config validation.
- Type-hint new code; tests can be lighter.

## Configuration & settings

There is **one global settings singleton**:
`from langgraph_agent_toolkit.core.settings import settings`. It is a
`pydantic_settings.BaseSettings` (`_base_settings.py`) with `extra="ignore"`,
loaded from the nearest `.env`.

- New config knobs go on the `Settings` class in `core/_base_settings.py` (with
  a sensible default), **and** the corresponding entry in `.env.example`.
- `settings.setup()` (called at import) applies `LANGGRAPH_<NAME>` env overrides
  (raw `setattr`, **bypasses Pydantic validation**) and loads `MODEL_CONFIGS` /
  `DB_CONFIGS` (inline JSON, base64, or file path).
- Multi-model setups use `MODEL_CONFIGS` (named provider/param dicts) looked up
  by `model_config_key`.
- Observability backends read their env vars from `os.environ` directly (e.g.
  LangSmith wants `LANGSMITH_*`), not from the `Settings` object — the entry
  points `load_dotenv()` the `.env` so those vars are present at runtime.
- Notable defaults: `ENV_MODE=production`, `HOST/PORT=0.0.0.0/8080`,
  `MEMORY_BACKEND` unset (→ no persistence), `OBSERVABILITY_BACKEND` unset (→
  `EMPTY` at runtime), `DEFAULT_AGENT=react-agent`, `CHECK_INTERRUPTS=False`,
  `CORS_ENABLED=False`, `AUTH_SECRET=None` (→ auth disabled).

## Architecture & core patterns

**Factory pattern (used by every `core/` subsystem).** Each follows the same
shape — copy it when adding a backend:

1. `types.py` — a `StrEnum` of backends (lowercase `auto()` values).
2. `base.py` — an abstract `Base*` interface.
3. one module per concrete backend.
4. `factory.py` — `*Factory.create(...)` coerces input → enum and `match`es it,
   raising `ValueError` on the default case.

Applies to **models** (`CompletionModelFactory`/`EmbeddingModelFactory`; OpenAI
uses `ChatOpenAIPatched`, everything else defers to LangChain `init_chat_model`;
`FAKE` → `FakeToolModel`), **memory** (`POSTGRES`/`SQLITE`; SQLite has no
store), **observability** (`LANGFUSE`/`LANGSMITH`/`EMPTY`, lazy-imported so
optional deps stay optional), and **prompts** (`PromptManager` +
`ObservabilityChatPromptTemplate`, files are the source of truth, rendered with
Jinja2).

**Agents are registered by import string, not filesystem scan.** `AgentExecutor`
takes `"module.path:object"` strings from `settings.AGENT_PATHS` and dynamically
imports them; an imported object can be a compiled graph or an `Agent`
dataclass. **Import failures are logged and silently dropped.** To add an agent:
create `agents/blueprints/<name>/agent.py` exposing a compiled graph or `Agent`,
then register its path in `AGENT_PATHS` (default list in `_base_settings.py`).
⚠️ `run_api.py` **hard-codes its own `AGENT_PATHS`** list, overriding the
settings default — update it there too if the service entry point must load your
agent. Several `react*` blueprints (`react_old`, `react_so_old`) are legacy
near-duplicates of `react` / `react_so`; confirm which variant is actually wired
before editing one.

**Service.** `service/handler.py:create_app()` is the FastAPI factory; an async
`lifespan` boots observability + memory checkpointer + `AgentExecutor` onto
`app.state` and flips readiness only after agents initialize. Routers: a public
one (health, `/`) and a `verify_bearer`-gated private one. Most agent endpoints
exist twice — `/{agent_id}/…` and a default-agent alias.

**Streaming = SSE.** The wire contract is
`data: {"type": "token"|"message"|"error", "content": …}` lines terminated by
`data: [DONE]`. Produced in `service/utils.py: message_generator`, parsed in
`client/client.py:_parse_stream_line`. Keep both ends in sync if you change it.

## Testing

- Framework: `pytest` + `pytest-asyncio` + `pytest-cov` + `pytest-env`
  (`[dependency-groups] tests`).
- `pythonpath = ["langgraph_agent_toolkit", "."]` (set in `pyproject.toml`).
- **`asyncio_mode` is unset → strict**: every async test **must** carry
  `@pytest.mark.asyncio`, or it silently no-ops. Don't forget the marker.
- The suite is heavily **mock-driven** — the LangGraph runtime is usually
  replaced by `AsyncMock`. Real coverage lives at the FastAPI route and
  `AgentClient` HTTP seams. The fake LLM is `ModelProvider.FAKE` →
  `FakeToolModel` (canonical response:
  `"This is a test response from the fake model."`).
- **Docker e2e** (`tests/integration/`) is gated behind `--run-docker` + the
  `docker` marker; a normal `pytest` run skips it.
- **Coverage caveat**: `.coveragerc` **omits `agents/blueprints/*`** (and
  `scripts/*`, `tests/*`) — blueprints are not in the coverage denominator.
  Codecov fails a PR only if project coverage drops >2%.
- Add tests for new features under the matching `tests/<area>/` directory.

## Adding dependencies

- **Runtime**: add to `[project].dependencies` in `pyproject.toml`, then
  `make uv_install_deps` (`uv sync --all-extras --no-install-project`).
- **Optional features**: use `[project.optional-dependencies]` extras — LLM
  providers (`openai`, `anthropic`, `google-vertexai`, `google-genai`, `aws`,
  `all-llms`), service backends (`uvicorn-backend`, `gunicorn-backend`,
  `aws-backend`, `azure-backend`), observability (`langfuse`, `langsmith`). The
  `client` group is the minimal Streamlit-only set.
- **Dev / lint / test / docs**: use the matching `[dependency-groups]` group.
- After any change, `uv.lock` must be updated — the pre-commit `uv-lock` hook
  enforces this. Prefer `~=` for closely-tracked libs, `>=,<` for broad ranges.

## Environment variables (`.env` / `.env.example`)

- `.env.example` is the committed **source of truth** for root (backend/app) env
  vars. `.env` is **gitignored** — never commit secrets.
- When code reads a new env var, add it to `.env.example` with a placeholder.
- Per-service container config lives in `configs/<svc>/.<svc>.env` (each has a
  committed `.example`). `configs/litellm/config.yaml` is gitignored and holds a
  real key locally — keep it out of commits.
- `.local.env` is used by the `scripts/python/` examples (they `load_dotenv`
  it), separate from the service's `.env`.

## Docker / local stack

`docker-compose.yaml` brings up the full stack: `lat-agent-backend` (:8080),
`lat-agent-frontend` (:8501), `litellm` (:4000), and a self-hosted Langfuse
stack (postgres, redis, clickhouse, minio, langfuse-web :3000, langfuse-worker).
Postgres bootstraps three DBs (`agents`, `litellm`, `langfuse`) + a
`checkpoints` schema via `scripts/postgres-init/`. Both app images are
single-stage `python:3.13-slim` built with `uv`. Use `make rebuild_api` /
`make rebuild_app` to rebuild one service.

## Pre-commit hooks (what runs on commit)

`pre-commit-hooks` basics (incl. `detect-private-key`, `detect-aws-credentials`,
`check-added-large-files --maxkb=1024`), `ruff` (fix + format), `codespell`,
`prettier` (md/yaml/toml/json/dockerfile/shell,
`--print-width=80 --prose-wrap=always`), and `uv-lock`. Pinned to `python3.13`.
Do not bypass with `--no-verify` unless explicitly asked.

## Finishing a task (verify before reporting done)

Always run an appropriate verification before declaring a task complete — don't
rely on the diff "looking right." Pick the lightest command that exercises the
change:

- **Code touching the package** → `uv run pytest` (or a focused
  `uv run pytest tests/<area>/test_x.py::test_y` when the suite is slow).
- **Style / imports** → `uv run ruff check --fix .` and `uv run ruff format .`.
- **Just before a commit** → `make pre_commit_run`.
- **Service / endpoint changes** → boot it
  (`uv run python langgraph_agent_toolkit/run_api.py` with
  `USE_FAKE_MODEL=true`) and hit `/health/ready`, or run the in-process e2e
  (`tests/service/test_service_e2e.py`).
- **New example script** → run it with a minimal config to confirm it boots.

If verification can't be run (no DB, needs real LLM keys, external service), say
so explicitly rather than implying success. Don't suppress errors to make a
command pass — fix the root cause.
