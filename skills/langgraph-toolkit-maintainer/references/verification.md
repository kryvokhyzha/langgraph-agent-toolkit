# Select a test layer

Run commands from the toolkit repository root in its prepared environment. Read
`pyproject.toml` and `docs/testing.rst` if dependencies are missing.

## Focused checks

| Contract                              | Useful starting point                                                           |
| ------------------------------------- | ------------------------------------------------------------------------------- |
| Shared client/UI imports              | `tests/ui/test_minimal_imports.py`                                              |
| Model HTTP connections and cleanup    | `tests/core/test_llm_transport.py`                                              |
| Settings and secret handling          | `tests/core/test_config_secrets.py` and `tests/core/test_settings.py`           |
| Authentication and feedback ownership | `tests/service/test_auth.py` and `tests/service/test_feedback_authorization.py` |
| History and thread discovery          | `tests/service/test_functional_history.py` and `tests/service/test_threads.py`  |
| Real graph interrupt and resume       | `tests/service/test_real_graph_interrupts.py`                                   |
| Checkpoint write timing               | `tests/agents/test_checkpoint_durability.py`                                    |
| SQLite cancellation and locks         | `tests/core/test_sqlite_reliability.py`                                         |
| MCP lifecycle and tools               | `tests/core/test_mcp.py` and `tests/agents/test_mcp_tools.py`                   |

For example:

```sh
uv run --no-sync pytest tests/service/test_functional_history.py tests/service/test_threads.py
```

Assert an observable contract. Do not add tests that only count helper calls or
copy the implementation's branches. Use a local HTTP transport for wire errors,
not a replacement for the serializer under test.

## Process and database behavior

Use a real API process for restart, streaming completion, or cross-process
behavior:

```sh
uv run --no-sync pytest tests/e2e/test_api_journeys.py --run-e2e
uv run --no-sync pytest tests/service/test_worker_recovery.py --run-e2e
```

PostgreSQL tests require `LAT_TEST_POSTGRES_DSN` for a disposable database:

```sh
uv run --no-sync pytest tests/integration/test_postgres_reliability.py --run-postgres
```

These tests can terminate database sessions and create or remove test schemas.
Do not point them at an application database. Read the test fixture and supply
explicit test configuration before running them.

Use `--run-docker` only with the test API and UI containers configured as
described in `docs/testing.rst`. Run image import checks inside the final
images. Source-environment imports cannot establish that an image contains its
runtime dependencies.

## Optional observability and live models

Keep SDK and server versions separate. Langfuse extras select a Python SDK
range. Use the compatibility table before changing it. Inspect
`.github/workflows/test.yml` for the SDK test matrix. A successful offline SDK
test does not prove that a server stored a trace.

Live tests use explicit `LAT_TEST_*` credentials and opt-in flags. Read
`docs/testing.rst` and `docs/live_llm_testing.rst` for the selected integration.
Do not turn a normal regression run into a live-provider or production
observability test.

## Finish the requested change

For broad package changes, run the local suite and relevant process tests:

```sh
uv run --no-sync pytest
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
```

Build changed Sphinx documentation with warnings treated as errors:

```sh
uv run --no-sync sphinx-build -W --keep-going -b html docs docs/_build/html
```

Run the applicable pre-commit checks before a commit. Refresh `uv.lock` only
when dependency changes require it. Do not change an unrelated version or
release file as part of verification.
