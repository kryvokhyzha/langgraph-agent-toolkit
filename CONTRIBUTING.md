# Contributing to LangGraph Agent Toolkit

Read [CLAUDE.md](CLAUDE.md) for the package architecture and project rules. Use
the [quickstart](docs/quickstart.rst) to run the first API example with a fake
model. It needs no provider credentials.

## Development setup

The package supports Python 3.11, 3.12, 3.13, and 3.14. Use Python 3.13 for
development: the pre-commit configuration requires it. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/), then clone your
fork:

```bash
git clone https://github.com/YOUR-USERNAME/langgraph-agent-toolkit.git
cd langgraph-agent-toolkit
git remote add upstream https://github.com/kryvokhyzha/langgraph-agent-toolkit.git
uv sync --frozen --no-install-project --extra all --python 3.13
uv run --no-sync pre-commit install
```

The sync command creates `.venv` and installs the locked development groups and
all optional features. Source imports use the repository path. Keep `--no-sync`
on later commands to preserve the selected environment.

Use `--extra all`, not `--all-extras`. The latter selects three incompatible
Langfuse SDK versions. To work with SDK v2, add `--extra langfuse-v2` to the
sync command. Use one selector from `langfuse-v2`, `langfuse-v3`, or
`langfuse-v4`. These extras select the Python SDK, not a Langfuse server. See
the [installation guide](docs/installation.rst) for smaller feature selections.

Keep real credentials outside committed files. Local tests disable `.env`
loading and use fake models or local protocol servers. Do not copy deployment
credentials into test fixtures.

## Development process

1. Create a branch for one change.
2. Implement the change and add a regression test for its behavior.
3. Run the relevant tests, then the full local suite.
4. Run pre-commit on all files before committing.
5. Update the affected guide and add a changelog entry when behavior changes.
6. Open a pull request against `main`.

Describe the problem, the new behavior, and the checks that passed in the pull
request. State any checks that could not run. Keep unrelated changes separate.

## Tests and checks

Run commands from the repository root:

```bash
uv run --no-sync pytest
uv run --no-sync pre-commit run --all-files
```

The default test suite needs no external service or model key. It covers real
graphs, temporary SQLite databases, in-process HTTP, and local MCP subprocesses.
PostgreSQL tests also run when `LAT_TEST_POSTGRES_DSN` is set. Use a disposable
test database for that setting.

Process journeys require a separate opt-in:

```bash
uv run --no-sync pytest tests/e2e tests/service/test_worker_recovery.py --run-e2e
```

These tests start local API processes and check streaming, persistence, restart,
and worker replacement. They use fake models. Worker signal tests require a
POSIX platform. Docker checks require running test containers and
`--run-docker`. Live Langfuse checks require `--run-langfuse` and a dedicated
test project. Real OpenAI checks require both `--run-e2e` and `--run-llm`, plus
explicit test credentials and a model name. They can incur provider charges. See
the [testing guide](docs/testing.rst) for setup and test boundaries.

Assert returned data, stored state, and failure behavior. Use mocks for external
dependencies. Avoid tests that only repeat a mocked return value or copy the
implementation. Remove duplicate tests only when another test preserves their
behavioral coverage. The coverage floor combines line and branch coverage; it
does not prove that every package path is correct.

## Build the documentation

From the repository root, install the locked documentation dependencies and the
optional features needed by the API reference:

```bash
uv sync --frozen --no-install-project --extra all --group docs
make -C docs html
```

The build regenerates `docs/generated`, then builds the full documentation.
Sphinx warnings cause failure. Open `docs/_build/html/index.html` to review the
result. Edit package docstrings or source guides instead of generated pages.

On Windows, run the same sync command. Then use Windows Command Prompt:

```bat
docs\make.bat html
```

## Code and documentation style

Use the conventions in [AGENTS.md](AGENTS.md) and [CLAUDE.md](CLAUDE.md). Write
comments, docstrings, and documentation in ASD-STE100 Simplified Technical
English. Use short, direct sentences. Preserve technical identifiers.

Pre-commit runs Ruff, codespell, Prettier, format checks, Gitleaks, credential
checks, and lockfile validation. Fix the reported cause before committing. Keep
secrets out of logs and test artifacts.

## Dependencies

Use a published extra for an optional runtime feature. Use a dependency group
for tools needed only in a source checkout:

```bash
uv add package-name
uv add --optional openai package-name
uv add --group tests package-name
uv lock
```

Choose the command for the dependency's role. The `openai` command above adds to
an existing extra; replace it with the relevant feature extra. Review the
version bounds and test the supported Python versions. Commit `pyproject.toml`
and `uv.lock` together. Reinstall the selected extras before testing a changed
dependency set.

## Questions and review

Open an issue for a reproducible bug or a proposed change. Include the package
version, selected extras, relevant configuration names, and a minimal example.
Remove credentials and private data. A maintainer reviews each pull request
before merge. Treat other contributors with respect.
