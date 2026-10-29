---
name: langgraph-toolkit-maintainer
description:
  Change and verify the langgraph-agent-toolkit package repository. Use for
  package bug fixes, dependency upgrades, service or SDK contracts, optional
  integrations, and focused regression tests. Use application-facing skills for
  ordinary downstream agent or client development.
---

# Maintain the toolkit

Use this workflow in a source checkout of `langgraph-agent-toolkit`. Follow that
checkout's `AGENTS.md` and `CLAUDE.md`. Inspect local changes before editing. Do
not treat a downstream application as the toolkit repository.

## Find the contract

Trace the public entry point to its implementation and a test that exercises it.
For a reported failure, first establish whether it occurs in the pinned
dependencies. Keep a regression test for confirmed behavior changes.

| Change                   | Read together                                                           |
| ------------------------ | ----------------------------------------------------------------------- |
| HTTP request or response | `schema/`, `service/routes.py`, `client/client.py`, and UI consumers    |
| Agent startup            | `agents/agent.py`, `agents/agent_executor.py`, and `service/handler.py` |
| Settings or credentials  | `core/_base_settings.py`, `core/config_secrets.py`, and `.env.example`  |
| Persistence              | `core/memory/`, service identity helpers, and real graph tests          |
| Provider integration     | Matching factory, transport lifecycle, and minimal-extra tests          |

Package paths in the table are under `langgraph_agent_toolkit/`.

## Preserve package boundaries

- Shared schema and settings imports must work in the minimal client/UI
  environment. Keep agent runtime imports out of that import path.
- Keep optional integrations lazy. A missing optional extra must not break an
  unrelated feature. State the required extra when an optional feature is used.
- Keep required startup failures visible. Do not keep a worker ready when its
  agent or configured database failed to initialize.
- Preserve `SecretStr` for credentials. Unwrap secrets at the SDK or transport
  boundary. Do not mask ordinary model names, URLs without credentials, or
  numeric settings as secrets.
- Preserve explicit authentication and saver choices. Add migration notes when a
  public default or response contract changes.

## Select verification

Read [verification.md](references/verification.md) for test locations and
commands. Use the smallest layer that exercises the changed contract. Test real
graphs and persistence for history or recovery behavior. A mocked saver cannot
establish that checkpoints survive a restart.

Use `uv run --no-sync` in a prepared environment. This retains selected extras.
For a fresh environment, check the required uv version in `pyproject.toml` and
follow the repository's dependency setup. Use `--extra all` when the full
feature set is required; `--all-extras` selects conflicting Langfuse SDK ranges.

Do not bump versions, publish, tag, or change release notes unless the task
requires that work. Report the behavior changed, verification performed, and any
test that could not run. Separate scripted-model results from live-provider
results.

## Source map

- [Contributor workflow](https://github.com/kryvokhyzha/langgraph-agent-toolkit/blob/main/docs/contributing.rst)
- [Test layers](https://github.com/kryvokhyzha/langgraph-agent-toolkit/blob/main/docs/testing.rst)
- [Dependency updates](https://github.com/kryvokhyzha/langgraph-agent-toolkit/blob/main/docs/dependency_updates.rst)

Use the checked-out branch's files when they differ from these source links.
