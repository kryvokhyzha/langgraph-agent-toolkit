<div align="center">
  <img alt="LangGraph Agent Toolkit" src="https://raw.githubusercontent.com/kryvokhyzha/langgraph-agent-toolkit/main/docs/media/logo.svg" width="260">
</div>

# LangGraph Agent Toolkit

[![Tests](https://github.com/kryvokhyzha/langgraph-agent-toolkit/actions/workflows/test.yml/badge.svg)](https://github.com/kryvokhyzha/langgraph-agent-toolkit/actions/workflows/test.yml)
[![Documentation](https://github.com/kryvokhyzha/langgraph-agent-toolkit/actions/workflows/sphinx.yml/badge.svg)](https://github.com/kryvokhyzha/langgraph-agent-toolkit/actions/workflows/sphinx.yml)
[![PyPI](https://img.shields.io/pypi/v/langgraph-agent-toolkit.svg)](https://pypi.org/project/langgraph-agent-toolkit/)

Serve LangGraph agents through an authenticated HTTP API. Add streaming,
persistent conversation history, managed model connections, and optional tools
and observability. Use your own client or the included Streamlit interface.

**Python 3.11–3.14.** The API image uses Python 3.13. Existing deployments
should read the [0.10.x migration guide](docs/migration.rst) and
[changelog](CHANGELOG.md) before upgrading.

[Quickstart](#quickstart) · [Integrations](#choose-an-integration) ·
[Onboarding](docs/onboarding.rst) ·
[Memory and authentication](#memory-and-authentication) ·
[Deployment](#deployment) · [Documentation](#documentation) ·
[Development](#development)

## What the toolkit provides

- FastAPI routes for invocation, SSE, JSON Lines, history, and feedback.
- Sync and async `AgentClient` interfaces, including multimodal messages.
- SQLite or PostgreSQL checkpoints with conversation ownership and ordered
  updates across workers.
- Model factories with connection reuse, timeouts, bounded retries, and an
  optional aiohttp transport for OpenAI and Azure.
- Native LangChain agents, custom LangGraph workflows, human approvals, and
  optional Deep Agents.
- MCP tools, Langfuse SDK v2/v3/v4 integration, LangSmith, and local or managed
  prompts.

Install only the integrations that your service needs. LiteLLM, Langfuse, MCP,
Deep Agents, and Streamlit are optional.

## Quickstart

This local demo uses a deterministic fake model and SQLite. It needs no model
key, Docker, or external service. It checks API behavior and persistence. Model
quality requires separate evaluation.

Install `uv`, then use a fresh checkout:

```sh
git clone https://github.com/kryvokhyzha/langgraph-agent-toolkit.git
cd langgraph-agent-toolkit
uv sync --frozen --no-install-project --no-dev --extra uvicorn-backend
```

Create `.env` in this fresh checkout with these values. Use a shell without
other toolkit or tracing overrides.

```dotenv
USE_FAKE_MODEL=true
AUTH_MODE=trusted
AUTH_SECRET=local-demo-token
AGENT_PATHS=["langgraph_agent_toolkit.agents.blueprints.chatbot.agent:chatbot_agent"]
DEFAULT_AGENT=chatbot-agent
MEMORY_BACKEND=sqlite
SQLITE_DB_PATH=quickstart.sqlite
OBSERVABILITY_BACKEND=empty
MCP_SERVERS={}
MODEL_CONFIGS={}
LANGSMITH_TRACING=false
LANGCHAIN_TRACING_V2=false
```

Start the API on localhost:

```sh
uv run --no-sync python -m langgraph_agent_toolkit.run_api --host 127.0.0.1 --port 8080
```

In another terminal, check readiness and send a request:

```sh
export AUTH_SECRET=local-demo-token

curl --fail http://127.0.0.1:8080/health/ready

curl --fail-with-body http://127.0.0.1:8080/chatbot-agent/invoke \
  -H "Authorization: Bearer ${AUTH_SECRET}" \
  -H 'Content-Type: application/json' \
  -d '{"input":{"message":"Hello"},"user_id":"demo-user","thread_id":"demo-thread"}'
```

The response content is `This is a test response from the fake model.` Reuse the
same `user_id` and `thread_id` for the next turn. The SQLite file retains
history after the API stops. Use the demo token only for this local example.

Use the [quickstart guide](docs/quickstart.rst) for streaming, saved history,
the Python client, the UI, and real-model configuration.

## Choose an integration

| Need                                         | Start with                                          | Guide                                                     |
| -------------------------------------------- | --------------------------------------------------- | --------------------------------------------------------- |
| Chat without tools                           | `chatbot`                                           | [Agent patterns](docs/integrations.rst)                   |
| A model that selects business tools          | Native `create_agent`                               | [Usage](docs/usage.rst)                                   |
| Structured extraction                        | `create_agent_structured`                           | [Agent patterns](docs/integrations.rst)                   |
| A fixed workflow or required approval        | Custom `StateGraph` or human-in-the-loop middleware | [Usage](docs/usage.rst)                                   |
| Planning, intermediate files, and delegation | Optional Deep Agents                                | [Deep Agents](docs/deepagents.rst)                        |
| Tools exposed by another service             | MCP, combined with a supported agent                | [MCP](docs/mcp.rst)                                       |
| Traces, prompts, and feedback                | Langfuse or LangSmith                               | [Langfuse compatibility](docs/langfuse_compatibility.rst) |

For an existing Python project, select provider and backend extras:

```sh
uv add 'langgraph-agent-toolkit[openai,uvicorn-backend,langfuse-v4]'
```

Follow the [application onboarding guide](docs/onboarding.rst) to register
agents, set identity and memory contracts, connect a client, and verify the
deployment. It also gives the upgrade sequence for existing applications.

`langfuse-v2`, `langfuse-v3`, and `langfuse-v4` select a Python SDK version
range. Choose one. SDK and server versions are separate. Use the compatibility
guide for supported combinations. `mcp`, `deepagents`, `ui`, and
`openai-aiohttp` are separate extras. See [installation](docs/installation.rst)
for all options.

## Memory and authentication

`thread_id` identifies short-term conversation state. A long-term store can use
`user_id` to identify one user across threads. Passing `user_id` does not create
a store. SQLite supplies checkpoints but has no long-term store.

The service separates conversation storage by authenticated user, agent, and
public thread ID. Keep these values consistent when reading or updating history.

For one deployment per client, keep one `AUTH_SECRET`. The default
`AUTH_MODE=trusted` preserves 0.9.2 shared bearer-token authentication.
`user_id` is optional. Your trusted application can supply an end user's ID. If
it omits the field, the service uses `AUTH_SERVICE_USER_ID` (default:
`service`). Keep that identity stable for each conversation.

No new auth header or per-user token is required. If an existing deployment sets
`AUTH_MODE=token`, remove that override or set it to `trusted` to use the
compatible behavior. Token mode remains available as an explicit option.
`AUTH_USERS` tokens always identify one user, in either mode. The service
rejects a different supplied user ID for those tokens.

Authentication compatibility does not remove the history pagination, checkpoint
migration, or validation changes. See the
[authentication and migration guide](docs/migration.rst) for curl examples and
existing checkpoint migration.

## Deployment

The API can run alone under Uvicorn or Gunicorn. Docker Compose adds the
optional frontend, model proxy, and observability services. Configure the full
stack through the [environment guide](docs/environment_setup.rst).

- The API Docker image installs and selects **aiohttp** for managed async OpenAI
  and Azure calls. Python installations default to HTTPX. Set
  `LLM_HTTP_ASYNC_TRANSPORT=httpx` to override the image default.
- Each worker accepts eight active requests by default. Excess work receives
  `503` before an agent run starts. The optional admission queue is bounded.
- Use PostgreSQL for replicas across hosts. SQLite workers must share a file on
  a local filesystem. Configure persistence; the default backend is unset.
- Supervisors replace failed workers. Requests running in a failed worker can
  fail. The toolkit does not provide durable jobs or exactly-once tool writes.

Read [deployment](docs/deployment.rst) for health probes and worker recovery,
and [reliability](docs/reliability.rst) for database pools, connection limits,
timeouts, cancellation, and retry boundaries.

## Documentation

Read the
[documentation site](https://kryvokhyzha.github.io/langgraph-agent-toolkit/) or
the source guides below. The running service exposes its HTTP schema at `/docs`
and `/openapi.json`.

| Task                                 | Guide                                                                                                                                                            |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Make the first request               | [Quickstart](docs/quickstart.rst)                                                                                                                                |
| Add the toolkit to an application    | [Onboarding](docs/onboarding.rst)                                                                                                                                |
| Select extras and configure services | [Installation](docs/installation.rst), [environment](docs/environment_setup.rst)                                                                                 |
| Call the API or register an agent    | [Usage](docs/usage.rst)                                                                                                                                          |
| Choose agents and tools              | [Integrations](docs/integrations.rst), [MCP](docs/mcp.rst), [Deep Agents](docs/deepagents.rst)                                                                   |
| Upgrade an existing deployment       | [Migration](docs/migration.rst), [dependency review](docs/dependency_updates.rst), [changelog](CHANGELOG.md)                                                     |
| Deploy and operate the service       | [Deployment](docs/deployment.rst), [reliability](docs/reliability.rst)                                                                                           |
| Configure Langfuse                   | [SDK and server compatibility](docs/langfuse_compatibility.rst)                                                                                                  |
| Verify behavior and capacity         | [Testing](docs/testing.rst), [live model checks](docs/live_llm_testing.rst), [load tests](docs/load_testing.rst), [recorded results](docs/load_test_results.rst) |

## Development

Install the locked dependencies and run the local tests. The pre-commit hook
environments require Python 3.13.

```sh
uv sync --frozen --no-install-project --extra all
uv run --no-sync pytest
uv run --no-sync pre-commit run --all-files
```

Use `--extra all` because `--all-extras` selects incompatible Langfuse SDK
versions. Ordinary tests use fake models and local services. Process, Docker,
PostgreSQL, live Langfuse, and real-model checks have separate setup
requirements. Test coverage and local load results do not establish production
capacity or model quality.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow. Examples
are in [scripts/python](scripts/python), including the
[Deep Agents example](scripts/python/10-deep-agent.py).

## License

[MIT](LICENSE).
