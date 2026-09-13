Testing
=======

Use the smallest test layer that can detect the failure. Keep a regression test
for each corrected contract. Run the full local suite before release.

Install the locked dependencies and run the local suite:

.. code-block:: bash

   uv sync --extra all --frozen --no-install-project
   uv run --no-sync pytest

To test Python 3.14 without replacing the main environment, use:

.. code-block:: bash

   UV_PROJECT_ENVIRONMENT=.venv-py314 uv sync --python 3.14 --frozen \
     --extra all --no-install-project
   UV_PROJECT_ENVIRONMENT=.venv-py314 uv run --no-sync pytest --run-e2e --cov

Keep ``UV_PROJECT_ENVIRONMENT`` on both commands. This selects the same
interpreter and dependencies for installation and tests.

Use ``--extra all`` instead of the earlier ``--all-extras`` flag. The ``all``
extra includes the complete feature set. It avoids selecting the incompatible
``langfuse-v2``, ``langfuse-v3``, and ``langfuse-v4`` alternatives together.
To test all features with another SDK generation, add one selector, such as
``--extra all --extra langfuse-v2``. The dependencies must resolve together.

The local suite needs no external service or model credentials. It includes
temporary SQLite databases, in-process HTTP transports, and local MCP stdio
processes. Test setup disables loading credentials and database paths from
``.env``. PostgreSQL tests also run when ``LAT_TEST_POSTGRES_DSN`` is set.
Test setup also ignores exported ``LANGFUSE_*`` and
``LANGGRAPH_LANGFUSE_*`` project settings. Live Langfuse tests use only the
explicit ``LAT_TEST_LANGFUSE_*`` configuration.

Test layers
-----------

Use :doc:`load_testing` for repeatable HTTP pressure tests, provider faults,
database faults, worker replacement during traffic, and bounded soak tests.

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Layer
     - Contract
     - Location
   * - Focused tests
     - Validation, error mapping, configuration, middleware, and message formats.
     - ``tests/core``, ``tests/agents``, ``tests/client``, ``tests/helper``,
       ``tests/service``, and ``tests/ui``
   * - Component integration
     - Real graphs, tools, HTTP protocols, persistence, and dependency SDKs.
       Fakes replace model providers and external services.
     - Tests beside their components and ``tests/integration``
   * - API process journeys
     - A real service process, HTTP client, SQLite database, restart, and resume.
     - ``tests/e2e`` and ``tests/service/test_worker_recovery.py``
   * - Container checks
     - Built images, model transports, readiness, chat history, and HTML.
     - ``tests/integration/test_docker_e2e.py`` and
       ``tests/integration/docker_transport_smoke.py``
   * - External service integration
     - PostgreSQL locking and recovery; Langfuse ingestion and stored results.
     - PostgreSQL and Langfuse live tests in ``tests/integration``

The directories name the component under test. They do not always identify a
test layer. For example, ``tests/service/test_stream_integration.py`` uses an in-process
ASGI transport. The ``tests/e2e`` tests use a separate service process and TCP.

Component tests
---------------

Native graph tests use the production builders. They check local tool results,
structured output, bounded local retries, and remote calls without automatic
retries. The offline import test starts a fresh Python process and blocks
network connections. It imports every built-in blueprint and assigns a saver.

Deep Agents integration tests use the optional ``deepagents`` extra. They check
planning, virtual file isolation, SQLite recovery, subagent context, approval
resume, and MCP connection closure. A process test also reads a virtual file
after a forced API restart. Langfuse SDK tests export a real Deep Agent run
through each supported SDK generation. Internal summarization must stay out
of the user-facing stream.

MCP tests use real FastMCP and LangChain adapters. They cover tool discovery,
HTTP credentials, stdio, cancellation, timeouts, connection closure, and
elicitation. HTTP tests replace the network transport with an in-process
server. They retain the actual protocol and client setup.

Regression tests also check minimal-package imports, SSE comment and keepalive
frames, and history queries with omitted optional values. These checks protect
interfaces that can fail even when a mocked client returns a valid response.
SQLite service tests check both StateGraph and Functional API history. They
read existing checkpoints, append messages, stream one new answer, reopen the
database, and clear the conversation. Functional API history reads LangGraph's
private saved-state channel. Keep these tests when upgrading LangGraph.

API process journeys
--------------------

.. code-block:: bash

   uv run --no-sync pytest tests/e2e tests/service/test_worker_recovery.py --run-e2e

These tests start the service on a local port. They check shared-client
authentication, separate user memory and thread history, SSE and JSON Lines,
safe error responses, and concurrent updates. They restart the process and
check durable history and a pending interrupt. The worker tests kill a worker
and check replacement under Uvicorn and Gunicorn. They require POSIX signals.

CI runs these journeys on Python 3.11, 3.12, 3.13, and 3.14. The model is deterministic.
These tests do not measure model quality or simulate a production load.

Container checks
----------------

Start the backend with the fake model and a persistent checkpointer. Start the
frontend and set ``AGENT_URL``, ``APP_URL``, and ``AUTH_SECRET`` for the tests.

.. code-block:: bash

   uv run --no-sync pytest tests/integration/test_docker_e2e.py --run-docker

CI builds both images before this check. Tests call readiness, invoke, stream,
history, and clear-history endpoints. They check the frontend health endpoint
and HTML response. They do not control a browser or test Streamlit interaction.
An unavailable requested container causes failure.

The API image also has a separate transport check. It runs with the image's
runtime dependencies and no test packages. Build the image, then run:

.. code-block:: bash

   docker build -f docker/api/Dockerfile -t toolkit-api .
   docker run --rm --network none \
     --mount "type=bind,source=$PWD/tests/integration/docker_transport_smoke.py,target=/tmp/docker_transport_smoke.py,readonly" \
     toolkit-api python /tmp/docker_transport_smoke.py

This check verifies the image's aiohttp default and an explicit HTTPX override.
It checks actual SDK requests, streaming, usage, shared clients, and client closure. A local server
returns synthetic model responses. The test uses a dummy key and has no external
network access. CI runs it after building the default API image. It verifies the
installed transport dependencies; it does not verify a live model provider.

PostgreSQL integration
----------------------

Set ``LAT_TEST_POSTGRES_DSN`` to a disposable test database. The tests create
and remove test schemas and records.

.. code-block:: bash

   uv run --no-sync pytest tests/integration/test_postgres_reliability.py \
     tests/integration/test_postgres_migration.py \
     tests/core/test_concurrency.py tests/core/test_deadlocks.py --run-postgres

``--run-postgres`` fails when the DSN is missing. Tests check migration,
checkpoint recovery, lock release, cancellation, connection loss, and bounded
connection pools. CI supplies a PostgreSQL 16 service for this job.

Langfuse SDK compatibility
--------------------------

CI installs SDK 2.60.10, 3.15.0, and 4.15.2 separately on Python 3.11, 3.12,
3.13, and 3.14. Each installation selects the matching ``langfuse-v2``, ``langfuse-v3``,
or ``langfuse-v4`` extra and pins the exact SDK baseline. Each environment also
resolves the MCP and Deep Agents extras. Compatibility tests check callbacks,
prompts, feedback, configuration, and dependency contracts.

``tests/integration/test_langfuse_sdk.py`` uses the installed SDK. It runs real
graph callbacks and checks serialized HTTP requests and OpenTelemetry exports.
The receiving transport is controlled by the test. SDK tests therefore verify
the output protocol; live tests verify server ingestion and stored records.

Use a separate environment to test another SDK without changing the main one:

.. code-block:: bash

   uv venv .venv-langfuse
   uv pip install --python .venv-langfuse/bin/python \
     '.[langfuse-v2,openai,mcp,deepagents]' 'langfuse==2.60.10' \
     pytest pytest-asyncio pytest-env
   uv pip check --python .venv-langfuse/bin/python
   uv run --no-project --python .venv-langfuse/bin/python python -m pytest \
     tests/core/test_langfuse_compatibility.py tests/core/test_langfuse_settings.py \
     tests/core/test_observability.py tests/core/test_dependency_contracts.py \
     tests/integration/test_langfuse_sdk.py tests/integration/test_deepagents.py

Change both the selector and the exact SDK pin for another matrix entry.
Use ``langfuse-v3`` with ``langfuse==3.15.0`` or ``langfuse-v4`` with
``langfuse==4.15.2``. The selectors are published package extras. The exact pin
reproduces the CI baseline; it is optional for ordinary application installs.
See :doc:`langfuse_compatibility` for SDK and server selection.

Live Langfuse servers
---------------------

Use a dedicated test project. Set ``LAT_TEST_LANGFUSE_BASE_URL``,
``LAT_TEST_LANGFUSE_PUBLIC_KEY``, ``LAT_TEST_LANGFUSE_SECRET_KEY``, and
``LAT_TEST_LANGFUSE_SERVER_VERSION``. The version must match the exact server
version from its health endpoint. Use ``LAT_TEST_LANGFUSE_TIMEOUT`` to change
the ingestion wait limit from its 90-second default.

.. code-block:: bash

   uv run --no-project --python .venv-langfuse/bin/python python -m pytest \
     tests/integration/test_langfuse_live.py --run-langfuse

The tests run native ``create_agent`` and Deep Agent graphs with scripted
models and real local tools. The Deep Agent delegates a calculation to a
subagent. Tests read prompt versions, completed graph and model records, tool
inputs and outputs, parent links, user and session IDs, and feedback from the
server. They wait for ingestion to finish. Trace and score records keep
``lat-test-`` names. SDK v2 can also leave test prompts. Missing configuration,
an incorrect server version, or missing records causes failure.

The manual ``langfuse-live.yml`` workflow tests SDK v2 with server v2, all three
SDK generations with server v3, and SDK v4 with server v4. Configure the repository
environments ``langfuse-v2``, ``langfuse-v3``, and ``langfuse-v4``. Each needs
variables ``LANGFUSE_TEST_URL`` and ``LANGFUSE_TEST_VERSION``, plus secrets
``LANGFUSE_TEST_PUBLIC_KEY`` and ``LANGFUSE_TEST_SECRET_KEY``. Server v3 must be
at least 3.63.0 for SDK v3 and v4. A passing SDK matrix does not replace this
live-server result.

The manual workflow omits SDK v3 with server v4. Use the command above to test
this supported but deprecated combination. Set ``LAT_TEST_LANGFUSE_TIMEOUT=1200``.
This allows for the server's delayed processing of legacy SDK observations.

Live verification on 2026-09-12
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following checks ran for release ``0.10.0``. Each server used a disposable
database and test project. Each row passed both the native and Deep Agent tests
described above. These server tests use scripted models and real local tools.

.. list-table:: Completed live server checks
   :header-rows: 1
   :widths: 35 35 30

   * - Langfuse Server
     - Python SDK
     - Result
   * - 2.95.11
     - 2.60.10
     - 2 passed
   * - 3.225.7
     - 2.60.10
     - 2 passed
   * - 3.225.7
     - 3.15.0
     - 2 passed
   * - 3.225.7
     - 4.15.2
     - 2 passed
   * - 4.35.0
     - 4.15.2
     - 2 passed on the second attempt

The first server v4 attempt failed both cases. A prompt read timed out before
the native graph ran. The Deep Agent ran, but trace export timed out and the
stored-record check exceeded its 180-second limit. The records and feedback
became available later. During this attempt, the disposable server's worker
and ClickHouse were heavily CPU throttled. Ingestion jobs lost their worker
locks and were requeued automatically. No container ran out of memory or
restarted. Web health remained successful, so health alone did not establish
that ingestion was ready.

The second v4 attempt passed in 133.08 seconds after the worker limit changed
from 1 to 2 CPUs and the ClickHouse limit changed from 1.5 to 2 CPUs. Memory
limits stayed the same. The 10-second HTTP timeout and 180-second ingestion
limit also stayed the same. The v4 reader now combines duplicate observation
IDs only when all returned fields match. Conflicting rows fail the test.
The v4 checks also require user and session IDs on every observation,
including model calls and subagents.

SDK v3 with server v4 and server v4 migration modes were not checked in this
run. All v4 checks used the default ``events_only`` mode.

Azure OpenAI ``gpt-4o-mini-2024-07-18`` also ran through
``CompletionModelFactory`` with real model calls. The native agent called
``multiply`` and returned 42. The Deep Agent called ``multiply``, wrote a report,
delegated review through ``task``, and read the report. A new graph and SQLite
connection then read the saved report. These graph checks used 12 model calls
and 29,179 reported tokens. A separate connection probe used 18 tokens.

The built ``0.10.0`` wheel was also installed in a clean environment with the
locked ``deepagents``, ``openai``, and ``langfuse`` extras. Its dependency check
passed. The production app ran through its ASGI API with Azure, a temporary
SQLite database, and server v4. Invocation, JSON Lines streaming, virtual-file
retrieval, history, and feedback passed. This used four model calls and 13,413
reported tokens. This check used an in-process HTTP transport; worker process
replacement is covered by the separate E2E tests.

That combined run logged Langfuse export timeouts and its first trace read
timed out. The harness used one-observation batches and the toolkit's default
5-second Langfuse timeout. The v4 server also timed out on later diagnostic
reads after the successful compatibility test. Thus, the passing matrix does
not establish consistent responsiveness in this resource-limited environment.
A separate read-only check retried the same API trace IDs 19 times within
180 seconds and also failed with read timeouts. The API flow is verified, but
complete stored tracing and token totals for that combined run are not
verified. No extra model calls were made for these read retries.
Later diagnostics could read the first API trace with either field selection.
The final complete-record check reached the server after teardown had started
and could not run. That connection failure does not indicate a package defect.

These checks verify tool execution, checkpoint recovery, and stored tracing
contracts. The scripted server tests do not verify model token accounting.
These checks do not measure general model quality or production load.

Real model requests
-------------------

The opt-in OpenAI journey uses the real provider with both HTTPX and aiohttp.
It checks tools, concurrent streaming, usage, cancellation, and recovery with at
most twelve model requests. It is separate from the offline SDK and load tests.
See :doc:`live_llm_testing` for configuration, budgets, and the exact scope.

Connection and overload tests
-----------------------------

Connection and overload regressions use local HTTP endpoints and disposable
databases. They do not call a paid model provider:

.. code-block:: bash

   uv sync --frozen --no-install-project --extra all
   uv run --no-sync pytest tests/core/test_llm_transport.py tests/service/test_admission.py \
     tests/service/test_stream_transport_cleanup.py tests/service/test_blocking_feedback.py
   uv run --no-sync pytest tests/integration/test_postgres_reliability.py --run-postgres

Set ``LAT_TEST_POSTGRES_DSN`` to a disposable PostgreSQL database for the second
test command. The tests create temporary schemas, terminate their own sessions,
and hold locks to verify recovery. OpenAI and Azure transport tests use the real
SDK with a local HTTP server. They check connection reuse, pool saturation,
bounded retries, streaming stalls, dropped sockets, cancellation, proxy and
credential separation, embeddings, and successive application lifespans.

ASGI tests check overload before body reads, mounted health probes, optional
queue cancellation, slow sends, disconnect cleanup, and stalled-cleanup health
signals. Real graph tests verify trace context cleanup and conversation ordering
for both SSE and JSON Lines. Feedback tests retain a thread slot until its real
synchronous call finishes. See :doc:`reliability` for deployment limits.

The initial combined reliability run on 2026-09-12 passed 695 tests, with 12
expected skips and 80.50% combined line/branch coverage. It included PostgreSQL 16.15 fault
tests and both Uvicorn and Gunicorn worker replacement. The skipped tests required
live Langfuse/Docker services or a different Langfuse SDK generation.

After the pressure-test fixes, the full regression suite passed 800 tests with
12 skips. That run included real PostgreSQL faults and both worker supervisors;
it did not collect a new coverage percentage. After the final load-driver changes,
186 focused load, provider-transport, error-response, and factory tests passed.
See :doc:`load_test_results` for
the separate load measurements, failures, and follow-up runs.

A separate source matrix checked SDKs 2.60.10, 3.15.0, and 4.15.2 against SDK
tracing, executor cleanup, and both streaming formats. SDK v2 passed 36 tests.
SDK v3 and v4 each passed 28 tests with eight version-specific skips. This matrix
used local HTTP interception, not live Langfuse servers. A final focused rerun
of the observability and SDK wire tests passed 20 tests on each of those three
SDK versions. The rebuilt base wheel also passed two application lifespans, fake-model invocation, SQLite persistence,
and resource cleanup without model-provider or Deep Agents extras.

Dependency update review
------------------------

Dependabot checks Python and Docker updates each Tuesday. It checks GitHub
Actions and pre-commit updates monthly. Routine updates have a seven-day
cooldown. The open version-update PR limits are five for Python, two for
Actions, one for Docker, and one for pre-commit.

Python updates retain separate framework, observability, development-tool, and
general dependency groups. Major upgrades remain separate. Hook revisions share
one group without a semantic-version filter. Some hook tags have a ``v`` prefix
that the updater cannot classify for a minor/patch group. Review major hook
changes within the grouped PR.

Security updates use a separate schedule and do not wait for the routine
cooldown. Compatible Python security updates share a separate group. Major
security updates remain separate. These settings do not enable Dependabot
alerts or security updates in repository settings. Keep those features enabled.
See the `Dependabot options reference
<https://docs.github.com/en/code-security/reference/supply-chain-security/dependabot-options-reference>`_.

Lower PR limits do not close existing PRs. A PR that already exists can occupy
a slot until it is merged or closed. Review obsolete PRs after a manual upgrade
or a dependency-group change. Do not merge an old lockfile across a package
refactor. Rebase the PR or let Dependabot create an update from the current base.

Check the complete CI matrix before merging a dependency update. A green wheel
import or lint job does not establish API, database, or Python-version
compatibility. Reproduce failures with the PR's exact ``pyproject.toml`` and
``uv.lock`` in an isolated environment. Run the failing job with coverage when
CI uses coverage. A short timing test can behave differently with that overhead.

The pull-request hook job runs the revisions in ``.pre-commit-config.yaml``.
It stages the PR changes in its disposable checkout because Gitleaks and the
file-size hook inspect staged changes. It then runs the configured hooks on
those files. This preparation does not change the files or other CI jobs.

When hook configuration changes, an additional smoke check uses representative
Python, YAML, and TOML files. A YAML-only PR would otherwise skip Python hooks.
This checks hook installation and execution. The smoke check does not replace
a complete repository review after a major formatter or linter upgrade.
Hook failures and automatic formatting changes fail the job.

Coverage and test review
------------------------

.. code-block:: bash

   uv run --no-sync pytest --cov --cov-report=term-missing

Coverage includes the whole package, including blueprints, UI, and legacy
creators. Subprocess coverage is enabled. The 77% minimum combines executed
lines and branch destinations. It is not a 77% branch-coverage requirement.
Read line and branch totals separately when reviewing the report.

CI collects coverage from the package tests and API process journeys on each
supported Python version. Each job enforces the 77% minimum and retains its
test and coverage reports. Codecov receives one complete Python 3.13 report.
This avoids combining repeated uploads from the Python matrix. Codecov fails
a project decrease of more than two percentage points. Patch coverage remains
informational. Upload failures fail trusted CI runs. Fork and Dependabot runs
still enforce the local coverage minimum when an upload is unavailable.

If tokenless uploads are disabled, configure ``CODECOV_TOKEN`` as both an
Actions secret and a Dependabot secret. Fork pull requests do not receive these
secrets. Keep the Codecov GitHub integration enabled for remote status checks.

Prefer assertions on returned data, stored state, resource closure, and failure
behavior. Keep focused unit tests where they explain a boundary. Remove a
duplicate only when another test checks the same contract. Avoid tests that
copy a production implementation, assert only a mock return value, or check
incidental wording. Test production builders directly. Use a fresh process
when the contract concerns module import behavior.

Coverage does not establish complete correctness. Live Langfuse checks require
configured servers. Browser interaction, other model/provider combinations,
and sustained production load need separate verification.

Python 3.14 verification
------------------------

On 2026-09-13, the full local suite on CPython 3.14.3 passed 938 tests with
36 skips and 82.08% combined line/branch coverage. The run included API process
journeys, worker replacement, SQLite persistence, MCP, Deep Agents, and both
OpenAI HTTP transports. It used synthetic model responses. External service
tests were not enabled.

Separate Langfuse SDK checks used both Pydantic 2.13.0 and 2.13.4. For each
Pydantic version, SDK v2 passed 46 tests with one skip. SDK v3 and v4 each passed
39 tests with eight skips. Those skips select contracts for another SDK
generation. Pydantic 2.12.5 reproduced the legacy SDK import failure on Python
3.14. The conditional dependency minimum prevents that installation.

A separate Python 3.14 run passed 43 PostgreSQL 16 fault, migration, concurrency,
and deadlock tests. The database used a disposable container, which was removed
after the tests. A fresh wheel passed base-service and UI checks outside the
source checkout. The complete frozen ``all`` installation also passed. The
actual Studio development server passed readiness, assistant discovery, and a
deterministic graph run. These checks used no live model or Langfuse service.

The exact CI dependency set and test command also passed on CPython 3.13.12:
938 tests passed, 36 skipped, and 82.16% combined line/branch coverage. This
check included process journeys and generated the canonical Codecov XML report.
