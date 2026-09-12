Dependency update review
========================

Review date: 2026-09-12.

This review uses stable releases from PyPI and the official release notes.
It excludes pre-release and withdrawn versions.
The tables record updates in implementation order. Later sections record
additional updates required by optional integrations.

.. list-table:: Reviewed dependency updates
   :header-rows: 1
   :widths: 45 20 20

   * - Package
     - Before
     - After
   * - ``langgraph``
     - 1.2.4
     - 1.2.11
   * - ``langgraph-checkpoint``
     - 4.1.1
     - 4.2.0
   * - ``langgraph-checkpoint-postgres``
     - 3.1.0
     - 3.1.2
   * - ``langgraph-checkpoint-sqlite``
     - 3.1.0
     - 3.1.1
   * - ``langgraph-sdk``
     - 0.4.2
     - 0.4.4
   * - ``langchain``
     - 1.3.4
     - 1.4.0
   * - ``langchain-core``
     - 1.4.2
     - 1.6.3
   * - ``langchain-openai``
     - 1.2.2
     - 1.6.2
   * - ``openai``
     - 2.41.1
     - 3.13.0
   * - ``langfuse``
     - 4.7.1
     - 4.15.2

``langchain-community==0.4.2`` and ``langgraph-prebuilt==1.1.0`` remain unchanged.
They are the latest stable versions at the review date.
This update does not upgrade all provider integrations or all transitive packages.
The OpenAI SDK update is required by the selected OpenAI integration.

The test suite still reports upstream deprecations for ``langchain-community``
and the supervisor package's use of ``create_react_agent``. The current package
ranges retain these supported APIs. A future LangGraph major upgrade requires
another compatibility review. Langfuse SDK v4 also warns about legacy trace I/O;
the adapter retains it for server v3 consumers and excludes SDK v5.

The release metadata is available from
`LangGraph on PyPI <https://pypi.org/project/langgraph/1.2.11/>`_,
`LangChain on PyPI <https://pypi.org/project/langchain/1.4.0/>`_,
`LangChain Core on PyPI <https://pypi.org/project/langchain-core/1.6.3/>`_, and
`Langfuse on PyPI <https://pypi.org/project/langfuse/4.15.2/>`_.

Changes that affect this package
--------------------------------

LangGraph
~~~~~~~~~

* Version 1.2.6 fixes checkpoint namespace inheritance in nested subgraphs.
  It also cancels active subgraphs when a v3 stream stops.
  See the `1.2.6 release notes
  <https://github.com/langchain-ai/langgraph/releases/tag/1.2.6>`_.
* Versions 1.2.5 through 1.2.9 fix several ``DeltaChannel`` state-update cases.
  These include new threads, overwrite snapshots, and state counters.
  Version 1.2.11 adds ``trace_policy`` to ``add_node``.
  See the `LangGraph release notes
  <https://github.com/langchain-ai/langgraph/releases>`_.
* The PostgreSQL and SQLite patches match checkpoint namespaces at segment
  boundaries. The PostgreSQL patch also fixes delta-history seed lookup.
  Checkpoint 4.2.0 adds optional ``omit_expired`` reads.
  See the `SQLite 3.1.1 release
  <https://github.com/langchain-ai/langgraph/releases/tag/checkpointsqlite%3D%3D3.1.1>`_
  and the `PostgreSQL 3.1.2 release
  <https://github.com/langchain-ai/langgraph/releases/tag/checkpointpostgres%3D%3D3.1.2>`_.

The toolkit keeps its current streaming API.
The dependency update does not enable v3 streaming or ``omit_expired``.
The PostgreSQL 3.1.0 and 3.1.2 schema migration lists are identical.
Keep a database backup before each deployment.
Use the checkpoint migration guide for conversation ownership changes.

LangChain
~~~~~~~~~

* Version 1.3.14 limits tool retries to retryable errors.
  See the `1.3.14 release notes
  <https://github.com/langchain-ai/langchain/releases/tag/langchain%3D%3D1.3.14>`_.
* Version 1.3.15 fixes approval gates that could fail open.
  It clears stale structured responses between saved turns.
  It preserves history when summarization fails.
  It also fixes orphaned tool calls and middleware control flow.
  These fixes affect the built-in ``create_agent`` agents.
  See the `1.3.15 release notes
  <https://github.com/langchain-ai/langchain/releases/tag/langchain%3D%3D1.3.15>`_.
* Version 1.4.0 adds ``langchain.mcp`` and ``MCPAdapter``.
  It reduces middleware trace input work and fixes model tool routing.
  The toolkit's optional ``mcp`` extra enables this integration.
  Operators select servers through ``MCP_SERVERS``.
  See the `1.4.0 release notes
  <https://github.com/langchain-ai/langchain/releases/tag/langchain%3D%3D1.4.0>`_.
* Core 1.6.0 adds standard model errors and fixes strict tool schemas.
  Core 1.6.3 permits gateway responses to set model and provider trace metadata.
  See the `Core 1.6.0 release
  <https://github.com/langchain-ai/langchain/releases/tag/langchain-core%3D%3D1.6.0>`_
  and the `Core 1.6.3 release
  <https://github.com/langchain-ai/langchain/releases/tag/langchain-core%3D%3D1.6.3>`_.

MCP and Studio compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The MCP integration uses ``langchain[mcp]>=1.4.0,<1.5`` and
``fastmcp>=4.0.3,<4.1``.
The lockfile selects FastMCP 4.0.3 and MCP SDK 2.2.0.
The base package keeps these dependencies optional.
The ``all`` extra and backend Docker image include them.
See :doc:`mcp` for configuration and compatibility limits.

MCP SDK 2 requires ``sse-starlette>=3``.
The previously locked Studio API required ``sse-starlette<2.2``.
These ranges cannot coexist.
The lockfile now uses the following compatible versions:

.. list-table:: Dependency changes required by MCP with Studio
   :header-rows: 1
   :widths: 45 20 20

   * - Package
     - Before
     - After
   * - ``langsmith``
     - 0.4.60
     - 0.6.9
   * - ``langgraph-api``
     - 0.6.35
     - 0.10.3
   * - ``langgraph-runtime-inmem``
     - 0.22.1
     - 0.30.3
   * - ``sse-starlette``
     - 2.1.3
     - 3.3.4

The LangSmith extra now permits ``>=0.4.60,<0.7``.
The new Studio API requires LangSmith 0.6.3 or later.
The LangGraph CLI stays at 0.4.29.
The toolkit's graph runtime stays at LangGraph 1.2.11.
The resolver also aligns the Studio gRPC packages at 1.80.0.

These choices use the official metadata for
`FastMCP 4.0.3 <https://pypi.org/project/fastmcp/4.0.3/>`_,
`MCP 2.2.0 <https://pypi.org/project/mcp/2.2.0/>`_, and
`Studio API 0.10.3 <https://pypi.org/project/langgraph-api/0.10.3/>`_.
The resolver checks the ``all`` feature set with Langfuse SDK 2.60.10 on Python
3.11 through 3.13.
SDK v2 keeps its required ``packaging<25`` constraint.
The CI matrix runs MCP tests with each supported Langfuse SDK generation.

The OpenAI integration now uses OpenAI SDK 3.13.0 in the lockfile.
The regression tests exercise HTTPX client injection with a local mock transport.
They cover synchronous and asynchronous requests, streaming, and gateway roles.
They do not call a live provider.

Deep Agents
~~~~~~~~~~~

The optional ``deepagents`` extra uses ``deepagents>=0.7.13,<0.8``.
The lockfile selects 0.7.13, released on 2026-09-02. This release fits the
existing LangChain 1.4.0, Core 1.6.3, and LangGraph 1.2.11 versions.
Its required provider and telemetry dependencies need these additional updates:

.. list-table:: Dependency changes required by Deep Agents
   :header-rows: 1
   :widths: 45 20 20

   * - Package
     - Before
     - After
   * - ``langchain-anthropic``
     - 1.4.4
     - 1.7.2
   * - ``anthropic``
     - 0.109.1
     - 1.5.0
   * - ``langchain-google-genai``
     - 4.2.4
     - 4.4.0
   * - ``google-genai``
     - 2.8.0
     - 2.23.0
   * - ``langsmith``
     - 0.6.9
     - 0.12.4

The corresponding optional ranges now allow these tested releases. Deep Agents
requires these provider packages even when an application selects another model.
The base package does not require Deep Agents. The ``all`` extra includes it.
The backend Docker build enables it only with ``INSTALL_DEEPAGENTS=true``.

Version 0.5.3 was the last release that fit all previous optional ranges.
The integration uses the current release to include later fixes for subagent
private state, composite backend routing, and filesystem tool dispatch.
Version 0.7 also requires an explicit StoreBackend namespace and removes the
default planning middleware. The toolkit adds planning explicitly and uses
StateBackend for thread-scoped virtual files. See :doc:`deepagents` for the
memory and execution limits.

Sources: `Deep Agents 0.7.13 metadata
<https://pypi.org/project/deepagents/0.7.13/>`_,
`Deep Agents 0.7 release notes
<https://github.com/langchain-ai/deepagents/releases/tag/deepagents%3D%3D0.7.0>`_,
`Anthropic integration 1.7.2 <https://pypi.org/project/langchain-anthropic/1.7.2/>`_,
`Google integration 4.4.0 <https://pypi.org/project/langchain-google-genai/4.4.0/>`_,
and `LangSmith 0.12.4 <https://pypi.org/project/langsmith/0.12.4/>`_.

Langfuse
~~~~~~~~

The Python SDK version and the Langfuse server version are separate.
The ``langfuse`` extra permits SDK versions from 2.60.10 to below 5.
Published extras now select one SDK generation: ``langfuse-v2`` requires
``>=2.60.10,<3``, ``langfuse-v3`` requires ``>=3.15.0,<4``, and ``langfuse-v4``
requires ``>=4.15.2,<5``. Use only one selector. The generic ``langfuse`` extra
can combine with any one selector. ``all-observability`` keeps its existing
generic Langfuse and LangSmith dependencies.
The default lockfile selects SDK 4.15.2.
The compatibility matrix tests SDK 2.60.10, 3.15.0, and 4.15.2.
See :doc:`langfuse_compatibility` to select the SDK for each server.

Recent SDK fixes include these changes:

* Version 4.14.0 fixes nested Anthropic cache usage and adds prompt linking
  through ``propagate_attributes``.
  See the `4.14.0 release notes
  <https://github.com/langfuse/langfuse-python/releases/tag/v4.14.0>`_.
* Version 4.14.2 fixes tool-definition normalization, zero sampling rates,
  and generator cancellation. It preserves the legacy score-create alias.
  See the `4.14.2 release notes
  <https://github.com/langfuse/langfuse-python/releases/tag/v4.14.2>`_.
* Version 4.15.1 avoids span formatting when debug logging is disabled.
  Version 4.15.2 updates generated API types.
  See the `4.15.1 release notes
  <https://github.com/langfuse/langfuse-python/releases/tag/v4.15.1>`_
  and the `4.15.2 release notes
  <https://github.com/langfuse/langfuse-python/releases/tag/v4.15.2>`_.

Server v4 changes the ingestion and read data models.
Python SDK v4 works with server v3.
SDK 4.7.0 or later sends data directly to the new v4 tables.
Older SDK combinations can require a migration mode or delayed propagation.
SDK v2 is rejected by server v4 in ``events_only`` mode.
See the `official server v3-to-v4 upgrade guide
<https://langfuse.com/self-hosting/upgrade/upgrade-guides/upgrade-v3-to-v4>`_.

Install and verify
------------------

Use the lockfile to reproduce the tested dependency set:

.. code-block:: bash

   uv sync --extra all --frozen --no-install-project
   uv run --no-sync pytest

Replace ``--all-extras`` in earlier source installation commands with
``--extra all``. The old flag selects the three incompatible SDK selectors.
The ``all`` extra keeps all features without forcing a selector. It can combine
with one selector, such as ``--extra all --extra langfuse-v2``, when the complete
dependency set is compatible.

For an application that uses Langfuse server v2, select SDK v2 and commit the
application's lockfile:

.. code-block:: bash

   uv add 'langgraph-agent-toolkit[langfuse-v2]'

For an application that must retain Python SDK v3:

.. code-block:: bash

   uv add 'langgraph-agent-toolkit[langfuse-v3]'

Use ``langfuse-v4`` for SDK v4. Add an exact SDK pin only when a specific patch
version is required. For example, add ``langfuse==2.60.10`` beside the v2 selector
to reproduce that test baseline. These selectors are published package extras,
not local dependency groups.

SDK v2 requires ``packaging<25``.
SDK v3 requires ``packaging<26``.
Resolve all application dependencies together.
Do not copy a v4 lockfile and replace only the installed Langfuse package.

The CI matrix installs each version selector with an exact SDK baseline.
It checks the resulting environment and runs the observability contracts.
The regular suite checks the default lockfile on Python 3.11, 3.12, 3.13, and 3.14.
Worker tests run with package coverage. Database tests have a separate job.
Live Langfuse server acceptance tests need a staging server for each deployed
server version and migration mode.

Python 3.14 and CI follow-up
--------------------------------

Review date: 2026-09-13. The package now permits Python 3.11 through 3.14.
Docker images and the LangGraph deployment configuration retain Python 3.13.
The Ruff target remains ``py311`` to preserve the minimum supported syntax.

Two dependency changes are needed for Python 3.14:

* Require ``pydantic>=2.13.0`` on Python 3.14. Pydantic 2.12 installs, but
  Langfuse's legacy models fail during import. The
  `Pydantic 2.13 release <https://pydantic.dev/articles/pydantic-v2-13-release>`_
  restores Python 3.14 support in ``pydantic.v1``. Python 3.11 through 3.13 keep
  their existing Pydantic minimum.
* Update locked ``jsonschema-rs`` from 0.29.1 to 0.44.1. Require at least 0.44.1
  in the ``studio`` extra on Python 3.14. The old native extension cannot build
  on this interpreter. The new release supplies compatible stable-ABI wheels
  and stays within ``langgraph-api==0.10.3`` requirements. See the
  `wheel metadata <https://pypi.org/pypi/jsonschema-rs/0.44.1/json>`_
  and `API requirements <https://pypi.org/pypi/langgraph-api/0.10.3/json>`_.

The other locked package versions remain unchanged. See :doc:`testing` for
the Python 3.14 and Langfuse SDK results.

CI now tests wheels, package behavior, process journeys, and each Langfuse SDK
on all four Python versions. PostgreSQL checks cover Python 3.13 and 3.14.
Lint and lockfile validation run once. Jobs have time limits, superseded pull
request runs stop, and Docker builds use separate caches for each image.
CI and the ``uv-lock`` pre-commit hook use uv 0.12.1.
Releases wait for the test workflow. The publish job checks the release tag,
builds distributions, and validates their metadata before it uploads to PyPI.
The metadata check uses isolated Twine v7. Twine v6 rejects the current
backend's metadata version 2.5. See the
`Twine changelog <https://github.com/pypa/twine/blob/main/docs/changelog.rst>`_.
If branch protection names the old ``test-e2e`` checks, replace them with the
corresponding ``test-python`` checks. Process tests now run in those jobs.

Dependabot uses its native ``uv`` updater to keep ``pyproject.toml`` and
``uv.lock`` consistent. It groups related minor and patch updates. Major and
security updates remain separate. It also checks GitHub Actions, Docker base
images, and pre-commit hooks. See the
`uv integration guide <https://docs.astral.sh/uv/guides/integration/dependabot/>`_.
Enable Dependabot security updates in the repository settings. The configuration
file alone does not enable that GitHub setting.
