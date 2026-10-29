Dependency update review
========================

CA security follow-up: 2026-10-28
---------------------------------

The `Python security notice
<https://discuss.python.org/t/information-for-python-users-regarding-emudhra-emsign-ca-certificate-security-incident/109576>`_
requires updated certificate bundles after the eMudhra intermediate-key incident.
Both the base package and the source-only client group now require
``certifi>=2026.10.19``. The existing lock already selects 2026.10.23.

CI, Docker builds, and the lock hook now use uv 0.13.6. Source checkouts reject
older uv versions. This includes the certificate fixes from uv 0.13.4 and 0.13.5
and the managed Python updates in `uv 0.13.6
<https://github.com/astral-sh/uv/releases/tag/0.13.6>`_. The new uv lock format
changes metadata; it does not change the selected package versions.

The Docker base is Python 3.13.18. Both images ensure ``pip>=26.2.2`` and install
current system CA certificates. The `pip release notes
<https://pip.pypa.io/en/stable/news/#v26-2-2>`_ confirm that this release updates
pip's private certifi copy. Updating application dependencies alone does not
update that copy or the pip wheel bundled with ``ensurepip``. Rebuild both
images with ``--pull --no-cache`` and recreate the containers.

``[tool.uv.exclude-newer-package]`` sets ``certifi = "0 days"``. Dependabot also
excludes certifi from its seven-day cooldown. The project has no general uv
release-age delay. The per-package setting permits certificate updates if a
general delay is supplied later. It does not upgrade an existing lockfile;
review and merge the dependency update. Other Dependabot rules stay unchanged.

Current baseline: 2026-10-27
----------------------------

The 0.10.2 candidate now selects the latest stable releases listed below.
See :doc:`dependency_releases_2026_10` for each intervening release and source.
These versions supersede the historical reviews later in this document.

.. list-table:: Current dependency baseline
   :header-rows: 1
   :widths: 45 20 20

   * - Package
     - Previous lock
     - Current lock
   * - ``deepagents``
     - 0.7.23
     - 0.7.32
   * - ``langchain``
     - 1.4.4
     - 1.4.7
   * - ``langchain-core``
     - 1.6.9
     - 1.7.0
   * - ``langgraph``
     - 1.2.11
     - 1.2.18
   * - ``langgraph-checkpoint``
     - 4.2.0
     - 4.3.1
   * - ``langgraph-checkpoint-postgres``
     - 3.1.2
     - 3.2.0
   * - ``langgraph-checkpoint-sqlite``
     - 3.1.1
     - 3.1.2
   * - ``langchain-openai``
     - 1.6.2
     - 1.7.3
   * - ``langchain-anthropic``
     - 1.7.5
     - 1.7.6
   * - ``langchain-aws``
     - 1.5.1
     - 1.9.1
   * - ``langchain-google-genai``
     - 4.4.0
     - 4.4.2
   * - ``langchain-google-vertexai``
     - 3.2.3
     - 3.2.5
   * - ``langsmith``
     - 0.14.7
     - 0.14.14
   * - ``fastmcp``
     - 4.0.11
     - 4.1.4
   * - ``mcp``
     - 2.2.0
     - 2.3.0
   * - ``langgraph-cli``
     - 0.4.29
     - 0.4.40
   * - ``langgraph-api``
     - 0.10.3
     - 0.16.7
   * - ``langgraph-runtime-inmem``
     - 0.30.3
     - 0.36.7

``langchain-community==0.4.2`` and ``langgraph-supervisor==0.0.31`` remain
unchanged. No newer stable release exists at this review date.
The lockfile retains OpenAI SDK 3.13.0 and the Langfuse SDK generation selectors.
The service still uses the same HTTP routes and streaming format.
Authentication and ``CHECKPOINT_DURABILITY=sync`` keep their current defaults.

The minimum Core version is now 1.7.0. The OpenAI integration requires it.
The package also requires Checkpoint 4.3.1 and Prebuilt 1.1.1.
These minimums preserve namespace validation and invalid-resume fixes when
an application adds its own dependency pins.
The minimum PostgreSQL saver version is now 3.2.0.
Its new pruning API is available to embedded applications. The service does not
run it automatically. ``keep_latest`` refuses threads with ``DeltaChannel``
state, which includes the Deep Agents blueprint. Do not delete its supporting
checkpoint history with custom SQL.

Rebuild deployment images to refresh operating-system trust stores.
FastMCP's release notes also require current CA certificates. The Python
lockfile now selects ``certifi==2026.10.23``. Updating that package alone does
not update the operating-system or custom trust stores.

Langfuse follow-up
------------------

The later Langfuse review changes the default SDK from 4.15.2 to 5.0.4.
The explicit SDK v4 selector now uses 4.18.0.
SDK v2 and v3 remain 2.60.10 and 3.15.0.
SDK v5 requires server 4.59.0 or later. Use an explicit older selector for
server v3. The generic extra now permits ``>=2.60.10,<6``.
The server Compose example selects 4.71.0 for fresh deployments.
It requires an explicit version before it can start existing volumes.

See :doc:`langfuse_compatibility` for each SDK release and
:doc:`langfuse_server_upgrade` for service changes and database upgrade steps.
The earlier verification counts below describe the state before this follow-up.

Verification on 2026-10-27
--------------------------

The full suite, including local API process journeys, passed 1,120 tests.
It skipped 51 tests that need explicit external services or live-test flags.
Combined line and branch coverage was 84.35 percent. The required floor is 77.
Focused dependency tests also passed on Python 3.11 and 3.14.
The PostgreSQL 16 suite passed 37 tests on Python 3.12 and 3.14.

Offline Langfuse SDK checks passed with versions 2.60.10, 3.15.0, and 4.15.2.
Model serialization tests used real SDKs with local HTTP transports.
Provider construction and tool binding used blocked network connections.
No live model or Langfuse server was called.
The Studio CLI loaded, but a running Studio deployment was not tested.

Historical reviews
------------------

Initial review date: 2026-09-12. Multimodal follow-up date: 2026-10-10.
The version statements below describe those earlier reviews.

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
the adapter retains it for server v3 consumers. The initial review excluded
SDK v5. The later Langfuse follow-up adds its current API handling.

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
The 2026-10-10 lockfile selected FastMCP 4.0.11 and MCP SDK 2.2.0.
The initial MCP update selected FastMCP 4.0.3.
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

The initial MCP update allowed LangSmith ``>=0.4.60,<0.7``.
The later Deep Agents update raises that requirement as described below.
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

The 2026-10-10 optional extra used ``deepagents>=0.7.23,<0.8``.
That lockfile selected 0.7.23, released on 2026-10-07.
The initial integration selected 0.7.13.
The multimodal update requires these changes to that dependency set:

.. list-table:: Deep Agents multimodal dependency update
   :header-rows: 1
   :widths: 45 20 20

   * - Package
     - Before
     - After
   * - ``deepagents``
     - 0.7.13
     - 0.7.23
   * - ``langchain``
     - 1.4.0
     - 1.4.4
   * - ``langchain-core``
     - 1.6.3
     - 1.6.9
   * - ``langchain-anthropic``
     - 1.7.2
     - 1.7.5
   * - ``langsmith``
     - 0.12.4
     - 0.14.7
   * - ``fastmcp`` and ``fastmcp-slim``
     - 4.0.3
     - 4.0.11

Deep Agents 0.7.23 requires LangChain 1.4.3, Core 1.6.7, Anthropic integration
1.7.5, Google integration 4.4.0, and LangSmith 0.14.4 or later.
The toolkit's existing base LangChain and Core ranges permit these versions.
Their declared minimums remain unchanged for installations without Deep Agents.
The ``langsmith`` extra now requires ``>=0.14.4,<0.15``.
Its earlier ``<0.13`` limit prevented the combined extras from resolving.
FastMCP 4.0.11 matches the new native MCP integration requirement.
LangGraph remains 1.2.11. OpenAI SDK remains 3.13.0.

LangSmith 0.14.7 still accepts legacy feedback identified by ``run_id``.
The SDK warns about that path. A SmithDB-only deployment requires a
``session_id`` that identifies its LangSmith project UUID.
Pass that value through feedback ``kwargs`` when the server requires it.
It is not the toolkit's public ``thread_id``.

Deep Agents requires the provider packages even when an application selects
another model. The initial integration already upgraded Anthropic SDK to
1.5.0, Google integration to 4.4.0, and Google SDK to 2.23.0.
The base package does not require Deep Agents. The ``all`` extra includes it.
The backend Docker build enables it only with ``INSTALL_DEEPAGENTS=true``.

Version 0.5.3 was the last release that fit all previous optional ranges.
Version 0.7 requires an explicit ``StoreBackend`` namespace and removes the
default planning middleware. The toolkit adds planning explicitly.
Its default ``StateBackend`` keeps virtual files in thread state.

The update includes these upstream changes:

* Version 0.7.19 checks content support against the active model.
  It retries a rejected model call once after removing newly read unsupported
  file content. It does not broadly retry remote tool writes.
* Version 0.7.21 adds optional binary offloading to external backends.
  The toolkit keeps it disabled by default.
* Version 0.7.22 loads skill tools when required and fixes stale file encoding.
  The combined updates include broader native document inputs for OpenAI
  Responses.
* Version 0.7.23 adds pinned skills and fixes each subagent's skill middleware.

The toolkit exposes native backend and store options in ``build_graph``.
It adds document upload types and preserves filenames.
Enable Responses only on a compatible model or gateway.
Binary offloading needs durable storage outside checkpoint state.
See :doc:`deepagents` for configuration, memory boundaries, and retention limits.

Sources: `Deep Agents 0.7.23 metadata
<https://pypi.org/pypi/deepagents/0.7.23/json>`_,
`Deep Agents 0.7 release notes
<https://github.com/langchain-ai/deepagents/releases/tag/deepagents%3D%3D0.7.0>`_,
`Deep Agents 0.7.19 release
<https://github.com/langchain-ai/deepagents/releases/tag/deepagents%3D%3D0.7.19>`_,
`Deep Agents 0.7.21 release
<https://github.com/langchain-ai/deepagents/releases/tag/deepagents%3D%3D0.7.21>`_,
`Deep Agents 0.7.22 release
<https://github.com/langchain-ai/deepagents/releases/tag/deepagents%3D%3D0.7.22>`_,
and `Deep Agents 0.7.23 release
<https://github.com/langchain-ai/deepagents/releases/tag/deepagents%3D%3D0.7.23>`_.

Langfuse
~~~~~~~~

The Python SDK version and the Langfuse server version are separate.
The ``langfuse`` extra now permits SDK versions from 2.60.10 to below 6.
Published extras select one SDK generation: ``langfuse-v2`` requires
``>=2.60.10,<3``, ``langfuse-v3`` requires ``>=3.15.0,<4``, ``langfuse-v4``
requires ``>=4.18.0,<5``, and ``langfuse-v5`` requires ``>=5.0.4,<6``. Use only one selector. The generic ``langfuse`` extra
can combine with any one selector. ``all-observability`` keeps its existing
generic Langfuse and LangSmith dependencies.
The default lockfile now selects SDK 5.0.4. SDK v5 needs server 4.59.0 or later.
The current SDK matrix covers 2.60.10, 3.15.0, 4.18.0, and 5.0.4.
See :doc:`langfuse_compatibility` to select the SDK for each server.

The initial SDK review included these fixes:

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
``--extra all``. The old flag selects the four incompatible SDK selectors.
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

Use ``langfuse-v4`` for server v3 or an older server v4 deployment.
Use ``langfuse-v5`` for server 4.59.0 or later. Add an exact SDK pin when a specific patch
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
CI and the ``uv-lock`` pre-commit hook now use uv 0.13.6.
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
