Langfuse service upgrade
========================

Review date: 2026-10-28.

The reviewed server versions are Langfuse 4.71.0 and the maintained v3 release
3.225.15. The web and worker must use the same exact server version.
SDK versions and server versions are separate. See :doc:`langfuse_compatibility`
before selecting a Python SDK extra.

The `4.71.0 release
<https://github.com/langfuse/langfuse/releases/tag/v4.71.0>`_ includes patched
Node.js packages, authorization fixes, and improved stored message extraction.
The `3.225.15 release
<https://github.com/langfuse/langfuse/releases/tag/v3.225.15>`_ backports
authorization and webhook-secret fixes. Upstream supports v3 security updates
until the end of January 2027.

The previous Compose example used floating ``:3`` tags. Its actual patch
version depends on the last image pull. Record ``/api/public/health`` and the
image digest before an existing deployment upgrade. Use the
`official release list <https://github.com/langfuse/langfuse/releases>`_
to review changes from that recorded version.

Select the Compose server version
---------------------------------

The main Compose file requires ``LANGFUSE_SERVER_VERSION`` explicitly.
The current ``.env.example`` selects 4.71.0 for a new deployment.
It also selects ``LANGFUSE_SDK_MAJOR=5`` for the toolkit API image build.
Compose reads this variable before it loads service ``env_file`` entries.
Set it in the root ``.env`` or the shell. A value only in
``configs/langfuse/.langfuse.env`` does not select the image.
Compose requires this value even for commands that select another service.

For an existing v3 installation, first select:

.. code-block:: text

   LANGFUSE_SERVER_VERSION=3.225.15
   LANGFUSE_SDK_MAJOR=4

Complete the migration checks below before changing that value to 4.71.0.
Do not point a test stack at existing deployment volumes.
Use ``LANGFUSE_SDK_MAJOR=3`` when a client deployment must retain SDK v3.
Rebuild the API image after changing this build option.

Infrastructure and existing data
--------------------------------

Langfuse v4 needs ClickHouse 25.12 or later, PostgreSQL 15 or later, and Redis
7.0 or later. The reviewed stack uses ClickHouse 26.4, PostgreSQL 16, and Redis
7.2. Follow the `official v3-to-v4 guide
<https://langfuse.com/self-hosting/upgrade/upgrade-guides/upgrade-v3-to-v4>`_
for ClickHouse grants, storage planning, and deployment-specific procedures.
Historic backfill can need about three times the current ClickHouse data
volume. Check disk capacity before enabling it.

Use this sequence for existing data:

1. Back up PostgreSQL and ClickHouse. Retain the object-store data and secrets.
   Test the restore procedure.
2. Update the v3 server and finish its required data migrations. Inspect each
   unfinished job and its prerequisites on the background migration page.
3. Update ClickHouse while the server still uses v3. Verify ingestion and reads.
4. Select the v4 images with a migration write mode. Review SDK producers,
   API consumers, evaluators, and exports before using ``events_only``.
5. Complete historic backfill or a retention-based rollover. Verify old and new
   records before removing the migration settings.

The v4 schema update removes superseded tables. An image downgrade alone is
not a rollback. Prefer a controlled restore or the upstream schema rollback
procedure. Records written only in the new format are unavailable to a v3
read path.

Keep ``SALT``, ``ENCRYPTION_KEY``, ``NEXTAUTH_SECRET``, and the database and
object-store credentials. Do not generate replacement secrets during this
server upgrade. Existing records and API keys depend on them.

The fresh 3.225.15 test server has five pending ``20260701_v4_step_*`` jobs.
They are gated v4 migration work. The fresh v4 server also keeps an optional
scratch-table cleanup job pending. A raw count of unfinished rows does not
distinguish these jobs from required v3 data migrations.
Check each job's name, gate, and error state. Finish required v3 data movement
before the schema upgrade. Do not mark migration rows complete manually.

For a first server upgrade that retains v3 ingestion and reads, put these
values in the Langfuse service environment before starting v4:

.. code-block:: text

   LANGFUSE_MIGRATION_V4_WRITE_MODE=legacy
   LANGFUSE_MIGRATION_V4_NATIVE_OTEL_BEHAVIOUR=dual_write
   LANGFUSE_MIGRATION_V4_ALLOW_PREVIEW_OPT_IN=false
   LANGFUSE_BACKGROUND_MIGRATION_V4_ENABLE_HISTORIC_BACKFILL=false

These are temporary migration settings. Fresh v4 deployments use
``events_only`` and ``direct``. The old trace and observation read endpoints
are unavailable in that mode. Use Observations API v2 for trace records.
Python SDK v2 ingestion is rejected. Python SDK v5 needs server 4.59.0 or later.

The worker probe includes ``failIfEventPropagationStuck=true``. This detects
a stalled compatibility propagation job during a dual-write migration.
The Langfuse images include ``wget`` and do not include ``curl``.
Their probes use the supplied executable. Toolkit API and UI images retain
their ``curl`` probes.
Docker health status alone does not restart a running container.
Use an orchestrator liveness policy to replace an unhealthy service.

Keep legacy SDK batching within the worker shutdown budget.
The toolkit defaults are ``LANGFUSE_FLUSH_INTERVAL=5`` seconds and
``OBSERVABILITY_SHUTDOWN_TIMEOUT=10`` seconds. SDK v2, v3, and v4 legacy queues
can wait for their batch interval during a public flush. Leave time for export
and network operations when changing either value.
A graceful stop can flush pending records. A network failure, host failure,
or ``SIGKILL`` can prevent delivery. Checkpoint durability has a separate policy.

The worker now waits for web readiness before it starts. The web process
applies schema migrations. This order avoids worker reads against incomplete
tables during the first startup.

Redis holds pending Langfuse jobs. The Compose configuration mounts a named
``/data`` volume and enables AOF with ``appendfsync everysec``.
Container recreation retains pending Redis data. A host crash can still lose
about one second of recent writes. Keep Redis backups and monitor disk space.
``noeviction`` prevents Redis from deleting queue keys under memory pressure.
New writes can fail when memory is exhausted. Size the service for its load.
The example limits Redis data to 256 MiB inside its 512 MiB container limit.
This leaves space for allocator and AOF overhead. It is a development-stack
baseline. Measure peak memory, fragmentation, and AOF rewrites in production.
Set ``REDIS_AUTH`` once in ``configs/redis/.redis.env``. The server and its
probe now read the same container environment value. No separate root ``.env``
copy of the password is required.
Move any root-only ``REDIS_AUTH`` value into that service environment file.
Keep its existing value during the migration.
Before switching an existing deployment, copy its current effective password
into this service file. Align LiteLLM and other Redis clients with that value.
Do not replace a custom password with the example password.
Drain pending jobs before the first Redis container replacement.
An empty new volume does not copy the old container's in-memory keys.
Back up and restore Redis data when existing queue or cache keys must remain.

Media URLs must be reachable by SDK clients and users. Replace the example's
``localhost:9090`` endpoint with the appropriate address for container clients
or a remote deployment. The v4 server uses the separate internal MinIO endpoint.
The service tests below use text records. They do not verify browser media
uploads or externally signed media URLs.

Run a disposable service test
-----------------------------

The integration fixture creates a separate Compose project and named volumes.
It exposes only the web port, bound to ``127.0.0.1``. It does not load the
application's ``.env`` or the deployment's service configuration.
The helper generates synthetic credentials in the ignored ``data/`` directory.
Its command output does not include those credentials.

From the repository root, run:

.. code-block:: bash

   uv run --no-sync python scripts/ci/langfuse_stack.py prepare \
     --directory=data/langfuse-test-v4 --server_version=4.71.0
   uv run --no-sync python scripts/ci/langfuse_stack.py start \
     --directory=data/langfuse-test-v4

The first start downloads the server and database images. The fixture needs
enough memory for six services. The local verification uses an 8 GiB Docker
memory budget. Run v3 and v4 test stacks sequentially on a small computer.
The main Compose web and worker limits match the tested 2 GiB and 1.5 GiB
limits. Add ``NODE_OPTIONS=--max-old-space-size=1024`` to the Langfuse service
environment when updating an older file. The current example includes it.
The full application stack needs more memory than this isolated Langfuse stack.
These are development baselines. Measure and tune production memory limits.

Load only the generated test variables, then run the explicit live tests:

.. code-block:: bash

   set -a
   source data/langfuse-test-v4/server.env
   set +a
   uv run --no-sync pytest tests/integration/test_langfuse_live.py --run-langfuse -q

Use an environment with a compatible SDK. The test checks the reported server
version and reads stored prompts, observations, tool results, and feedback.
It uses synthetic model responses. It makes no paid model requests.

To test persistence, keep the same project and restart only web and worker:

.. code-block:: bash

   uv run --no-sync python scripts/ci/langfuse_stack.py restart \
     --directory=data/langfuse-test-v4

Read the same record IDs after the restart. Do not delete the volumes between
the write and read phases.

Remove the generated test stack after verification:

.. code-block:: bash

   uv run --no-sync python scripts/ci/langfuse_stack.py stop \
     --directory=data/langfuse-test-v4

The helper removes only its generated test project and volumes. It retains
local logs and test credentials. Do not publish those files.
Use a new directory for another independent project.
To test the maintained v3 server, repeat the prepare command with
``--server_version=3.225.15`` and a separate directory.

These tests use synthetic data. They do not prove that an existing production
database completed its historic backfill. Verify that migration in staging
with a restored backup before replacing a production server.

Verification record
-------------------

The local verification used native arm64 images and synthetic model responses.
SDK 5.0.4 and SDK 4.18.0 each passed four live tests against server 4.71.0.
The tests read prompts, complete native and Deep Agents tool traces, feedback,
and public user and session identities after toolkit API worker shutdown.

After the v4 web and worker restarted, the same eight API process traces,
172 observations, eight scores, and linked prompt revisions remained unchanged.
The check read those existing IDs. It did not write replacement records.

The Redis check used the actual main Compose command with a smaller 4 MiB
test limit. It verified correct and incorrect authentication, a non-root
server process, AOF recovery after container recreation, and explicit write
errors under memory pressure. The earlier marker remained present.
All Redis verification resources were disposable.

SDK 2.60.10, 3.15.0, and 4.18.0 each passed four live tests against server
3.225.15. That test project then upgraded to 4.71.0 in ``legacy`` mode with
``dual_write`` OTel behavior and historic backfill disabled.
The upgrade retained its original keys and volumes. PostgreSQL and cold
ClickHouse backups were taken before the schema update.
All 12 API process traces, 254 observations, 12 scores, and linked prompt
revisions remained unchanged through the legacy read APIs.
This verifies the initial server schema step. It does not verify historic
backfill or a cutover to ``events_only``.

Compare synthetic records during a test upgrade
-----------------------------------------------

The read-only verifier accepts only a localhost test endpoint and paths in
``data/``. It captures only the synthetic API process identities.
Run the API tests for SDK v2, v3, and v4 against one disposable v3 project
before taking its 12-trace snapshot.
Prepare a migration fixture with ``--write_mode=legacy`` so later starts keep
the legacy overlay. Retain its generated keys. Change only its server version
after taking backups. The normal live test suite targets server defaults;
this reader deliberately uses legacy APIs during the initial migration step:

.. code-block:: bash

   uv run --no-sync python scripts/ci/verify_langfuse_legacy_upgrade.py snapshot \
     --env_file=data/langfuse-test-v3/server.env \
     --manifest_path=data/langfuse-test-v3/before-upgrade.json \
     --expected_traces=12

Take backups and perform the controlled compatibility-mode upgrade described
above. Keep the project keys and volumes. Then compare the original IDs:

.. code-block:: bash

   uv run --no-sync python scripts/ci/verify_langfuse_legacy_upgrade.py verify \
     --env_file=data/langfuse-test-v3/server.env \
     --manifest_path=data/langfuse-test-v3/before-upgrade.json \
     --expected_server_version=4.71.0

The check compares trace identities and I/O, model and tool observation data,
parent links, score identities and values, and both prompt revisions.
It does not write records or perform a database migration.
