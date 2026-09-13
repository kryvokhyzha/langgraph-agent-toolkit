API load tests
==============

Use ``scripts/load_test.py`` to measure a real local API under load. The script
starts separate Uvicorn processes for the toolkit and an OpenAI protocol
simulator. It uses the real OpenAI SDK, the selected HTTP transport, and the
configured checkpointer. It disables environment files and external tracing.
It does not call a paid model provider.

These tests measure local API behavior. They do not measure provider quotas,
internet latency, a production proxy, or live Langfuse ingestion. Run a separate
deployment test with those components before you set production capacity.

Run the tests
-------------

Run the harness on Linux or macOS. It uses POSIX process groups and signals.
The ``ps`` command is required for worker ownership checks. Install ``lsof``
to include file-descriptor counts. Missing resource measurements are omitted.

Install the locked dependencies from a source checkout:

.. code-block:: bash

   uv sync --extra all --frozen --no-install-project
   uv run --no-sync python scripts/load_test.py --quick=True

The quick run checks the harness with SQLite. A full run adds higher concurrency,
constant arrival traffic, provider faults, slow readers, and history checks:

.. code-block:: bash

   uv run --no-sync python scripts/load_test.py \
     --duration=10 --soak_seconds=120 --output=data/load-test

To include PostgreSQL and two workers, use a disposable local database:

.. code-block:: bash

   docker run --detach --name lat-load-postgres \
     --cpus=2 --memory=512m --publish=127.0.0.1:18516:5432 \
     --env POSTGRES_USER=lat_load --env POSTGRES_DB=lat_load \
     --env POSTGRES_HOST_AUTH_METHOD=trust postgres:16

   LAT_LOAD_TEST_DATABASE=yes uv run --no-sync python scripts/load_test.py \
     --postgres_dsn=postgresql://lat_load@127.0.0.1:18516/lat_load \
     --duration=10 --soak_seconds=120 --output=data/load-test

   docker rm --force --volumes lat-load-postgres

Wait until PostgreSQL accepts connections before starting the test. The script
requires both ``LAT_LOAD_TEST_DATABASE=yes`` and a literal loopback database
address. It creates a separate schema for each configuration and removes it
after the service stops. It terminates only sessions with that configuration's
unique application name. It binds every test HTTP service to loopback.

Use ``--configuration_names=postgres-httpx-2`` to select one configuration.
Use ``--phase_names=worker-kill,soak`` to select measured phases. Warm-up still
runs. PostgreSQL selections require ``--postgres_dsn``. Use a new output
directory for each run. ``results.json`` and process logs are saved there.

The default ``--timeout_profile=normal`` allows more time for API and database
operations. Use ``--timeout_profile=aggressive`` to repeat the earlier tests
with short failure deadlines:

.. code-block:: bash

   uv run --no-sync python scripts/load_test.py \
     --timeout_profile=aggressive --output=data/load-test-aggressive

Both profiles are explicit test configurations. Their values differ from the
package's production defaults. Compare runs with the same profile unless the
timeout difference is the subject of the test.

The default ``--client_pool=slots`` gives each active load slot its own persistent
HTTPX client. A bounded FIFO queue loans these clients to requests. Each client
has one connection. All clients share one SSL context. This avoids contention
inside a single HTTPX connection pool. Use ``--client_pool=shared`` only when
you want to investigate that pool behavior. The load client still counts queue
waiting in request latency and the request deadline.

The load clients expire idle connections after one second, below Uvicorn's
five-second server keepalive. This margin reduces races when an idle connection
closes just as the client reuses it. Use ``--client_keepalive_expiry=5`` to
investigate equal client/server idle deadlines. This setting applies to the
load driver, not the model transport. Connection failures are never retried.

The phase duration is limited to 1–60 seconds. The soak duration is limited to
1–600 seconds. Each phase also has a request limit. A closed-loop phase can
reach that limit before its time limit when the server rejects requests fast.
The JSON report includes the actual send window and completed request count.

Test conditions
---------------

The simulator delays the first stream token by 50 ms. It sends four more chunks
at 10 ms intervals. Its nonstream response delay is 100 ms. The test graph uses
upstream streaming by default for both invoke and stream API routes.

Each worker permits eight active requests. Only ``postgres-queue-1`` permits
admission waiters: up to eight requests can wait for up to 250 ms. The other
configurations have no admission waiters. These limits apply to both timeout
profiles.

.. list-table:: Timeout profiles, in seconds
   :header-rows: 1
   :widths: 60 20 20

   * - Setting
     - ``normal``
     - ``aggressive``
   * - ``REQUEST_TIMEOUT``
     - 15
     - 5
   * - ``THREAD_QUEUE_TIMEOUT``
     - 10
     - 4
   * - ``POSTGRES_POOL_TIMEOUT``
     - 10
     - 2
   * - ``POSTGRES_CONNECT_TIMEOUT``
     - 10
     - 2
   * - ``POSTGRES_HEALTH_CHECK_TIMEOUT``
     - 5
     - 0.5
   * - ``THREAD_LOCK_HEARTBEAT_TIMEOUT``
     - 5
     - 0.5
   * - Load-client request deadline
     - 20
     - 10

The response send deadline remains one second in both profiles. Provider
connection and pool waits remain one second. The provider read timeout remains
two seconds. The SDK permits one retry. Conversation-lock heartbeat checks run
every 0.2 seconds; the table gives the timeout for each check.
The Uvicorn worker heartbeat allowance is ten seconds. Worker replacement tests
allow time for both failure detection and fresh worker startup.

Warm-up requires a complete, correct agent response from every worker PID.
The client checks final message identity, thread identity, and run metadata
before it counts a response. The script also requires a managed transport pool
in each worker and checks that work has drained. It records warm-up attempts,
successful response counts by PID, and warm-up duration separately.

After the worker replacement phase, the script repeats this verification before
the steady-state soak. A healthy PID or an initialized transport pool alone does
not complete warm-up. Do not compare a cold worker with a warm worker.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Phase
     - Purpose
   * - Invoke concurrency sweep
     - Compare one request, full configured capacity, and overload.
   * - Nonstream upstream
     - Compare complete provider responses with upstream streaming. Check
       actual TCP connection counts for each HTTP transport.
   * - SSE and JSON Lines
     - Check stream framing, first data, final message identity, and cleanup.
   * - Constant arrival traffic
     - Offer 128 requests per second. Record unsent arrivals and scheduler lag.
   * - Shared conversation
     - Send 40 turns with four concurrent callers. Read history and check each
       human message and its final reply. Cross-worker ordering is not FIFO.
   * - Client disconnects
     - Close after the first token. Check that API and provider work drains.
   * - Slow readers
     - Stop consuming response bodies with small receive buffers. Use large
       streams to exercise socket backpressure and the response send deadline.
   * - Provider faults
     - Inject rate limits, stalled responses, and dropped response connections.
       Restore the provider and check recovery.
   * - Database faults
     - Hold a checkpoint table lock for two seconds. Terminate tagged lock
       and saver connections during traffic. Check bounded failures and recovery.
   * - Worker replacement
     - Kill one owned worker during traffic. Check that the supervisor starts
       a replacement and that both workers return complete, correct replies.
       Requests affected by the failure can lose their connection or return
       503/504. They are not replayed automatically.
   * - Soak
     - Offer 80 JSON Lines requests per second to two PostgreSQL workers.
       Compare tasks, memory, file descriptors, and database sessions.

Read the results
----------------

Successful goodput counts complete, correct responses per second. Successful
latency excludes overload rejections and failed requests. A stream with HTTP
200 and an error event is a failed stream. Intentional disconnects have their
own count. The client does not retry API requests.

The script fails when responses have incorrect identities, malformed streams,
unexpected errors, incomplete history, or retained work. It also fails when an
injected fault was not observed or a killed worker was not replaced. Expected
overload rejections are reported separately. Passing these checks does not
establish a throughput or latency service-level objective.

Constant arrival tests report latency from the scheduled arrival as well as
from the actual send time. They cap in-flight requests and count dropped
arrivals. This prevents a slow load generator from silently reducing the
offered rate. Check scheduler lag before drawing a capacity conclusion.

The report also contains HTTPcore trace timing and driver event-loop lag.
Up to five exception examples record classes, numeric error codes, and transport
stages without messages, headers, URLs, or credentials.
Compare request-to-send time with send-to-response time. Pre-send time includes
client-pool waiting, connection setup, and request encoding. It is not a direct
measurement of server work. Local diagnostics found delays in the shared HTTPX
pool before requests reached the API. The default slot pool removes that shared
pool contention. Use the optional shared mode to reproduce the comparison.
Check these driver measurements before attributing client timeouts to the API.

HTTP 503 under overload is expected when bounded capacity is full. Compare it
with goodput and successful latency. Persistent HTTP connections can distribute
unevenly between Uvicorn workers. A total concurrency of 16 does not guarantee
eight requests reach each worker. Streaming cleanup can briefly overlap with
the next request from the same client. Queue capacity can absorb these bursts.

Samples include worker activity, queue depth, event-loop lag, current resident
memory, process CPU averages, and model connection counts. File descriptors
are counted before and after each phase. The ``ps`` CPU value is a process
average. It is not an instantaneous CPU measurement. Health requests run
during load. Readiness can fail if a database pool is exhausted, while liveness
continues to pass.
Provider connection counts are observed distinct client address/port pairs.
Long runs can reuse source ports, so this is not an exact lifetime connection
counter. The recovery duration starts after the phase's requests finish. It is
a drain check, not the duration of a provider or database outage.

Recovery requires zero active requests, zero waiters, zero stalled cleanups,
zero active provider calls, and successful readiness and liveness checks.
Compare post-drain task and connection counts with the baseline. A short soak
can expose growth. It cannot prove the absence of a slow memory leak.

In the tested OpenAI 3.13.0 SDK, HTTPX streaming stops at ``[DONE]`` before
HTTP EOF. The HTTPX connection then closes. Reusing a managed client does not
prove TCP reuse on this path. The local tests compare streamed and nonstream
calls and record distinct provider connections. The aiohttp transport reused
TCP connections for these streams. Recheck this behavior after SDK upgrades.

Run the load-generator checks separately:

.. code-block:: bash

   uv run --no-sync pytest tests/load --no-cov

See :doc:`load_test_results` for the dated measurements and investigated failures,
:doc:`reliability` for production settings, and :doc:`testing` for the functional,
database, worker, and telemetry test layers.
