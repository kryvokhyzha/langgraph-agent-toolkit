Local pressure-test results: 2026-09-12
=======================================

The final two-minute soak passed: all 9,600 arrivals were sent, 4,618 returned
complete correct replies, and 4,982 received explicit overload rejections. There
were no transport or stream errors and no retained locks. The broader 57-phase
matrix had three failed scenarios, investigated below; its failures were retained.

The tests exercised real toolkit HTTP routes, OpenAI SDK clients, SQLite and
PostgreSQL persistence, and worker supervision. They exposed a provider error
translation bug and slow worker replacement, plus load-generator and
test-configuration limits. These were addressed and verified with focused reruns.

Environment and scope
---------------------

The local machine ran macOS 26.4 on ARM with eight logical CPUs and Python
3.12.0. PostgreSQL 16.15 ran in a disposable container limited to two CPUs and
512 MiB. Background host load was high: observed load averages were roughly
100–147. The API, load driver, simulator, and database shared this machine.
These numbers describe recovery under local pressure, not production capacity.
They do not establish a performance ranking between HTTPX and aiohttp.

The dependency set included OpenAI 3.13.0, langchain-openai 1.6.2,
langchain-core 1.6.3, LangGraph 1.2.11, HTTPX 0.28.1, httpx2 2.12.0,
aiohttp 3.14.3, Uvicorn 0.49.0, psycopg 3.2.13, and psycopg-pool 3.3.1.
The simulated model used 50 ms to first token and four subsequent 10 ms gaps,
or a 100 ms nonstream response. No paid model or external telemetry was used.

Five configurations covered SQLite with one HTTPX worker, PostgreSQL with one
HTTPX worker, one aiohttp worker, two HTTPX workers, and one HTTPX worker with
eight admission waiters. Every worker allowed eight active agent requests.
Tests covered invoke, SSE, JSON Lines, upstream streaming and nonstreaming,
constant arrivals, concurrent conversation updates, client disconnects,
slow readers, provider faults, database locks and lost connections, worker
death, and a two-minute soak.

Findings and changes
--------------------

Provider disconnects after response headers escaped the usual SDK error
translation and returned HTTP 500. The OpenAI and Azure model wrappers now
normalize interrupted transport streams. Provider rate limits return 429,
connection failures return 503, and timeouts return 504 when an HTTP response
has not started. Failures after stream headers remain stream errors; HTTP 200
alone never counts as success. Received tokens are not retried.

Uvicorn did replace killed workers. A heartbeat wait could delay detection,
and fresh worker startup added more time. With the former 30-second allowance,
one observed replacement took about 40 seconds. The toolkit allowance is now
10 seconds and remains configurable. This is not a guaranteed recovery bound.
The complete matrix observed replacement after 9.29 seconds. Requests served
by the killed worker can fail; they are not replayed automatically.

The initial load driver used one shared HTTPX connection pool. Its own pre-send
wait reached 5.5 seconds at p95, while send-to-response p95 was 345 ms. The
bounded driver now loans one persistent client per active slot. A comparison
sent all 1,280 scheduled requests at 128 arrivals/second without client timeouts
or dropped arrivals. Pre-send p95 fell to 27 ms. This corrected the measurement
method rather than increasing server capacity.

In the tested OpenAI SDK, HTTPX streamed responses stopped at ``[DONE]`` before
HTTP EOF and closed the TCP connection. The aiohttp path reused connections.
Managed client reuse and TCP reuse are different measurements. The harness
records observed provider connections for both streaming modes; it does not
patch SDK internals. Recheck this behavior after dependency upgrades.

Complete matrix and timeout failures
------------------------------------

The 57-phase matrix completed 53,679 attempted requests, plus four separate
slow-reader connections. It recorded 15,577 complete correct replies, 114
intentional first-token disconnects, and 256 arrivals the driver could not send
within its bounds. There were no request-ID, metadata, or framing mismatches.
There were 37,866 overload HTTP 503 responses. This was deliberate overload;
rejections are excluded from successful goodput and latency percentiles.

All 57 phases drained to zero active requests, waiters, stalled cleanups, and
active provider calls. Three phases failed the stricter correctness verdict:

* SQLite's 40-turn concurrent conversation test completed 39 turns and rejected
  one with HTTP 409 when its four-second queue deadline expired. Stored history
  contained every successful turn. The original verdict mislabeled this as
  lost history; the checker now distinguishes rejected work from missing data.
* One PostgreSQL JSON Lines phase had two stream failures during pool pressure.
* The two-worker soak had two stream failures, four transport failures, and one
  HTTP 504. A replacement worker was still cold at the start of that phase;
  one observed event-loop delay reached 5.8 seconds.

These runs used a five-second request deadline, two-second PostgreSQL pool and
connect waits, and 500 ms health and conversation-heartbeat deadlines. Those
were aggressive test overrides. A subsequent run increased most deadlines and
passed all six selected JSON Lines and concurrent-history phases, but its soak
still had two stream errors. The logs identified the remaining 500 ms
conversation-heartbeat deadline. This was shorter than observed event-loop
delays above one second. The final normal profile uses the package's five-second
heartbeat timeout. It also requires a complete correct agent reply from every
worker, including replacements, before steady-state measurements.

The next normal-profile run removed those stream errors and observed a new
worker after 5.24 seconds, but still failed its soak: 116 requests raised a
client read error before response headers. All 3,295 successful replies were
complete and correct. The trace recorded 372 TCP connections: 256 initial client
slots plus 116 reconnects. Both the driver and server used a five-second idle
expiry, leaving no margin for delayed connection reuse. This evidence was kept
as another failed run while the connection timing was investigated. A bounded
TCP reproduction confirmed this race with both Uvicorn HTTP parsers: immediate
connection reuse returned the expected 503, but delaying a reused request after
pool selection until the server idle timeout expired raised a read error before
headers. The normal-profile soak logs contained no corresponding application
errors. The probe used a 120 ms server idle expiry, a five-second client expiry,
and a 200 ms pause after connection selection. This confirms a possible cause,
not the cause of every individual soak failure. The harness now expires idle
client connections after one second, below Uvicorn's five-second allowance,
and retains bounded exception-cause and transport-stage diagnostics. It does
not retry agent requests.

Final soak verification
-----------------------

The subsequent two-minute PostgreSQL soak passed with two warm workers, the
normal timeout profile, and the one-second client idle expiry. All 9,600
scheduled arrivals were sent. There were no client transport errors, stream
errors, identity mismatches, malformed responses, or dropped arrivals.

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Measurement
     - Observed result
   * - Offered rate / duration
     - 80 requests/second for 120 seconds
   * - Complete correct replies
     - 4,618
   * - Explicit overload rejections
     - 4,982 HTTP 503 responses
   * - Successful goodput
     - 38.44 replies/second
   * - Successful response p95 / p99
     - 893 ms / 1,518 ms
   * - First stream data p95
     - 553 ms
   * - Driver scheduler lag p95
     - 136 ms
   * - Client transport or stream errors
     - 0
   * - PostgreSQL deadlock counter increase
     - 0

Both workers returned to 12 asyncio tasks, one managed transport pool, zero
active requests, zero waiters, and zero stalled cleanups. Each retained one sync
and one async model client. All 256 load clients closed. Open file descriptors
grew from 34 and 30 to 38 per worker while pooled database sessions grew from
8 to 20 in total. No advisory locks or idle transactions remained. Sampled peak
resident memory was about 165 MiB per worker; post-drain resident memory was
below each worker's baseline. These short observations show bounded cleanup in
this run, not proof that no long-term leak is possible.

This final run selected only the soak phase. Worker replacement was verified in
the preceding run, where the new process appeared after 5.24 seconds and then
completed a correct agent request. The earlier full matrix still has its three
recorded failures. The focused JSON Lines and history reruns, heartbeat diagnosis,
and idle-expiry comparison explain the follow-up verification; no entire matrix
was silently reclassified as passing.

Database and cleanup evidence
-----------------------------

A two-second checkpoint table lock and termination of ten tagged toolkit
connections both recovered. No PostgreSQL deadlock counter increased. After
phases drained, no conversation advisory locks or idle transactions remained.
Pooled idle sessions remained available for reuse. The largest observed session
counts at phase boundaries were 12 for one HTTPX worker, 11 for one aiohttp
worker, 20 for two HTTPX workers, and 11 for the queue configuration. These are
boundary observations, not continuous peak connection counts.

The original five history checks retained all 199 successful turns and their
matching replies. Follow-up checks completed all 40 turns each on SQLite,
one PostgreSQL worker, and two PostgreSQL workers. The slow-reader checks
observed the server release capacity while the client sockets were still held
open. Client disconnects also drained without retained work.

Reproduction and limits
-----------------------

See :doc:`load_testing` for commands, profiles, report fields, and fault controls.
Use a fresh output directory for each run. Keep failed reports beside reruns;
a later pass does not turn an earlier failed run into a pass. The normal and
aggressive profiles are explicit harness settings, not full production presets.

A two-minute soak can expose immediate resource growth. It cannot exclude slow
leaks. These tests do not measure real model quality, external quotas, remote
network failure, proxy behavior, live Langfuse ingestion, MCP tools, or complex
multi-agent graphs under load. Repeat on an isolated deployment with its actual
request mix before choosing throughput targets or increasing admission limits.
