# Reliability checks by failure

Read only the section that applies to the task. Use disposable resources for
fault injection. Do not terminate a production database session to test
recovery.

## Checkpoints and history

1. Check the graph's actual saver after service startup. A database environment
   variable alone does not prove that the graph uses that backend.
2. Confirm that each request uses the same owner, agent, and public thread ID.
   Keep public IDs at the API boundary. Do not pass an already scoped storage
   key to an API that scopes it again.
3. Check a completed run, then read history after a worker restart. For SQLite,
   use the same persistent file. SQLite has no toolkit long-term store.
4. Inspect errors from both checkpoint and pending-write operations. Preserve
   those errors instead of converting them into a successful response.

With `CHECKPOINT_DURABILITY=async`, saving can overlap the next super-step. A
hard worker failure can lose pending progress. A successful run still waits for
pending saves. With `exit`, intermediate progress has a larger loss window.
Neither mode changes an in-memory saver into durable storage.

History pagination limits the response. It still reads the latest saved state.
Functional API graphs can save history under `__previous__`. A nested graph's
internal messages are not automatically the public conversation history.
Preserve `get_graph_conversation` when changing this behavior.

Thread listing reads metadata from initial root checkpoints with `step=0`.
Pruning that checkpoint can remove a conversation from the list while history
remains available by its public ID. Missing identity metadata can also prevent
discovery. Do not infer ownership from raw storage IDs or scan all checkpoint
payloads as a fallback.

## Concurrent requests and database connections

The request queue and conversation queue have different purposes. A request can
occupy an admission slot while it waits for its conversation lock. Repeated
submissions for one thread are serialized; they are not deduplicated. Changing
the thread ID to avoid a queue creates a different conversation.

Calculate the maximum connection budget across all replicas and workers. Include
checkpoint pools, lock pools, separately opened stores, and other applications.
Check provider quotas before increasing active request limits.

For PostgreSQL, the lock-owning session also performs checkpoint operations
during a run. Replacing it with an unrelated connection can permit stale writes
after lock loss. Preserve the coordinated saver and its scope checks.

For SQLite, respect the shared connection lock. Do not hold a saver iterator
open while requesting another operation that needs the same lock. Preserve
rollback and cancellation cleanup when changing transaction code.

Compare pool checkout timeouts with SQL statement and lock timeouts. Their units
differ. Read `core/_base_settings.py` instead of guessing units. Do not retry an
entire agent run merely because a connection failed.

## Model transport and stream completion

Use `CompletionModelFactory` and the managed clients when changing a toolkit
model path. Creating clients per token, tool call, or request prevents
connection reuse. Caller-supplied clients keep their own transport and
lifecycle.

The API image selects aiohttp for managed asynchronous OpenAI and Azure calls.
Python installations default to HTTPX. Check effective overrides before changing
the transport. Synchronous calls still use HTTPX. Transport timeout and pool
controls do not all map to the same aiohttp settings.

Record time to first token, total duration, pool waits, provider errors, and
cleanup time. Separate provider rate limits from local connection exhaustion. Do
not add an outer retry that repeats tool side effects after partial output.

HTTP 200 does not prove stream success. SSE needs `[DONE]` with no error event.
JSONL uses the response body end and has no separate completion marker. Received
tokens can precede the final checkpoint write.

## Worker recovery

Distinguish a process exit, a blocked event loop, a failed readiness check, and
a database outage. They need different recovery actions.

Use a supervisor to replace failed workers. A Uvicorn heartbeat thread can
respond while the event loop is blocked. Test an HTTP health probe too. Docker
health status alone does not restart a container. Preserve the curl probes in
the production images when changing their build.

Check startup time and shutdown cleanup under the configured worker count. Use
the API process tests for persisted history and interrupted-run recovery. Use
`tests/service/test_worker_recovery.py --run-e2e` for supervisor behavior.
Checkpoint recovery does not automatically resume a lost HTTP request.
