Real LLM verification
=====================

The pressure tests use a local OpenAI protocol simulator. They exercise the real
SDK and transport without provider traffic. The separate live test calls a real
OpenAI model through the toolkit's HTTP API, graph, model factory, and SQLite
checkpointer. It does not replace the model response with a test value.

Run a small live check
----------------------

Install the locked dependencies from a source checkout:

.. code-block:: bash

   uv sync --frozen --no-install-project --extra all

Provide ``LAT_TEST_OPENAI_API_KEY`` through your environment or CI secret manager.
Set ``LAT_TEST_OPENAI_MODEL`` to the model that you want to verify. If your key
is already exported as ``OPENAI_API_KEY``, run:

.. code-block:: bash

   LAT_TEST_OPENAI_API_KEY="$OPENAI_API_KEY" \
   LAT_TEST_OPENAI_MODEL=gpt-5.4 \
   LAT_TEST_OPENAI_MODEL_KWARGS='{"reasoning_effort":"none"}' \
   uv run --no-sync pytest tests/e2e/test_llm_live.py \
     --run-e2e --run-llm --no-cov -x -q \
     --basetemp=data/live-llm-tests

Use a new ``--basetemp`` directory for each report you want to retain. Pytest
clears its selected temporary directory. The test does not load a repository
``.env`` file. Ordinary tests replace ``OPENAI_API_KEY`` with a fake value;
the explicit ``LAT_TEST_OPENAI_API_KEY`` is passed only to the live child service.
Both live flags and both required variables must be present. Normal test runs
skip this test, even when normal provider credentials are configured.

``LAT_TEST_OPENAI_MODEL_KWARGS`` can set only ``reasoning_effort`` and
``temperature``. Use values supported by the selected model. The documented
GPT-5.4 configuration supports ``reasoning_effort=none``; see the
`official model reference <https://developers.openai.com/api/docs/models/gpt-5.4>`_.
The endpoint is fixed to ``https://api.openai.com/v1``. This suite does not test
Azure deployments, third-party proxies, or other providers.

Each of the two transport cases has a six-call budget checked before model
invocation. SDK retries are disabled. Every request uses ``n=1``,
``store=False``, and ``max_completion_tokens=256``. The token cap includes
reasoning tokens as well as visible output; see the
`Chat Completions reference <https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create>`_.
The default suite therefore starts at most twelve model requests. It uses only
short synthetic prompts and a local arithmetic tool. API charges can apply.

What the live test checks
-------------------------

For each of HTTPX and aiohttp, the test runs one journey:

1. Request a real ``add`` tool call with arguments 2 and 5. Execute the local
   tool and request a final model reply. Check the tool ID, arguments, result,
   token usage, and saved history. This uses two model requests.
2. Run SSE and JSON Lines responses concurrently on separate threads. Check
   first tokens, final AI messages, thread/run IDs, token counts, and framing.
   This uses two model requests.
3. Close a longer SSE response after its first token. Wait for model work,
   admission slots, and cleanup to drain. Record whether model cancellation was
   observed. A fast provider can finish before cancellation reaches the model.
4. Send another request on that same thread. Check its actual reply, usage,
   health probes, and resource counters. Total model attempts must remain six.

Each child uses a fresh local SQLite database and a loopback HTTP listener.
External telemetry and external tools are disabled. The test stops its own
service after the journey. ``live-result.json`` in each case directory records
model/transport names, request counts, usage, timings, and cleanup evidence.
It contains no credentials or request headers. A failure keeps the partial report
and stops the run when ``-x`` is used.

Do not infer total billed usage from a cancelled response. Its final usage chunk
may never arrive. Recorded token counts cover the completed model responses.
A passing small run checks real-provider compatibility; it does not establish
production throughput, quota headroom, response quality, or long-term stability.
Use :doc:`load_testing` for controlled pressure and failure injection.

Verification on 2026-09-12
--------------------------

The offline regression suite passed 861 tests with 12 skips and 80.64% combined
line/branch coverage after the usage, refusal, and SSE completion fixes. The live
journey then sent one real request to the configured ``gpt-5.4`` model. OpenAI
rejected the configured credential with HTTP 401 and ``invalid_api_key``. No
model response was generated. The test stopped without retries, with zero active
model calls, admission slots, waiters, or stalled cleanups remaining.

This is a failed live verification, not evidence that either transport passed
against a real model. A valid test credential is required to complete the live
journeys. Replace credentials locally and rerun explicitly; do not retry an
unchanged rejected key. The two-transport live contract remains unverified until
both journeys finish successfully.

The rejected credential also exposed a generic API error and unsafe provider
error logging. The fix now returns HTTP 503 with
``error_code=model_authentication_failed`` from ``/invoke``. Streams return a
fixed error message after headers are sent. Toolkit logs omit provider error
text and tracebacks for authentication failures in all environment modes.
Twelve offline cases check OpenAI and Azure across invoke, SSE, and JSON Lines
in production and development. The final focused suite passed 154 tests, with
the two live transport cases skipped. This fix has not been checked in another
live request.

Offline edge cases
------------------

The deterministic tests cover cases that should not depend on a live provider
returning a particular failure:

* Empty choices and usage-only chunks; Unicode text; output limits.
* Interleaved streamed tool arguments and stable tool-call IDs.
* Refusal-only responses, including streamed refusal fragments.
* Cancellation during an active stream, response closure, and no partial retry.
* SSE EOF without ``[DONE]``, including EOF after a final message.
* Usage preservation through API output, history import, and SQLite reopen.
* Provider authentication failure, safe responses and logs, and no SDK retry.
* Live-test activation, concurrent call budgets, cancellation accounting, and
  rejection of configuration that could bypass request limits.

The public ``ChatMessage.usage_metadata`` field is optional. ``null`` means the
provider did not supply counts, not zero usage. Refusal text is retained in
``response_metadata["refusal"]`` and supplies public content when the provider
content is empty. The client raises ``AgentClientError`` if an SSE stream ends
without its completion marker. A caller can still close its own stream early.
JSON Lines does not use the SSE completion marker.

.. code-block:: bash

   uv run --no-sync pytest tests/core/test_llm_edge_cases.py \
     tests/client/test_stream_completion.py tests/service/test_usage_metadata.py \
     tests/service/test_provider_errors.py tests/core/test_live_llm_guard.py

See :doc:`testing` for the full test layers and :doc:`reliability` for production
connection and timeout settings.
