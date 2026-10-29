"""Compare synthetic legacy records before and after a local Langfuse upgrade."""

import json
from pathlib import Path
from urllib.parse import quote

import fire
import httpx
import rootutils
from dotenv import dotenv_values


_TRACE_FIELDS = ("id", "name", "userId", "sessionId", "input", "output", "timestamp")
_OBSERVATION_FIELDS = (
    "id",
    "traceId",
    "parentObservationId",
    "type",
    "name",
    "model",
    "input",
    "output",
    "promptName",
    "promptVersion",
    "startTime",
    "endTime",
)
_SCORE_FIELDS = ("id", "traceId", "name", "value", "observationId")
_PROMPT_FIELDS = ("name", "version", "type", "prompt")


def _test_path(value: str) -> Path:
    path = Path(value).resolve()
    data = rootutils.find_root(search_from=__file__, indicator=".project-root") / "data"
    if not path.is_relative_to(data):
        raise ValueError("Test environment and manifest paths must be inside the repository data directory.")
    return path


def _client(env_file: str) -> httpx.Client:
    values = dotenv_values(_test_path(env_file))
    names = ("BASE_URL", "PUBLIC_KEY", "SECRET_KEY")
    if any(not values.get(f"LAT_TEST_LANGFUSE_{name}") for name in names):
        raise ValueError("The test environment needs LAT_TEST_LANGFUSE_BASE_URL, PUBLIC_KEY, and SECRET_KEY.")
    base_url = values["LAT_TEST_LANGFUSE_BASE_URL"]
    url = httpx.URL(base_url)
    if url.host not in {"127.0.0.1", "localhost", "::1"} or url.username or url.password:
        raise ValueError("This helper accepts only a local disposable Langfuse server without URL credentials.")
    return httpx.Client(
        base_url=base_url.rstrip("/") + "/",
        auth=(values["LAT_TEST_LANGFUSE_PUBLIC_KEY"], values["LAT_TEST_LANGFUSE_SECRET_KEY"]),
        timeout=10,
        trust_env=False,
    )


def _get(client: httpx.Client, path: str, **params):
    response = client.get(path, params=params)
    response.raise_for_status()
    return response.json()


def _is_test_trace(trace: dict) -> bool:
    user, session = trace.get("userId") or "", trace.get("sessionId") or ""
    return (
        user.startswith("lat-test-")
        and user.endswith("-user")
        and session.startswith("lat-test-")
        and session.endswith("-thread")
        and trace.get("name") in {"lat-process-agent", "LangGraph"}
    )


def _capture(client: httpx.Client, trace_id: str) -> dict:
    trace = _get(client, "api/public/traces/" + quote(trace_id, safe=""))
    if not _is_test_trace(trace):
        raise ValueError("Only synthetic API process traces with lat-test- identities can be captured.")
    observations = {item["id"]: {key: item.get(key) for key in _OBSERVATION_FIELDS} for item in trace["observations"]}
    scores = [
        {key: item.get(key) for key in _SCORE_FIELDS}
        for item in trace["scores"]
        if item["name"] == "lat-test-correctness"
    ]
    if len(scores) != 1 or scores[0]["value"] != 1:
        raise RuntimeError("The synthetic trace must have exactly one completed correctness score.")
    prompt_name = trace["userId"].removesuffix("-user") + "-prompt"
    prompts = {}
    for version in (1, 2):
        prompt = _get(client, "api/public/v2/prompts/" + quote(prompt_name, safe=""), version=version)
        prompts[str(version)] = {key: prompt.get(key) for key in _PROMPT_FIELDS}
    return {
        "trace": {key: trace.get(key) for key in _TRACE_FIELDS},
        "observations": observations,
        "scores": sorted(scores, key=lambda item: item["id"]),
        "prompts": prompts,
    }


def snapshot(env_file: str, manifest_path: str, expected_traces: int = 12) -> None:
    """Save only the synthetic process records in a local test project."""
    if expected_traces < 1:
        raise ValueError("expected_traces must be positive.")
    path = _test_path(manifest_path)
    with _client(env_file) as client:
        page = _get(client, "api/public/traces", limit=100)
        if page.get("meta", {}).get("totalItems", 0) > 100:
            raise ValueError("Use a fresh disposable project with at most 100 traces.")
        records = {trace["id"]: _capture(client, trace["id"]) for trace in page["data"] if _is_test_trace(trace)}
    if len(records) != expected_traces:
        raise RuntimeError(f"Expected {expected_traces} synthetic process traces. Found {len(records)}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    path.chmod(0o600)
    print(f"Saved {len(records)} complete synthetic process traces.")


def verify(env_file: str, manifest_path: str, expected_server_version: str = "4.71.0") -> None:
    """Require unchanged IDs and payloads through the legacy APIs after upgrade."""
    records = json.loads(_test_path(manifest_path).read_text())
    if not records or any(not _is_test_trace(item["trace"]) for item in records.values()):
        raise ValueError("The manifest must contain only synthetic process traces.")
    with _client(env_file) as client:
        if _get(client, "api/public/health")["version"] != expected_server_version:
            raise RuntimeError("The server version differs from the selected upgrade target.")
        for trace_id, expected in records.items():
            if _capture(client, trace_id) != expected:
                raise RuntimeError(f"The stored trace, observations, score, or prompt versions changed for {trace_id}.")
    print(f"Verified {len(records)} unchanged synthetic process traces on server {expected_server_version}.")


if __name__ == "__main__":
    fire.Fire({"snapshot": snapshot, "verify": verify})
