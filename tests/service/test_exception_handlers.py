import os
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from langgraph_agent_toolkit.helper import exceptions as exc
from langgraph_agent_toolkit.service.exception_handlers import register_exception_handlers


def _client_raising(exception):
    """Build a tiny app whose single route raises `exception`, with toolkit handlers registered."""
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/boom")
    async def boom():
        raise exception

    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize(
    "exception, expected_status, expected_error_code",
    [
        (exc.AuthenticationError("nope"), 401, None),
        (exc.AuthorizationError("nope"), 403, None),
        (exc.ValidationError("bad"), 400, None),
        (exc.InputValidationError("bad input"), 422, None),
        (exc.UnsupportedMessageTypeError("weird"), 422, "UNSUPPORTED_MESSAGE_TYPE"),
        (exc.ModelNotFoundError("gpt-x"), 404, "MODEL_NOT_FOUND"),
        (exc.ModelConfigurationError("bad cfg"), 400, None),
        (exc.ToolNotFoundError("hammer"), 404, "TOOL_NOT_FOUND"),
        (exc.ToolExecutionError("hammer"), 500, "TOOL_EXECUTION_FAILED"),
        (exc.RateLimitError("api", 10), 429, "RATE_LIMIT_EXCEEDED"),
        (exc.ServiceUnavailableError("db"), 503, "SERVICE_UNAVAILABLE"),
        (exc.FeedbackError("run-1", "record"), 500, "FEEDBACK_FAILED"),
        (exc.AgentToolkitError("generic"), 500, None),
    ],
)
def test_toolkit_exception_status_map(exception, expected_status, expected_error_code):
    """Each toolkit exception maps to its documented HTTP status (and error_code when set)."""
    resp = _client_raising(exception).get("/boom")

    assert resp.status_code == expected_status
    body = resp.json()
    assert "detail" in body
    if expected_error_code:
        assert body["error_code"] == expected_error_code


def test_base_error_includes_error_type():
    """The catch-all AgentToolkitError handler reports the concrete class name."""
    resp = _client_raising(exc.AgentToolkitError("generic")).get("/boom")
    assert resp.json()["error_type"] == "AgentToolkitError"


def test_plain_value_error_maps_to_400():
    resp = _client_raising(ValueError("just a value error")).get("/boom")
    assert resp.status_code == 400
    assert "just a value error" in resp.json()["detail"]


def test_unexpected_exception_maps_to_500_with_type():
    resp = _client_raising(RuntimeError("kaboom")).get("/boom")
    assert resp.status_code == 500
    body = resp.json()
    assert body["error_type"] == "RuntimeError"
    assert "kaboom" in body["detail"]


@pytest.mark.parametrize("env_mode, expect_traceback", [("development", True), ("production", False)])
def test_traceback_gated_by_env_mode(env_mode, expect_traceback):
    """Tracebacks are included only when ENV_MODE != production (read at registration time)."""
    with patch.dict(os.environ, {"ENV_MODE": env_mode}):
        client = _client_raising(exc.ToolExecutionError("hammer"))
        resp = client.get("/boom")

    assert resp.status_code == 500
    assert ("traceback" in resp.json()) is expect_traceback
