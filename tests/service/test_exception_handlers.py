from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.helper import exceptions as exc
from langgraph_agent_toolkit.helper.types import EnvironmentMode
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


@pytest.mark.parametrize("env_mode, expose", [(EnvironmentMode.DEVELOPMENT, True), (EnvironmentMode.PRODUCTION, False)])
def test_value_error_detail_gated_by_env_mode(env_mode, expose):
    """Raw ValueError -> 400; its message reaches the client only outside production (no leak in prod)."""
    with patch.object(settings, "ENV_MODE", env_mode):
        resp = _client_raising(ValueError("secret=topsecret")).get("/boom")

    assert resp.status_code == 400
    body = resp.json()
    if expose:
        assert "secret=topsecret" in body["detail"]
        assert "traceback" in body
    else:
        assert body["detail"] == "Invalid request"
        assert "secret=topsecret" not in str(body)
        assert "traceback" not in body


@pytest.mark.parametrize("env_mode, expose", [(EnvironmentMode.DEVELOPMENT, True), (EnvironmentMode.PRODUCTION, False)])
def test_unexpected_exception_detail_gated_by_env_mode(env_mode, expose):
    """Unexpected exception -> 500; the internal type/message reaches the client only outside production."""
    with patch.object(settings, "ENV_MODE", env_mode):
        resp = _client_raising(RuntimeError("password=hunter2 host=internal-db")).get("/boom")

    assert resp.status_code == 500
    body = resp.json()
    if expose:
        assert body["error_type"] == "RuntimeError"
        assert "password=hunter2" in body["detail"]
        assert "traceback" in body
    else:
        assert body == {"detail": "Internal server error"}
        assert "password=hunter2" not in str(body)


@pytest.mark.parametrize(
    "env_mode, expect_traceback", [(EnvironmentMode.DEVELOPMENT, True), (EnvironmentMode.PRODUCTION, False)]
)
def test_toolkit_exception_traceback_gated_by_env_mode(env_mode, expect_traceback):
    """Toolkit exceptions expose their (intentional) detail; only the traceback is gated by env mode."""
    with patch.object(settings, "ENV_MODE", env_mode):
        resp = _client_raising(exc.ToolExecutionError("hammer")).get("/boom")

    assert resp.status_code == 500
    body = resp.json()
    assert "detail" in body  # toolkit detail is intentional and always present
    assert ("traceback" in body) is expect_traceback
