import os
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.testclient import TestClient

from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.service.handler import create_app
from langgraph_agent_toolkit.service.middleware import LoggingMiddleware


def _app_with_logging():
    app = FastAPI()
    app.add_middleware(LoggingMiddleware)

    @app.get("/ok")
    async def ok():
        return {"ok": True}

    @app.get("/redir")
    async def redir():
        return RedirectResponse(url="/ok", status_code=307)

    return app


def test_logging_middleware_logs_request_and_response():
    app = _app_with_logging()
    with patch("langgraph_agent_toolkit.service.middleware.logger") as mock_logger:
        resp = TestClient(app).get("/ok")

    assert resp.status_code == 200
    logged = " ".join(str(c.args[0]) for c in mock_logger.info.call_args_list)
    assert "HTTP Request: GET" in logged
    assert "HTTP Response: GET" in logged


def test_logging_middleware_skips_redirect_response_log():
    """With SKIP_REDIRECTION_LOGGING enabled, 3xx responses are not response-logged (only the request)."""
    with patch.dict(os.environ, {"SKIP_REDIRECTION_LOGGING": "true"}):
        app = _app_with_logging()
        with patch("langgraph_agent_toolkit.service.middleware.logger") as mock_logger:
            TestClient(app).get("/redir", follow_redirects=False)

    logged = [str(c.args[0]) for c in mock_logger.info.call_args_list]
    assert any("HTTP Request" in m for m in logged)
    assert not any("HTTP Response" in m for m in logged)


def test_cors_middleware_added_when_enabled():
    with patch.object(settings, "CORS_ENABLED", True):
        app = create_app()
    assert any(m.cls is CORSMiddleware for m in app.user_middleware)


def test_cors_middleware_absent_when_disabled():
    with patch.object(settings, "CORS_ENABLED", False):
        app = create_app()
    assert not any(m.cls is CORSMiddleware for m in app.user_middleware)
