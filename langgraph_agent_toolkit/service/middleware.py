import os
from http.client import responses

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, DispatchFunction
from starlette.types import ASGIApp

from langgraph_agent_toolkit.helper.logging import logger


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log incoming requests and outgoing responses."""

    def __init__(self, app: ASGIApp, dispatch: DispatchFunction | None = None):
        super().__init__(app=app, dispatch=dispatch)
        self.skip_redirection_logging = os.getenv("SKIP_REDIRECTION_LOGGING", "true").lower() == "true"

    async def dispatch(self, request: Request, call_next):
        logger.info(f"HTTP Request: {request.method} {request.url}")
        response = await call_next(request)

        if self.skip_redirection_logging and 300 <= response.status_code < 400:
            return response

        logger.info(
            f'HTTP Response: {request.method} {request.url} "{response.status_code} {responses[response.status_code]}"'
        )
        return response
