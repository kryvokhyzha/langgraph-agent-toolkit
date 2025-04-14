from http.client import responses

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from langgraph_agent_toolkit.helper.logging import logger


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log incoming requests and outgoing responses."""

    async def dispatch(self, request: Request, call_next):
        logger.info(f"HTTP Request: {request.method} {request.url}")
        response = await call_next(request)
        logger.info(
            f'HTTP Response: {request.method} {request.url} "{response.status_code} {responses[response.status_code]}"'
        )
        return response
