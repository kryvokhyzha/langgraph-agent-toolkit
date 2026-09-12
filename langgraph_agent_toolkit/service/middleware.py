import asyncio
import os
from http.client import responses

from starlette.datastructures import URL
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.helper.logging import logger


class RequestSizeLimitMiddleware:
    """Reject oversized request bodies before JSON parsing."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        from starlette.responses import JSONResponse

        limit = 0 if scope.get("state", {}).get("probe_without_body") else settings.REQUEST_MAX_BYTES
        headers = dict(scope.get("headers", []))
        try:
            length = int(headers.get(b"content-length", b"0"))
            if length < 0:
                raise ValueError("negative length")
        except ValueError:
            await JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)(scope, receive, send)
            return
        if length > limit:
            await JSONResponse({"detail": "Request body is too large"}, status_code=413)(scope, receive, send)
            return
        body = bytearray()
        try:
            async with asyncio.timeout(settings.REQUEST_TIMEOUT):
                while True:
                    event = await receive()
                    if event["type"] == "http.disconnect":
                        return
                    chunk = event.get("body", b"")
                    if len(body) + len(chunk) > limit:
                        await JSONResponse({"detail": "Request body is too large"}, status_code=413)(
                            scope, receive, send
                        )
                        return
                    body.extend(chunk)
                    if not event.get("more_body", False):
                        break
        except TimeoutError:
            await JSONResponse({"detail": "Request body time limit expired"}, status_code=408)(scope, receive, send)
            return
        replayed = False

        async def replay():
            nonlocal replayed
            if not replayed:
                replayed = True
                event = {"type": "http.request", "body": bytes(body), "more_body": False}
                body.clear()
                return event
            return await receive()

        await self.app(scope, replay, send)


class LoggingMiddleware:
    """Log incoming requests and outgoing responses."""

    def __init__(self, app: ASGIApp):
        self.app = app
        self.skip_redirection_logging = os.getenv("SKIP_REDIRECTION_LOGGING", "true").lower() == "true"

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        method, url = scope["method"], URL(scope=scope)
        logger.info(f"HTTP Request: {method} {url}")

        async def log_response(message: Message) -> None:
            if message["type"] == "http.response.start":
                code = message["status"]
                if not (self.skip_redirection_logging and 300 <= code < 400):
                    logger.info(f'HTTP Response: {method} {url} "{code} {responses.get(code, "Unknown")}"')
            await send(message)

        await self.app(scope, receive, log_response)
