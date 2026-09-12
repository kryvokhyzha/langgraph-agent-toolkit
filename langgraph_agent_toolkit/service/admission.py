"""Bound HTTP work and stop abandoned requests before releasing capacity."""

import asyncio
from contextlib import asynccontextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from langgraph_agent_toolkit.core.execution import request_deadline_scope
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.helper.logging import logger


@dataclass
class _Lease:
    active: bool = True


_request_lease: ContextVar[_Lease | None] = ContextVar("request_admission_lease", default=None)


class ServiceBusyError(Exception):
    """The worker has no request capacity available."""


class ResponseSendTimeout(Exception):
    """The client did not accept response data before the deadline."""


class RequestAdmission:
    """Limit active requests and queued waiters in one worker."""

    def __init__(self, capacity: int, max_waiters: int, queue_timeout: float):
        self._semaphore = asyncio.Semaphore(capacity)
        self.max_waiters = max_waiters
        self.queue_timeout = queue_timeout
        self.active = 0
        self.waiting = 0

    @asynccontextmanager
    async def acquire(self):
        queued = self._semaphore.locked()
        if queued and self.waiting >= self.max_waiters:
            raise ServiceBusyError
        if queued:
            self.waiting += 1
        try:
            try:
                async with asyncio.timeout(self.queue_timeout):
                    await self._semaphore.acquire()
            except TimeoutError as exc:
                raise ServiceBusyError from exc
        finally:
            if queued:
                self.waiting -= 1
        self.active += 1
        try:
            yield
        finally:
            self.active -= 1
            self._semaphore.release()


async def _drain(task: asyncio.Task) -> None:
    """Finish cleanup without sending a second cancellation to the task."""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
        except Exception:
            break


class RequestAdmissionMiddleware:
    """Admit requests before buffering bodies and hold capacity through cleanup."""

    # Probes must remain available when all execution slots are in use.
    _probe_paths = frozenset({"/health", "/health/live", "/health/ready", "/health/startup", "/health/db"})

    def __init__(self, app: ASGIApp):
        self.app = app
        self.admission = RequestAdmission(
            settings.REQUEST_MAX_CONCURRENT, settings.REQUEST_QUEUE_MAX_WAITERS, settings.REQUEST_QUEUE_TIMEOUT
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope["path"]
        root = scope.get("root_path", "").rstrip("/")
        if root and path.startswith(root + "/"):
            path = path[len(root) :]
        if scope["method"] in {"GET", "HEAD"} and path in self._probe_paths:
            scope["state"] = {**scope.get("state", {}), "probe_without_body": True}
            await self.app(scope, receive, send)
            return
        current = _request_lease.get()
        if current is not None and current.active:
            await self._error(scope, receive, send, 409, "Nested service requests are not supported", "nested_request")
            return
        scope["app"].state.request_admission = self.admission
        try:
            async with self.admission.acquire():
                lease = _Lease()
                token = _request_lease.set(lease)
                try:
                    await self._serve(scope, receive, send)
                finally:
                    lease.active = False
                    _request_lease.reset(token)
        except ServiceBusyError:
            await self._error(scope, receive, send, 503, "Worker request capacity is full", "service_busy")

    @staticmethod
    async def _error(scope: Scope, receive: Receive, send: Send, status: int, detail: str, code: str) -> None:
        async with asyncio.timeout(settings.RESPONSE_SEND_TIMEOUT):
            await JSONResponse(
                {"detail": detail, "error_code": code},
                status_code=status,
                headers={"Retry-After": "1"} if status == 503 else None,
            )(scope, receive, send)

    async def _serve(self, scope: Scope, receive: Receive, send: Send) -> None:
        disconnected = asyncio.Event()
        body_complete = False
        response_started = False
        response_complete = False
        monitor = None

        def cancel_once() -> None:
            if not task.done() and not task.cancelling():
                task.cancel()

        async def watch_disconnect() -> None:
            try:
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        if not response_complete:
                            disconnected.set()
                            cancel_once()
                        return
            except OSError:
                disconnected.set()
                cancel_once()

        async def guarded_receive() -> Message:
            nonlocal body_complete, monitor
            if body_complete:
                await disconnected.wait()
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.disconnect":
                disconnected.set()
            elif message["type"] == "http.request" and not message.get("more_body", False):
                body_complete = True
                # Only this task reads the ASGI channel after the final body chunk.
                monitor = asyncio.create_task(watch_disconnect(), name="http-disconnect")
            return message

        async def guarded_send(message: Message) -> None:
            nonlocal response_started, response_complete
            if message["type"] == "http.response.start":
                response_started = True
            try:
                async with asyncio.timeout(settings.RESPONSE_SEND_TIMEOUT):
                    await send(message)
                if message["type"] == "http.response.body" and not message.get("more_body", False):
                    response_complete = True
            except TimeoutError as exc:
                # Do not invoke FastAPI's request timeout handler after headers.
                raise ResponseSendTimeout("Response send time limit expired") from exc

        async def run() -> None:
            manager = getattr(scope["app"].state, "llm_transport_manager", None)
            with request_deadline_scope(), manager.bind() if manager is not None else nullcontext():
                await self.app(scope, guarded_receive, guarded_send)

        task = asyncio.create_task(run(), name="http-request")
        try:
            done, _ = await asyncio.wait({task}, timeout=settings.REQUEST_TIMEOUT)
            if not done and not task.done():
                cancel_once()
                await self._cleanup(task, scope)
                if not task.cancelled():
                    task.exception()
                if asyncio.current_task().cancelling():
                    raise asyncio.CancelledError
                raise TimeoutError("Request time limit expired")
            await task
        except asyncio.CancelledError:
            cancel_once()
            await self._cleanup(task, scope)
            if not task.cancelled():
                task.exception()
            if not disconnected.is_set() or asyncio.current_task().cancelling():
                raise
        except (TimeoutError, ResponseSendTimeout):
            if response_started:
                logger.warning("Request or response send deadline expired after response headers")
                raise
            await self._error(scope, receive, send, 504, "Request time limit expired", "request_timeout")
        finally:
            if monitor is not None:
                monitor.cancel()
                await _drain(monitor)
                if not monitor.cancelled():
                    monitor.result()

    @staticmethod
    async def _cleanup(task: asyncio.Task, scope: Scope) -> None:
        deadline = asyncio.get_running_loop().time() + settings.REQUEST_CLEANUP_TIMEOUT
        state = scope["app"].state
        stalled = False
        try:
            while not task.done():
                try:
                    if stalled:
                        await asyncio.shield(task)
                    else:
                        done, _ = await asyncio.wait(
                            {task}, timeout=max(0, deadline - asyncio.get_running_loop().time())
                        )
                        if not done:
                            stalled = True
                            state.stalled_request_cleanups = getattr(state, "stalled_request_cleanups", 0) + 1
                            logger.error("Request cleanup is stalled; health probes require worker recovery")
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
        finally:
            if stalled:
                state.stalled_request_cleanups -= 1
