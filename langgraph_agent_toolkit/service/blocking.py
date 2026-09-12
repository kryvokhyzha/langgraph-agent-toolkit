"""Bound synchronous feedback calls that can outlive their HTTP requests."""

import asyncio
import threading
from collections.abc import Callable
from contextvars import copy_context
from typing import Any, TypeVar

from langgraph_agent_toolkit.helper.logging import logger
from langgraph_agent_toolkit.service.admission import ServiceBusyError


_Result = TypeVar("_Result")


class BoundedBlockingExecutor:
    """Keep thread capacity reserved until each synchronous call finishes.

    Use one instance on the service event loop. The executor has no queue.
    Cancellation stops the caller's wait. It cannot stop synchronous code.
    """

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("Blocking capacity must be positive")
        self.capacity = capacity
        self._pending: set[asyncio.Future] = set()
        self._closed = False

    async def run(self, function: Callable[..., _Result], *args: Any, **kwargs: Any) -> _Result:
        """Run one synchronous call if a thread slot is available."""
        if self._closed or len(self._pending) >= self.capacity:
            raise ServiceBusyError("Feedback thread capacity is full or shutting down")
        loop = asyncio.get_running_loop()
        completed = loop.create_future()
        self._pending.add(completed)
        context = copy_context()

        def finish(value: _Result | None, error: BaseException | None) -> None:
            self._pending.discard(completed)
            # Store errors as values. A cancelled caller cannot consume an exception.
            completed.set_result((value, error))

        def execute() -> None:
            value = None
            error = None
            try:
                value = context.run(function, *args, **kwargs)
            except BaseException as exc:
                error = exc
            try:
                loop.call_soon_threadsafe(finish, value, error)
            except RuntimeError:
                pass  # A daemon thread can finish after the worker event loop closes.

        try:
            threading.Thread(target=execute, name="feedback", daemon=True).start()
        except BaseException:
            self._pending.discard(completed)
            raise
        value, error = await asyncio.shield(completed)
        if error is not None:
            raise error
        return value

    async def aclose(self, timeout: float) -> None:
        """Stop accepting calls and wait up to the shutdown time limit."""
        self._closed = True
        if self._pending:
            _, pending = await asyncio.wait(self._pending, timeout=timeout)
            if pending:
                logger.warning("Feedback calls exceeded the shutdown time limit; pending feedback may be lost")
