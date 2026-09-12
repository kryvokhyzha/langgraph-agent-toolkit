"""Keep one deadline owner active through request and graph cleanup."""

import asyncio
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass
class _DeadlineScope:
    active: bool = True


_request_deadline: ContextVar[_DeadlineScope | None] = ContextVar("request_deadline", default=None)


@contextmanager
def request_deadline_scope() -> Iterator[None]:
    """Mark a scope whose caller controls request cancellation and cleanup."""
    scope = _DeadlineScope()
    token = _request_deadline.set(scope)
    try:
        yield
    finally:
        scope.active = False
        _request_deadline.reset(token)


@asynccontextmanager
async def execution_timeout(timeout: float) -> AsyncIterator[None]:
    """Apply a direct execution deadline when no request owns the deadline."""
    scope = _request_deadline.get()
    if scope is not None and scope.active:
        # A second timer could cancel graph cleanup after the request has cancelled.
        yield
    else:
        async with asyncio.timeout(timeout):
            yield
