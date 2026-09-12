"""Close response iterators when transport errors stop streaming."""

import asyncio

from anyio import CancelScope
from starlette.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

from langgraph_agent_toolkit.service.admission import _drain


class ClosingStreamingResponse(StreamingResponse):
    """Finish generator cleanup before the request releases its resources."""

    async def stream_response(self, send: Send) -> None:
        async def consume() -> None:
            try:
                await super(ClosingStreamingResponse, self).stream_response(send)
            finally:
                # Close in the task that opened trace and conversation contexts.
                close = getattr(self.body_iterator, "aclose", None)
                if close is not None:
                    await close()

        # AnyIO task groups can repeat cancellation while LangGraph closes child tasks.
        # Give graph execution one cancellation and wait for all of its cleanup.
        task = asyncio.create_task(consume(), name="http-stream")
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if not task.done() and not task.cancelling():
                task.cancel()
            with CancelScope(shield=True):
                await _drain(task)
            if not task.cancelled():
                task.exception()
            raise

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Starlette can stop at send() while the generator is suspended at yield.
            # Its streaming tasks must stop before this iterator can close.
            close = getattr(self.body_iterator, "aclose", None)
            if close is not None:
                await close()
