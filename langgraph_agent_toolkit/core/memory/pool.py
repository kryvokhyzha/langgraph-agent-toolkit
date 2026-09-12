"""Bound connection checks and close database sessions after cancellation."""

import asyncio

from psycopg import AsyncConnection, waiting
from psycopg_pool import AsyncConnectionPool, PoolTimeout


class CancellableAsyncConnection(AsyncConnection):
    """Close the socket when cancellation makes the operation result uncertain."""

    async def wait(self, gen, interval: float = 0.1):
        try:
            return await waiting.wait_async(gen, self.pgconn.socket, interval=interval)
        except asyncio.CancelledError:
            # Server cancellation can arrive before the query starts.
            # Do not wait for the query result or reuse this session.
            await self.close()
            raise


class CheckedAsyncConnectionPool(AsyncConnectionPool):
    """Include the health probe in the checkout deadline and preserve cancellation."""

    def __init__(self, *args, health_check_timeout: float = 5.0, **kwargs):
        self.health_check_timeout = health_check_timeout
        # The upstream check loop can retry after it catches CancelledError.
        # Run the check in getconn() before the caller receives the connection.
        super().__init__(*args, connection_class=CancellableAsyncConnection, check=None, **kwargs)

    async def getconn(self, timeout: float | None = None):
        timeout = self.timeout if timeout is None else timeout
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise PoolTimeout(f"Could not get a healthy database connection after {timeout:.2f} seconds")
            connection = await super().getconn(timeout=remaining)
            try:
                remaining = deadline - asyncio.get_running_loop().time()
                async with asyncio.timeout(min(remaining, self.health_check_timeout)):
                    await self.check_connection(connection)
                return connection
            except BaseException as exc:
                await connection.close()
                returned = asyncio.create_task(self.putconn(connection))
                cancelled = isinstance(exc, asyncio.CancelledError)
                while not returned.done():
                    try:
                        await asyncio.shield(returned)
                    except asyncio.CancelledError:
                        cancelled = True
                returned.result()
                if cancelled:
                    raise asyncio.CancelledError from exc
                if not isinstance(exc, Exception):
                    raise
                # Only a connection check failed. No caller query has run.
