"""Initialize the optional queue schema or start a durable execution worker."""

import asyncio

import fire


def initialize() -> None:
    """Initialize or upgrade the background schema using deployment credentials."""
    from langgraph_agent_toolkit.core.background import initialize_background
    from langgraph_agent_toolkit.core.postgres import postgres_connection_string
    from langgraph_agent_toolkit.core.settings import settings

    asyncio.run(
        initialize_background(
            postgres_connection_string(
                settings,
                dsn=settings.BACKGROUND_POSTGRES_DSN.get_secret_value() if settings.BACKGROUND_POSTGRES_DSN else None,
            ),
            schema=settings.BACKGROUND_POSTGRES_SCHEMA,
        )
    )


def run(*, recovery_interval: int = 10, stalled_after: float = 60) -> None:
    """Start a worker with its own runtime and bounded execution capacity."""
    from langgraph_agent_toolkit.core.background import BackgroundUnavailable, queue_from_settings
    from langgraph_agent_toolkit.core.runtime import agent_runtime
    from langgraph_agent_toolkit.core.settings import settings

    if not settings.BACKGROUND_ENABLED:
        raise BackgroundUnavailable("Set BACKGROUND_ENABLED=true before starting a worker")

    async def serve() -> None:
        queue_context = queue_from_settings()
        from procrastinate.signals import on_stop

        async with agent_runtime() as runtime, queue_context as queue:
            worker = asyncio.create_task(
                queue.run_worker(
                    runtime,
                    concurrency=settings.BACKGROUND_WORKER_CONCURRENCY,
                    recovery_interval=recovery_interval,
                    stalled_after=stalled_after,
                    install_signal_handlers=False,
                )
            )
            stop_requested = False

            def stop() -> None:
                nonlocal stop_requested
                stop_requested = True
                worker.cancel()

            # The library can return normally after a fatal listener error.
            # Mark only an explicit stop signal as a successful worker shutdown.
            with on_stop(stop):
                try:
                    await worker
                except asyncio.CancelledError:
                    if not stop_requested:
                        raise
                else:
                    raise BackgroundUnavailable("The background worker stopped unexpectedly")

    asyncio.run(serve())


if __name__ == "__main__":
    fire.Fire({"initialize": initialize, "run": run})
