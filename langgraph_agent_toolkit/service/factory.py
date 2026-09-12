import asyncio
import json
import os
import sys
from collections.abc import Awaitable, Callable
from copy import deepcopy
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional

from pydantic import BaseModel, SecretStr


if TYPE_CHECKING:
    import azure.functions as func
    from fastapi import FastAPI

from langgraph_agent_toolkit.core import settings as base_settings
from langgraph_agent_toolkit.helper.logging import logger
from langgraph_agent_toolkit.service.types import RunnerType


def create_app() -> "FastAPI":
    """Load the application when needed, after worker imports complete."""
    from langgraph_agent_toolkit.service.handler import create_app as build_app

    return build_app()


def _setting_environment_value(value: Any) -> str:
    """Encode one setting for a worker environment. Do not log the result."""

    def plain(item: Any) -> Any:
        if isinstance(item, SecretStr):
            return item.get_secret_value()
        if isinstance(item, Enum):
            return plain(item.value)
        if isinstance(item, BaseModel):
            return plain(item.model_dump(mode="python"))
        if isinstance(item, dict):
            return {key: plain(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [plain(child) for child in item]
        return item

    converted = plain(value)
    encoded = converted if isinstance(converted, str) else json.dumps(converted, allow_nan=False)
    if "\x00" in encoded:
        raise ValueError("A setting cannot contain a NUL character in the worker environment.")
    return encoded


class ServiceRunner:
    """Run the API service with different runners.

    Supports Uvicorn, Gunicorn, Mangum for AWS Lambda, and Azure Functions.
    """

    def __init__(self, custom_settings: Optional[Dict[str, Any]] = None):
        """Initialize the ServiceRunner.

        Args:
            custom_settings: Optional settings that override default settings.

        """
        # Validate all overrides before changing the settings or environment.
        if custom_settings:
            candidate = base_settings.model_copy(deep=True)
            candidate.apply_overrides(custom_settings)
            validated = {key: getattr(candidate, key) for key in custom_settings}
            encoded = {key: _setting_environment_value(value) for key, value in validated.items()}
            base_settings.apply_overrides(validated)
            for key, env_value in encoded.items():
                os.environ[f"LANGGRAPH_{key}"] = env_value
                logger.info(f"Overriding setting {key}")

        self.app = create_app()
        self._azure_shutdown: Callable[[], Awaitable[None]] | None = None

    def run_uvicorn(self, **kwargs):
        """Run the API service with uvicorn."""
        try:
            import uvicorn

            # Spawned workers must finish launcher imports before their first heartbeat.
            kwargs.setdefault("timeout_worker_healthcheck", 10)

            # Use this logging configuration for uvicorn.
            log_config = deepcopy(uvicorn.config.LOGGING_CONFIG)
            log_config["handlers"] = {}
            log_config["loggers"]["uvicorn"]["handlers"] = []
            log_config["loggers"]["uvicorn.access"]["handlers"] = []
            log_config["loggers"]["uvicorn.error"]["handlers"] = []

            # Set a compatible event loop policy on Windows.
            if sys.platform == "win32":
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

            # Reload and multiple workers require an import string.
            workers = kwargs.get("workers", 1)
            reload = kwargs.get("reload", base_settings.is_dev())
            use_import_string = reload or workers > 1

            if use_import_string:
                if reload:
                    logger.info("Starting with hot reload enabled - using import string")
                if workers > 1:
                    logger.info(f"Starting with {workers} workers - using import string")

                parameters = (
                    dict(
                        host=base_settings.HOST,
                        port=base_settings.PORT,
                        reload=reload,
                        factory=True,
                        log_config=log_config,
                    )
                    | kwargs
                )

                uvicorn.run("langgraph_agent_toolkit.service.handler:create_app", **parameters)
            else:
                # Use the app instance for one worker without reload.
                parameters = (
                    dict(
                        host=base_settings.HOST,
                        port=base_settings.PORT,
                        reload=False,
                        log_config=log_config,
                    )
                    | kwargs
                )

                uvicorn.run(
                    self.app,
                    **parameters,
                )
        except ImportError:
            logger.error("Uvicorn not installed. Install it with 'pip install uvicorn'")
            sys.exit(1)

    def run_gunicorn(self, **kwargs):
        """Run the API service with gunicorn.

        Args:
            **kwargs: Arguments for gunicorn.

        """
        try:
            from gunicorn.app.base import BaseApplication

            class GunicornApp(BaseApplication):
                def __init__(self, app, options=None):
                    self.options = options or {}
                    self.application = app
                    super().__init__()

                def load_config(self):
                    for key, value in self.options.items():
                        self.cfg.set(key, value)

                def load(self):
                    return self.application

            options = {
                "bind": f"{base_settings.HOST}:{base_settings.PORT}",
                "worker_class": "uvicorn.workers.UvicornWorker",
            } | kwargs

            GunicornApp(self.app, options).run()
        except ImportError:
            logger.error("Gunicorn not installed. Install it with 'pip install gunicorn'")
            sys.exit(1)

    def run_aws_lambda(self, **kwargs):
        """Prepare the API service for AWS Lambda."""
        try:
            from mangum import Mangum

            return Mangum(self.app, **kwargs)
        except ImportError:
            logger.error("Mangum not installed. Install it with 'pip install mangum'")
            sys.exit(1)

    def run_azure_functions(self, **kwargs):
        """Return an Azure HTTP handler. Call aclose() before the event loop stops."""
        if self._azure_shutdown is not None:
            raise RuntimeError("Close the existing Azure handler before creating another handler.")
        try:
            import azure.functions as func

            middleware = func.AsgiMiddleware(self.app)
            startup_lock = asyncio.Lock()
            started = False
            closed = False

            async def main(req: "func.HttpRequest", context: "func.Context | None" = None) -> "func.HttpResponse":
                nonlocal started
                async with startup_lock:
                    if closed:
                        raise RuntimeError("The Azure handler is closed.")
                    if not started:
                        if not await middleware.notify_startup():
                            raise RuntimeError("Azure ASGI application startup failed.")
                        started = True
                return await middleware.handle_async(req, context)

            async def shutdown() -> None:
                nonlocal started, closed
                async with startup_lock:
                    if started:
                        await middleware.notify_shutdown()
                        started = False
                    closed = True

            self._azure_shutdown = shutdown

            return main
        except ImportError:
            logger.error("Azure Functions package not installed. Install with 'pip install azure-functions'")
            sys.exit(1)

    async def aclose(self) -> None:
        """Stop the Azure ASGI lifespan and release its resources."""
        if self._azure_shutdown is not None:
            await self._azure_shutdown()
            self._azure_shutdown = None

    def run(self, runner_type: RunnerType = RunnerType.UVICORN, **kwargs):
        """Run the API service with the selected runner type.

        Args:
            runner_type: Runner type.
            **kwargs: Arguments for the runner.

        """
        runner_type = RunnerType(runner_type)
        logger.info(f"Running service with runner type {runner_type.value} with options: {kwargs}")

        match runner_type:
            case RunnerType.UVICORN:
                self.run_uvicorn(**kwargs)
            case RunnerType.GUNICORN:
                self.run_gunicorn(**kwargs)
            case RunnerType.AWS_LAMBDA:
                return self.run_aws_lambda(**kwargs)
            case RunnerType.AZURE_FUNCTIONS:
                return self.run_azure_functions(**kwargs)
            case _:
                raise ValueError(f"Unknown runner type: {runner_type}")
