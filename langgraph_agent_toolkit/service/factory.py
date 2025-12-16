import asyncio
import json
import os
import sys
from typing import TYPE_CHECKING, Any, Dict, Optional


if TYPE_CHECKING:
    import azure.functions as func

from fastapi import FastAPI

from langgraph_agent_toolkit.core import settings as base_settings
from langgraph_agent_toolkit.helper.logging import logger
from langgraph_agent_toolkit.service.handler import create_app
from langgraph_agent_toolkit.service.types import RunnerType
from langgraph_agent_toolkit.service.utils import setup_logging


class ServiceRunner:
    """A factory class to run the API service in different ways.

    This class provides methods to run the service with different runners:
    - With Uvicorn
    - With Gunicorn
    - With Mangum (AWS Lambda)
    - With Azure Functions
    """

    def __init__(self, custom_settings: Optional[Dict[str, Any]] = None):
        """Initialize the ServiceRunner.

        Args:
            custom_settings: Optional dictionary of settings to override the default settings.

        """
        # Override global settings if provided
        if custom_settings:
            for key, value in custom_settings.items():
                if hasattr(base_settings, key):
                    # Set the value in the global settings object
                    setattr(base_settings, key, value)
                    logger.info(f"Overriding setting {key}")

                    # Also set it as an environment variable for child processes
                    # Handle different types appropriately for environment variables
                    env_value = None
                    if isinstance(value, list):
                        # Convert lists to JSON strings
                        env_value = json.dumps(value)
                    elif isinstance(value, bool):
                        # Convert booleans to string "True" or "False"
                        env_value = str(value)
                    elif value is not None:
                        # Convert other values to strings
                        env_value = str(value)

                    if env_value is not None:
                        os.environ[f"LANGGRAPH_{key}"] = env_value
                else:
                    logger.warning(f"Setting {key} not found in settings")

        self.app = create_app()

    def run_uvicorn(self, **kwargs):
        """Run the API service with uvicorn."""
        try:
            import uvicorn

            # Ensure uvicorn uses our logging configuration
            log_config = uvicorn.config.LOGGING_CONFIG.copy()
            log_config["handlers"] = {}
            log_config["loggers"]["uvicorn"]["handlers"] = []
            log_config["loggers"]["uvicorn.access"]["handlers"] = []
            log_config["loggers"]["uvicorn.error"]["handlers"] = []

            # Set Compatible event loop policy on Windows Systems.
            if sys.platform == "win32":
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

            # Check if we need to use import string (required for reload or workers > 1)
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
                # Single worker, no reload - use the app instance directly
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
            **kwargs: Additional arguments to pass to gunicorn

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
        """Prepare the API service for Azure Functions."""
        try:
            import azure.functions as func

            async def main(req: func.HttpRequest) -> func.HttpResponse:
                # Process the request through ASGI app
                return await self._handle_azure_request(self.app, req)

            return main
        except ImportError:
            logger.error("Azure Functions package not installed. Install with 'pip install azure-functions'")
            sys.exit(1)

    @staticmethod
    async def _handle_azure_request(app: FastAPI, req: "func.HttpRequest") -> "func.HttpResponse":
        """Handle Azure Functions HTTP request."""
        import azure.functions as func

        # Convert request to ASGI format
        scope = {
            "type": "http",
            "http_version": "1.1",
            "method": req.method,
            "path": req.url.path,
            "query_string": req.url.query.encode(),
            "headers": [(k.encode(), v.encode()) for k, v in req.headers.items()],
        }

        # Create response container
        response_body = []
        response_status = None
        response_headers = []

        async def receive():
            return {"type": "http.request", "body": req.get_body() or b""}

        async def send(message):
            nonlocal response_status, response_headers

            if message["type"] == "http.response.start":
                response_status = message["status"]
                response_headers = message["headers"]
            elif message["type"] == "http.response.body":
                response_body.append(message["body"])

        # Process the request through ASGI app
        await app(scope, receive, send)

        # Build Azure Functions response
        headers = {k.decode(): v.decode() for k, v in response_headers}
        body = b"".join(response_body)

        return func.HttpResponse(
            body=body,
            status_code=response_status,
            headers=headers,
        )

    def run(self, runner_type: RunnerType = RunnerType.UVICORN, **kwargs):
        """Run the API service with the specified runner type.

        Args:
            runner_type: The type of runner to use.
            **kwargs: Additional arguments to pass to the runner.

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
