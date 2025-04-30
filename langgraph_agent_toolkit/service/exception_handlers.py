import os
import sys
import traceback

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from langgraph_agent_toolkit.helper.logging import logger
from langgraph_agent_toolkit.helper.types import EnvironmentMode


def register_exception_handlers(app: FastAPI) -> None:
    """Register all exception handlers to the FastAPI app using decorators."""
    env_mode = EnvironmentMode(os.environ.get("ENV_MODE", EnvironmentMode.PRODUCTION))
    include_traceback = env_mode != EnvironmentMode.PRODUCTION

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        """Handle HTTPException with appropriate logging."""
        logger.warning(f"HTTPException: {exc.detail} (status {exc.status_code})")

        content = {"detail": exc.detail}

        # Include headers if present
        if exc.headers:
            return JSONResponse(status_code=exc.status_code, content=content, headers=exc.headers)

        return JSONResponse(status_code=exc.status_code, content=content)

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        """Handle ValueError exceptions."""
        logger.opt(exception=sys.exc_info()).error(f"ValueError: {exc}")

        content = {"detail": str(exc)}
        if include_traceback:
            content["traceback"] = traceback.format_exc()

        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=content,
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle all other unexpected exceptions."""
        # Special handling for ValueError with "Unsupported message type"
        if isinstance(exc, ValueError) and "Unsupported message type" in str(exc):
            logger.opt(exception=sys.exc_info()).error(f"Message conversion error: {exc}")

            content = {"detail": str(exc)}
            if include_traceback:
                content["traceback"] = traceback.format_exc()

            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content=content,
            )

        # For all other exceptions
        error_detail = f"{exc.__class__.__name__}: {exc}"
        logger.opt(exception=sys.exc_info()).error(f"Unexpected error: {error_detail}")

        content = {"detail": f"An unexpected error occurred: {str(exc)}"}
        if include_traceback:
            content["traceback"] = traceback.format_exc()

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=content,
        )
