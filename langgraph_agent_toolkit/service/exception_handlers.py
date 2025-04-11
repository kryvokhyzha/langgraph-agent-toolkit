import traceback

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from langgraph_agent_toolkit.helper.logging import logger


async def handle_value_error(request: Request, exc: ValueError) -> JSONResponse:
    """Handle ValueError exceptions."""
    logger.error(f"ValueError: {exc}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)},
    )


async def handle_message_conversion_error(request: Request, exc: Exception) -> JSONResponse:
    """Handle errors when converting between message types."""
    if isinstance(exc, ValueError) and "Unsupported message type" in str(exc):
        logger.error(f"Message conversion error: {exc}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": str(exc)},
        )

    raise exc


async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    error_detail = f"{exc.__class__.__name__}: {exc}"
    logger.error(f"Unexpected error: {error_detail}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"An unexpected error occurred: {str(exc)}"},
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register all exception handlers to the FastAPI app."""
    app.exception_handler(ValueError)(handle_value_error)
    app.exception_handler(Exception)(handle_unexpected_error)
    # Register the message conversion error handler
    app.add_exception_handler(ValueError, handle_message_conversion_error)
