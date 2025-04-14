from langgraph_agent_toolkit.service.handler import create_app
from langgraph_agent_toolkit.service.utils import setup_logging


# Create the FastAPI application
_ = setup_logging()
app = create_app()

__all__ = ["app"]
