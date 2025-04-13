import warnings

from langchain_core._api import LangChainBetaWarning

from langgraph_agent_toolkit.service.handler import create_app


# Suppress LangChain beta warnings
warnings.filterwarnings("ignore", category=LangChainBetaWarning)

# Create the FastAPI application
app = create_app()

__all__ = ["app"]
