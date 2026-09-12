"""Verify imports for the minimum UI installation."""

import streamlit

from langgraph_agent_toolkit.client import AgentClient
from langgraph_agent_toolkit.ui.main_page import main_page


def main() -> None:
    """Verify that the client, Streamlit, and the main page are available."""
    if not callable(AgentClient):
        raise AssertionError("AgentClient must be callable.")
    if streamlit.__name__ != "streamlit":
        raise AssertionError("The streamlit module has an unexpected name.")
    if not callable(main_page):
        raise AssertionError("main_page must be callable.")


if __name__ == "__main__":
    main()
