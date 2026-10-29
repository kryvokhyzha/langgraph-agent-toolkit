"""Verify imports for the minimum UI installation."""

import streamlit

from langgraph_agent_toolkit.client import AgentClient
from langgraph_agent_toolkit.core._base_settings import Settings
from langgraph_agent_toolkit.ui.main_page import main_page


def main() -> None:
    """Verify UI imports and settings without starting an agent."""
    with AgentClient(get_info=False) as client:
        if client.agent is not None:
            raise AssertionError("AgentClient must start without a selected agent.")
    if streamlit.__name__ != "streamlit":
        raise AssertionError("The streamlit module has an unexpected name.")
    if not callable(main_page):
        raise AssertionError("main_page must be callable.")
    for mode in ("off", "llm", "provider"):
        config = Settings(TOOL_SELECTION={"mode": mode}, _env_file=None)
        if config.TOOL_SELECTION.mode != mode:
            raise AssertionError(f"The UI must accept tool selection mode {mode!r} without agent dependencies.")


if __name__ == "__main__":
    main()
