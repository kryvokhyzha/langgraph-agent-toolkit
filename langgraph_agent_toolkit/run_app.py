import asyncio

from langgraph_agent_toolkit.ui.main_page import main_page


# Streamlit chat UI for interacting with the LangGraph agent service via the AgentClient SDK.
# The UI is organized under `langgraph_agent_toolkit/ui/`:
#   - ui/main_page.py             high-level page structure and request handling
#   - ui/components/side_panel.py sidebar controls and the settings popover
#   - ui/components/draw_message.py  message rendering (tokens, tool calls, task data)
#   - ui/utils/message.py         welcome message, feedback, and multimodal content blocks
#   - ui/utils/constants.py       titles, icons, accepted upload types
#
# Run with: streamlit run langgraph_agent_toolkit/run_app.py


if __name__ == "__main__":
    asyncio.run(main_page())
