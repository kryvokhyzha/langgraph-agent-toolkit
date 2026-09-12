import asyncio

from langgraph_agent_toolkit.ui.main_page import main_page


# Run the Streamlit chat UI with:
# `streamlit run langgraph_agent_toolkit/run_app.py`


if __name__ == "__main__":
    asyncio.run(main_page())
