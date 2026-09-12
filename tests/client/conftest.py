import pytest_asyncio

from langgraph_agent_toolkit.client import AgentClient


@pytest_asyncio.fixture
async def agent_client(mock_env):
    """Create a test client with a clean environment."""
    ac = AgentClient(base_url="http://test", get_info=False)
    ac.update_agent("test-agent", verify=False)
    yield ac
    await ac.aclose()
