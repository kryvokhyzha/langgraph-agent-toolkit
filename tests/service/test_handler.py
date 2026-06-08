from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.service.handler import create_app


def test_health_probes_before_lifespan():
    """Without running the lifespan, the app is live but not ready/started (503 on those probes)."""
    client = TestClient(create_app())  # no `with` -> lifespan does not run

    assert client.get("/health").status_code == 200
    assert client.get("/health/live").status_code == 200
    assert client.get("/health/ready").status_code == 503
    assert client.get("/health/startup").status_code == 503


def test_lifespan_marks_ready_after_initializing_agents():
    """Running the lifespan (mocked executor, no memory) flips readiness and /health/ready -> 200."""
    mock_agent = Mock()
    mock_agent.graph = Mock(checkpointer=None)
    mock_agent.observability = None

    agent_info = Mock()
    agent_info.key = "react-agent"
    mock_executor = Mock()
    mock_executor.get_all_agent_info.return_value = [agent_info]
    mock_executor.get_agent.return_value = mock_agent

    with patch("langgraph_agent_toolkit.service.handler.AgentExecutor", return_value=mock_executor):
        with patch.object(settings, "MEMORY_BACKEND", None):
            app = create_app()
            with TestClient(app) as client:  # entering the context runs the lifespan
                ready = client.get("/health/ready")
                startup = client.get("/health/startup")

    assert ready.status_code == 200
    assert startup.status_code == 200
    assert ready.json()["status"] == "ready"
    assert app.state.ready is True
    assert "react-agent" in app.state.initialized_agents


def test_health_db_pool_states():
    """/health/db reports no_pool / healthy / exhausted / error depending on the pool stats."""
    app = create_app()
    client = TestClient(app)

    # No pool configured (default).
    assert client.get("/health/db").json()["status"] == "no_pool"

    pool = Mock()
    app.state.db_pool = pool

    pool.get_stats.return_value = {"pool_size": 5, "pool_available": 3}
    assert client.get("/health/db").json()["status"] == "healthy"

    pool.get_stats.return_value = {"pool_size": 5, "pool_available": 0}
    assert client.get("/health/db").json()["status"] == "exhausted"

    pool.get_stats.side_effect = RuntimeError("boom")
    assert client.get("/health/db").json()["status"] == "error"
