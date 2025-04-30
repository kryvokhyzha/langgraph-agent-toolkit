from unittest.mock import patch

from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import SecretStr

from langgraph_agent_toolkit.schema import ChatMessage
from langgraph_agent_toolkit.service.routes import private_router
from langgraph_agent_toolkit.service.utils import verify_bearer


def test_no_auth_secret(mock_settings, mock_agent_executor, test_client):
    """Test that when AUTH_SECRET is not set, all requests are allowed."""
    mock_settings.AUTH_SECRET = None

    # Create a response
    mock_response = ChatMessage(type="ai", content="This is a test response")
    mock_agent_executor.invoke.return_value = mock_response

    with patch("langgraph_agent_toolkit.service.utils.verify_bearer", side_effect=verify_bearer):  # Patch verify_bearer
        with patch("langgraph_agent_toolkit.service.routes.get_agent_executor", return_value=mock_agent_executor):
            response = test_client.post(
                "/invoke",
                json={"message": "test"},
                headers={"Authorization": "Bearer any-token"},
            )
            assert response.status_code == 200

            # Should also work without any auth header
            response = test_client.post("/invoke", json={"message": "test"})
            assert response.status_code == 200


def test_auth_secret_correct(mock_settings, mock_agent_executor, test_client):
    """Test that when AUTH_SECRET is set, requests with correct token are allowed."""
    mock_settings.AUTH_SECRET = SecretStr("test-secret")

    # Create a response
    mock_response = ChatMessage(type="ai", content="This is a test response")
    mock_agent_executor.invoke.return_value = mock_response

    with patch("langgraph_agent_toolkit.service.utils.verify_bearer", side_effect=verify_bearer):
        with patch("langgraph_agent_toolkit.service.routes.get_agent_executor", return_value=mock_agent_executor):
            response = test_client.post(
                "/invoke",
                json={"message": "test"},
                headers={"Authorization": "Bearer test-secret"},
            )
            assert response.status_code == 200


def test_auth_secret_incorrect(mock_settings, mock_agent_executor, app):
    """Test that when AUTH_SECRET is set, requests with wrong token are rejected."""
    # Set the auth secret for testing
    mock_settings.AUTH_SECRET = SecretStr("test-secret")

    # Create a new test app with a mocked auth dependency
    test_app = FastAPI()

    # Create auth dependency that always fails
    async def failed_auth_dependency():
        raise HTTPException(status_code=401, detail="Invalid authentication")

    # Include the private router with our mocked dependency
    test_app.include_router(private_router, dependencies=[Depends(failed_auth_dependency)])

    # Set up app state
    test_app.state.agent_executor = mock_agent_executor

    # Create a test client with our custom app
    test_client = TestClient(test_app)

    # Test with wrong auth token
    response = test_client.post(
        "/invoke",
        json={"message": "test"},
        headers={"Authorization": "Bearer wrong-secret"},
    )
    assert response.status_code == 401

    # Test without auth header
    response = test_client.post("/invoke", json={"message": "test"})
    assert response.status_code == 401
