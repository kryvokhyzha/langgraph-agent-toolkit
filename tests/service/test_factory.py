import sys
from unittest.mock import Mock, patch

import pytest
from fastapi import FastAPI

from langgraph_agent_toolkit.service.factory import ServiceRunner
from langgraph_agent_toolkit.service.types import RunnerType


class TestServiceRunner:
    """Tests for the ServiceRunner class."""

    def test_init_no_custom_settings(self):
        """Test initialization without custom settings."""
        with patch("langgraph_agent_toolkit.service.factory.setup_logging") as mock_setup_logging:
            with patch("langgraph_agent_toolkit.service.factory.create_app") as mock_create_app:
                mock_create_app.return_value = Mock(spec=FastAPI)

                service_runner = ServiceRunner()

                mock_setup_logging.assert_called_once()
                mock_create_app.assert_called_once()
                assert service_runner.app is not None

    def test_run_uvicorn_dev_mode(self):
        """Test running with Uvicorn in development mode."""
        with patch("langgraph_agent_toolkit.service.factory.setup_logging"):
            with patch("langgraph_agent_toolkit.service.factory.create_app"):
                with patch("langgraph_agent_toolkit.service.factory.base_settings") as mock_settings:
                    # Patch uvicorn at import level
                    with patch("uvicorn.run") as mock_uvicorn_run:
                        mock_settings.is_dev.return_value = True
                        mock_settings.HOST = "0.0.0.0"
                        mock_settings.PORT = 8080

                        service_runner = ServiceRunner()
                        service_runner.run_uvicorn()

                        mock_uvicorn_run.assert_called_with(
                            "langgraph_agent_toolkit.service.handler:create_app",
                            host="0.0.0.0",
                            port=8080,
                            reload=True,
                            factory=True,
                        )

    def test_run_uvicorn_prod_mode(self):
        """Test running with Uvicorn in production mode."""
        with patch("langgraph_agent_toolkit.service.factory.setup_logging"):
            with patch("langgraph_agent_toolkit.service.factory.create_app") as mock_create_app:
                with patch("langgraph_agent_toolkit.service.factory.base_settings") as mock_settings:
                    # Patch uvicorn at import level
                    with patch("uvicorn.run") as mock_uvicorn_run:
                        mock_settings.is_dev.return_value = False
                        mock_settings.HOST = "0.0.0.0"
                        mock_settings.PORT = 8080

                        mock_app = Mock(spec=FastAPI)
                        mock_create_app.return_value = mock_app

                        service_runner = ServiceRunner()
                        service_runner.run_uvicorn()

                        mock_uvicorn_run.assert_called_with(mock_app, host="0.0.0.0", port=8080, reload=False)

    def test_run_gunicorn_import_error(self):
        """Test running with Gunicorn when gunicorn is not installed."""
        with patch("langgraph_agent_toolkit.service.factory.setup_logging"):
            with patch("langgraph_agent_toolkit.service.factory.create_app"):
                with patch("langgraph_agent_toolkit.service.factory.sys") as mock_sys:
                    with patch.dict(sys.modules, {}):
                        with patch("langgraph_agent_toolkit.service.factory.logger") as mock_logger:
                            # Simulate ImportError by making __import__ raise it
                            import builtins

                            original_import = builtins.__import__

                            def mock_import(name, *args, **kwargs):
                                if name == "gunicorn.app.base":
                                    raise ImportError("No module named 'gunicorn'")
                                return original_import(name, *args, **kwargs)

                            with patch("builtins.__import__", side_effect=mock_import):
                                service_runner = ServiceRunner()
                                service_runner.run_gunicorn()

                                mock_logger.error.assert_called_with(
                                    "Gunicorn not installed. Install it with 'pip install gunicorn'"
                                )
                                mock_sys.exit.assert_called_with(1)

    def test_run_aws_lambda(self):
        """Test running in AWS Lambda mode."""
        # Mock Mangum class
        mock_mangum = Mock()
        mock_mangum_instance = Mock()
        mock_mangum.return_value = mock_mangum_instance

        with patch("langgraph_agent_toolkit.service.factory.setup_logging"):
            with patch("langgraph_agent_toolkit.service.factory.create_app") as mock_create_app:
                with patch.dict(sys.modules, {"mangum": Mock()}):
                    # Set up the Mangum mock
                    sys.modules["mangum"].Mangum = mock_mangum

                    mock_app = Mock(spec=FastAPI)
                    mock_create_app.return_value = mock_app

                    service_runner = ServiceRunner()
                    handler = service_runner.run_aws_lambda()

                    # Check that Mangum was called with the app
                    mock_mangum.assert_called_with(mock_app)
                    assert handler == mock_mangum_instance

    def test_run_aws_lambda_import_error(self):
        """Test running in AWS Lambda mode when mangum is not installed."""
        with patch("langgraph_agent_toolkit.service.factory.setup_logging"):
            with patch("langgraph_agent_toolkit.service.factory.create_app"):
                with patch("langgraph_agent_toolkit.service.factory.sys") as mock_sys:
                    # Simulate ImportError
                    with patch.dict(sys.modules, {}):
                        with patch("langgraph_agent_toolkit.service.factory.logger") as mock_logger:
                            import builtins

                            original_import = builtins.__import__

                            def mock_import(name, *args, **kwargs):
                                if name == "mangum":
                                    raise ImportError("No module named 'mangum'")
                                return original_import(name, *args, **kwargs)

                            with patch("builtins.__import__", side_effect=mock_import):
                                service_runner = ServiceRunner()
                                service_runner.run_aws_lambda()

                                mock_logger.error.assert_called_with(
                                    "Mangum not installed. Install it with 'pip install mangum'"
                                )
                                mock_sys.exit.assert_called_with(1)

    def test_run_azure_functions(self):
        """Test running in Azure Functions mode."""
        mock_azure_functions = Mock()

        with patch("langgraph_agent_toolkit.service.factory.setup_logging"):
            with patch("langgraph_agent_toolkit.service.factory.create_app"):
                with patch.dict(sys.modules, {"azure.functions": mock_azure_functions}):
                    service_runner = ServiceRunner()
                    handler = service_runner.run_azure_functions()

                    # Check that a function was returned
                    assert callable(handler)

    def test_run_azure_functions_import_error(self):
        """Test running in Azure Functions mode when azure-functions is not installed."""
        with patch("langgraph_agent_toolkit.service.factory.setup_logging"):
            with patch("langgraph_agent_toolkit.service.factory.create_app"):
                with patch("langgraph_agent_toolkit.service.factory.sys") as mock_sys:
                    # Simulate ImportError
                    with patch.dict(sys.modules, {}):
                        with patch("langgraph_agent_toolkit.service.factory.logger") as mock_logger:
                            import builtins

                            original_import = builtins.__import__

                            def mock_import(name, *args, **kwargs):
                                if name == "azure.functions":
                                    raise ImportError("No module named 'azure.functions'")
                                return original_import(name, *args, **kwargs)

                            with patch("builtins.__import__", side_effect=mock_import):
                                service_runner = ServiceRunner()
                                service_runner.run_azure_functions()

                                mock_logger.error.assert_called_with(
                                    "Azure Functions package not installed. Install with 'pip install azure-functions'"
                                )
                                mock_sys.exit.assert_called_with(1)

    def test_run_dispatcher(self):
        """Test that the run method dispatches to the correct runner method."""
        with patch("langgraph_agent_toolkit.service.factory.setup_logging"):
            with patch("langgraph_agent_toolkit.service.factory.create_app"):
                # Patch each method individually for better control
                with patch.object(ServiceRunner, "run_uvicorn") as mock_run_uvicorn:
                    with patch.object(ServiceRunner, "run_gunicorn") as mock_run_gunicorn:
                        with patch.object(ServiceRunner, "run_aws_lambda") as mock_run_aws_lambda:
                            with patch.object(ServiceRunner, "run_azure_functions") as mock_run_azure_functions:
                                service_runner = ServiceRunner()

                                # Test UVICORN - use enum directly
                                service_runner.run(RunnerType.UVICORN)
                                mock_run_uvicorn.assert_called_once()

                                # Test GUNICORN - use enum directly
                                service_runner.run(RunnerType.GUNICORN, workers=8)
                                mock_run_gunicorn.assert_called_once_with(workers=8)

                                # Test AWS_LAMBDA - use enum directly
                                service_runner.run(RunnerType.AWS_LAMBDA)
                                mock_run_aws_lambda.assert_called_once()

                                # Test AZURE_FUNCTIONS - use enum directly
                                service_runner.run(RunnerType.AZURE_FUNCTIONS)
                                mock_run_azure_functions.assert_called_once()

    def test_run_invalid_runner_type(self):
        """Test that the run method raises an error for an invalid runner type."""
        with patch("langgraph_agent_toolkit.service.factory.setup_logging"):
            with patch("langgraph_agent_toolkit.service.factory.create_app"):
                service_runner = ServiceRunner()

                # The issue is with how the invalid runner type is being passed
                # In the actual code, it's trying to convert the string to an enum
                # and then access .value on it, which fails

                # Instead of directly passing a string, let's catch the ValueError
                # that will be raised when trying to create the RunnerType enum
                with pytest.raises(ValueError):
                    # This will try to create RunnerType("invalid_runner") internally
                    # which will fail with ValueError
                    service_runner.run("invalid_runner")

    def test_handle_azure_request(self):
        """Test the _handle_azure_request method."""
        mock_app = Mock(spec=FastAPI)
        mock_req = Mock()
        mock_func = Mock()

        # Mock the app to capture the ASGI interface calls
        async def mock_asgi_app(scope, receive, send):
            # Call send with a start response
            await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/plain")]})
            # Call send with a body
            await send({"type": "http.response.body", "body": b"Hello, World!"})

        mock_app.side_effect = mock_asgi_app

        # Mock Azure request and response
        mock_req.method = "GET"
        mock_req.url.path = "/test"
        mock_req.url.query = ""
        mock_req.headers = {"content-type": "text/plain"}
        mock_req.get_body.return_value = b""

        mock_func.HttpResponse.return_value = "response"

        # Test the method
        with patch.dict(sys.modules, {"azure.functions": mock_func}):
            import asyncio

            response = asyncio.run(ServiceRunner._handle_azure_request(mock_app, mock_req))

            # Verify HttpResponse was called with the right parameters
            mock_func.HttpResponse.assert_called_once_with(
                body=b"Hello, World!", status_code=200, headers={"content-type": "text/plain"}
            )
            assert response == "response"
