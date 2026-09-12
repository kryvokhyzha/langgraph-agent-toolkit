import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import FastAPI, Request

from langgraph_agent_toolkit.service.factory import ServiceRunner
from langgraph_agent_toolkit.service.types import RunnerType


class TestServiceRunner:
    """Tests for the ServiceRunner class."""

    def test_init_no_custom_settings(self):
        """Test initialization without custom settings."""
        with patch("langgraph_agent_toolkit.service.factory.create_app") as mock_create_app:
            mock_create_app.return_value = Mock(spec=FastAPI)

            service_runner = ServiceRunner()

            mock_create_app.assert_called_once()
            assert service_runner.app is mock_create_app.return_value

    def test_run_uvicorn_dev_mode(self):
        """Test running with Uvicorn in development mode."""
        with patch("langgraph_agent_toolkit.service.factory.create_app"):
            with patch("langgraph_agent_toolkit.service.factory.base_settings") as mock_settings:
                # Patch uvicorn at import level
                with patch("uvicorn.run") as mock_uvicorn_run:
                    mock_settings.is_dev.return_value = True
                    mock_settings.HOST = "0.0.0.0"
                    mock_settings.PORT = 8080

                    service_runner = ServiceRunner()
                    service_runner.run_uvicorn()

                    # Check that uvicorn.run was called with the expected parameters
                    # Including the log_config parameter
                    call_args = mock_uvicorn_run.call_args
                    assert call_args[0][0] == "langgraph_agent_toolkit.service.handler:create_app"
                    assert call_args[1]["host"] == "0.0.0.0"
                    assert call_args[1]["port"] == 8080
                    assert call_args[1]["reload"] is True
                    assert call_args[1]["factory"] is True
                    assert call_args[1]["timeout_worker_healthcheck"] == 10
                    assert "log_config" in call_args[1]

    def test_run_uvicorn_prod_mode(self):
        """Test running with Uvicorn in production mode."""
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

                    # Check that uvicorn.run was called with the expected parameters
                    # Including the log_config parameter
                    call_args = mock_uvicorn_run.call_args
                    assert call_args[0][0] == mock_app
                    assert call_args[1]["host"] == "0.0.0.0"
                    assert call_args[1]["port"] == 8080
                    assert call_args[1]["reload"] is False
                    assert call_args[1]["timeout_worker_healthcheck"] == 10
                    assert "log_config" in call_args[1]

    def test_run_gunicorn_import_error(self):
        """Test running with Gunicorn when gunicorn is not installed."""
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

        with patch("langgraph_agent_toolkit.service.factory.create_app"):
            with patch.dict(sys.modules, {"azure.functions": mock_azure_functions}):
                service_runner = ServiceRunner()
                handler = service_runner.run_azure_functions()

                # Check that a function was returned
                assert callable(handler)

    def test_run_azure_functions_import_error(self):
        """Test running in Azure Functions mode when azure-functions is not installed."""
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


@pytest.mark.asyncio
async def test_azure_real_requests_share_lifespan():
    """Use real Azure requests and initialize the app once before serving them."""
    azure = pytest.importorskip("azure.functions")
    lifecycle = []

    @asynccontextmanager
    async def lifespan(app):
        lifecycle.append("start")
        app.state.ready = True
        yield
        lifecycle.append("stop")
        app.state.ready = False

    app = FastAPI(lifespan=lifespan)

    @app.post("/echo")
    async def echo(request: Request):
        assert request.app.state.ready
        return {"query": request.query_params["value"], "body": (await request.body()).decode()}

    with patch("langgraph_agent_toolkit.service.factory.create_app", return_value=app):
        runner = ServiceRunner()
    handler = runner.run_azure_functions()
    request = azure.HttpRequest("POST", "https://example.invalid/echo?value=abc", body=b"payload")
    first, second = await asyncio.gather(handler(request), handler(request))
    assert first.status_code == second.status_code == 200
    assert json.loads(first.get_body()) == {"query": "abc", "body": "payload"}
    assert lifecycle == ["start"]
    await runner.aclose()
    assert lifecycle == ["start", "stop"]
    with pytest.raises(RuntimeError, match="handler is closed"):
        await handler(request)
    await runner.aclose()
    assert lifecycle == ["start", "stop"]


@pytest.mark.asyncio
async def test_azure_does_not_serve_after_startup_failure():
    """Reject a request when the SDK reports a lifespan startup failure."""
    azure = pytest.importorskip("azure.functions")
    middleware = Mock(notify_startup=AsyncMock(return_value=False), handle_async=AsyncMock())
    with patch("langgraph_agent_toolkit.service.factory.create_app"):
        runner = ServiceRunner()
    with patch.object(azure, "AsgiMiddleware", return_value=middleware):
        handler = runner.run_azure_functions()
    with pytest.raises(RuntimeError, match="startup failed"):
        await handler(azure.HttpRequest("GET", "https://example.invalid/", body=b""))
    middleware.handle_async.assert_not_awaited()


@pytest.mark.parametrize("heartbeat_timeout", [3, 30])
def test_uvicorn_keeps_multiworker_supervision_options(heartbeat_timeout):
    """Pass process health checks to Uvicorn without changing global logging settings."""
    import copy

    import uvicorn

    original_logging = copy.deepcopy(uvicorn.config.LOGGING_CONFIG)
    with patch("langgraph_agent_toolkit.service.factory.create_app"):
        runner = ServiceRunner()
    with patch("uvicorn.run") as run:
        runner.run_uvicorn(workers=2, reload=False, timeout_worker_healthcheck=heartbeat_timeout)
    assert run.call_args.args[0] == "langgraph_agent_toolkit.service.handler:create_app"
    assert run.call_args.kwargs["factory"] is True
    assert run.call_args.kwargs["workers"] == 2
    assert run.call_args.kwargs["timeout_worker_healthcheck"] == heartbeat_timeout
    assert uvicorn.config.LOGGING_CONFIG == original_logging


def test_runner_import_does_not_load_the_application_before_worker_start():
    import subprocess
    import textwrap

    from langgraph_agent_toolkit.core._base_settings import Settings

    code = textwrap.dedent(
        """
        import sys
        import dotenv

        dotenv.find_dotenv = lambda *args, **kwargs: ""
        dotenv.load_dotenv = lambda *args, **kwargs: False

        class BlockApplicationImport:
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "langgraph_agent_toolkit.service.handler":
                    raise AssertionError("The runner loaded the application before worker startup.")

        sys.meta_path.insert(0, BlockApplicationImport())
        from langgraph_agent_toolkit.service.factory import ServiceRunner, create_app

        assert callable(ServiceRunner)
        assert callable(create_app)
        """
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in Settings.model_fields and not key.startswith("LANGGRAPH_")
    }
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_custom_settings_keep_types_in_parent_and_child_environment(monkeypatch):
    """Use the same validated settings in the runner and its worker processes."""
    from pydantic import SecretStr

    from langgraph_agent_toolkit.core._base_settings import Settings

    configured = Settings(_env_file=None)
    monkeypatch.setattr("langgraph_agent_toolkit.service.factory.base_settings", configured)
    monkeypatch.setenv("LANGGRAPH_AUTH_SECRET", "old-test-value")
    monkeypatch.setenv("LANGGRAPH_AUTH_USERS", "{}")
    monkeypatch.setenv("LANGGRAPH_MODEL_CONFIGS", "{}")
    monkeypatch.setenv("LANGGRAPH_PORT", "8080")
    model_configs = {"fake": {"provider": "fake", "name": "test", "api_key": SecretStr("nested-test-token")}}
    with patch("langgraph_agent_toolkit.service.factory.create_app"):
        ServiceRunner(
            {
                "AUTH_SECRET": "test-token",
                "PORT": "9001",
                "MODEL_CONFIGS": model_configs,
                "AUTH_USERS": {"alice": "alice-test-token", "bob": "bob-test-token"},
            }
        )
    assert isinstance(configured.AUTH_SECRET, SecretStr)
    assert configured.PORT == 9001
    child = Settings(_env_file=None)
    child._apply_langgraph_env_overrides()
    assert child.AUTH_SECRET.get_secret_value() == "test-token"
    assert child.PORT == 9001
    assert child.MODEL_CONFIGS["fake"]["api_key"] == "nested-test-token"
    assert child.AUTH_USERS["alice"].get_secret_value() == "alice-test-token"
    assert child.AUTH_USERS["bob"].get_secret_value() == "bob-test-token"
    with patch("langgraph_agent_toolkit.service.factory.create_app"):
        ServiceRunner({"AUTH_SECRET": None})
    child._apply_langgraph_env_overrides()
    assert child.AUTH_SECRET is None


def test_invalid_custom_settings_do_not_change_global_settings(monkeypatch):
    from langgraph_agent_toolkit.core._base_settings import Settings

    configured = Settings(_env_file=None)
    original_port = configured.PORT
    monkeypatch.setattr("langgraph_agent_toolkit.service.factory.base_settings", configured)
    with pytest.raises(ValueError):
        ServiceRunner({"PORT": 9001, "UNKNOWN_SETTING": True})
    assert configured.PORT == original_port


@pytest.mark.parametrize("invalid", [object(), float("nan")])
def test_unserializable_settings_do_not_change_parent_or_worker_environment(monkeypatch, invalid):
    from langgraph_agent_toolkit.core._base_settings import Settings

    configured = Settings(_env_file=None)
    original_port = configured.PORT
    monkeypatch.setattr("langgraph_agent_toolkit.service.factory.base_settings", configured)
    monkeypatch.setenv("LANGGRAPH_PORT", str(original_port))
    with pytest.raises((TypeError, ValueError)):
        ServiceRunner({"PORT": 9001, "MODEL_CONFIGS": {"invalid": {"value": invalid}}})
    assert configured.PORT == original_port
    assert configured.MODEL_CONFIGS == {}
    assert os.environ["LANGGRAPH_PORT"] == str(original_port)
