"""Test native Phoenix prompt storage through the toolkit backend."""

import os
from uuid import uuid4

import httpx
import pytest


pytestmark = pytest.mark.phoenix_live


@pytest.mark.parametrize("template_format", ["f-string", "mustache"])
async def test_prompt_versions_tags_and_deletion_use_native_storage(template_format):
    from langgraph_agent_toolkit.core.observability.phoenix import PhoenixObservability

    names = ("BASE_URL", "SERVER_VERSION")
    missing = [f"LAT_TEST_PHOENIX_{name}" for name in names if not os.environ.get(f"LAT_TEST_PHOENIX_{name}")]
    if missing:
        pytest.fail("Set the Phoenix test server configuration: " + ", ".join(missing))
    endpoint = os.environ["LAT_TEST_PHOENIX_BASE_URL"].rstrip("/")
    name = "lat-test-prompt-" + uuid4().hex
    variable = "{name}" if template_format == "f-string" else "{{name}}"
    backend = PhoenixObservability(endpoint=endpoint, project_name="lat-test-prompts")
    metadata = {
        "model_name": "lat-test-calculator",
        "template_format": "F_STRING" if template_format == "f-string" else "MUSTACHE",
        "labels": ["production"],
    }
    with httpx.Client(base_url=endpoint + "/", timeout=10, trust_env=False) as reader:
        health = reader.get("healthz")
        health.raise_for_status()
        assert health.headers["x-phoenix-server-version"] == os.environ["LAT_TEST_PHOENIX_SERVER_VERSION"]
        try:
            backend.push_prompt(name, f"First revision: {variable}", metadata=metadata)
            prompt, first = await backend.apull_prompt(
                name, template_format=template_format, return_with_prompt_object=True
            )
            assert prompt.format_messages(name="user-1")[-1].content == "First revision: user-1"
            backend.push_prompt(name, f"Unused revision: {variable}", metadata=metadata, force_create_new_version=False)
            response = reader.get(f"v1/prompts/{name}/versions")
            response.raise_for_status()
            assert [item["id"] for item in response.json()["data"]] == [first.id]
            backend.push_prompt(
                name,
                [
                    {"role": "system", "content": "Answer briefly."},
                    {"role": "user", "content": f"Second revision: {variable}"},
                ],
                metadata={**metadata, "labels": ["candidate"]},
            )
            prompt, second = backend.pull_prompt(name, template_format=template_format, return_with_prompt_object=True)
            assert first.id != second.id
            assert prompt.format_messages(name="user-1")[-1].content == "Second revision: user-1"
            for options in ({"label": "production"}, {"version": first.id}):
                pinned, selected = backend.pull_prompt(
                    name, template_format=template_format, return_with_prompt_object=True, **options
                )
                assert selected.id == first.id
                assert pinned.format_messages(name="user-1")[-1].content == "First revision: user-1"
            response = reader.get(f"v1/prompts/{name}/versions")
            response.raise_for_status()
            assert {item["id"] for item in response.json()["data"]} == {first.id, second.id}
            for tag, version in (("production", first.id), ("candidate", second.id)):
                response = reader.get(f"v1/prompts/{name}/tags/{tag}")
                response.raise_for_status()
                assert response.json()["data"]["id"] == version
            backend.delete_prompt(name)
            assert reader.get(f"v1/prompts/{name}/latest").status_code == 404
        finally:
            backend.before_shutdown()


async def test_file_prompt_manager_creates_and_pulls_a_remote_prompt(tmp_path, monkeypatch):
    from langgraph_agent_toolkit.core.prompts.prompt_manager import PromptManager
    from langgraph_agent_toolkit.core.settings import settings

    names = ("BASE_URL", "SERVER_VERSION")
    missing = [f"LAT_TEST_PHOENIX_{name}" for name in names if not os.environ.get(f"LAT_TEST_PHOENIX_{name}")]
    if missing:
        pytest.fail("Set the Phoenix test server configuration: " + ", ".join(missing))
    endpoint = os.environ["LAT_TEST_PHOENIX_BASE_URL"].rstrip("/")
    monkeypatch.setattr(settings, "PHOENIX_ENDPOINT", endpoint)
    monkeypatch.setattr(settings, "PHOENIX_PROJECT_NAME", "lat-test-file-prompts")
    name = "lat-test-file-prompt-" + uuid4().hex
    (tmp_path / "system.txt").write_text("Answer the request from {name}.")

    class FilePrompts(PromptManager):
        async def get_system_prompt(self):
            return await self._aget_or_create_prompt(
                name,
                "system.txt",
                input_variables=["name"],
                has_messages_placeholder=False,
            )

    manager = FilePrompts(
        observability_backend="phoenix",
        prompts_dir=tmp_path,
        template_format="f-string",
        prompt_metadata={
            "model_name": "lat-test-calculator",
            "model_provider": "OPENAI",
            "template_format": "F_STRING",
        },
    )
    try:
        prompt = await manager.get_system_prompt()
        assert prompt.format_messages(name="user-1")[0].content == "Answer the request from user-1."
        with httpx.Client(base_url=endpoint + "/", timeout=10, trust_env=False) as reader:
            response = reader.get(f"v1/prompts/{name}/latest")
            response.raise_for_status()
            native = response.json()["data"]
            assert native["model_name"] == "lat-test-calculator"
            assert native["template"]["messages"] == [{"role": "system", "content": "Answer the request from {name}."}]
        manager.observability.delete_prompt(name)
    finally:
        manager.observability.before_shutdown()
