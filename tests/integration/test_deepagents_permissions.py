"""Check native file permissions and optional context sources in the toolkit graph."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command


pytest.importorskip("deepagents")

from deepagents import FilesystemPermission
from deepagents.backends import FilesystemBackend
from test_deepagents import ScriptedModel, call, tool_results

from langgraph_agent_toolkit.agents.blueprints.deep_agent.agent import build_graph


@pytest.mark.parametrize("path", ["/private/note.txt", "/public-note.txt"])
async def test_file_denial_survives_middleware_replacement_and_symlink_resolution(tmp_path, path):
    private = tmp_path / "private"
    private.mkdir()
    note = private / "note.txt"
    note.write_text("The confidential value.")
    (tmp_path / "public-note.txt").symlink_to(note)
    model = ScriptedModel(responses=[call("read_file", file_path=path), AIMessage("Read denied.")])
    graph = build_graph(
        model=model,
        backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
        permissions=[FilesystemPermission(operations=["read"], paths=["/private/**"], mode="deny")],
    )

    result = await graph.ainvoke({"messages": [HumanMessage("Read the requested file.")]})

    reads = tool_results(result["messages"], "read_file")
    assert len(reads) == 1
    assert reads[0].status == "error"
    assert "permission denied" in reads[0].content.lower()
    assert "The confidential value." not in str(model.seen)
    assert note.read_text() == "The confidential value."
    assert all("execute" not in names and "delete" not in names for names in model.bound_tool_names)


async def test_symlink_target_requires_approval_before_a_file_edit(tmp_path):
    private = tmp_path / "private"
    private.mkdir()
    note = private / "note.txt"
    note.write_text("Original approved text.")
    (tmp_path / "public-note.txt").symlink_to(note)
    model = ScriptedModel(
        responses=[
            call(
                "edit_file",
                file_path="/public-note.txt",
                old_string="Original approved text.",
                new_string="Changed approved text.",
            ),
            AIMessage("Approved edit complete."),
        ]
    )
    graph = build_graph(
        model=model,
        backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
        permissions=[FilesystemPermission(operations=["write"], paths=["/private/**"], mode="interrupt")],
    )
    graph.checkpointer = InMemorySaver()
    config = {"configurable": {"thread_id": "permission-approval"}}

    paused = await graph.ainvoke({"messages": [HumanMessage("Edit the requested file.")]}, config)

    assert paused["__interrupt__"][0].value["action_requests"][0]["name"] == "edit_file"
    assert note.read_text() == "Original approved text."
    assert not tool_results(paused["messages"], "edit_file")

    completed = await graph.ainvoke(Command(resume={"decisions": [{"type": "approve"}]}), config)

    assert note.read_text() == "Changed approved text."
    edits = tool_results(completed["messages"], "edit_file")
    assert len(edits) == 1
    assert edits[0].status == "success"
    assert completed["messages"][-1].content == "Approved edit complete."


async def test_subagents_inherit_file_denials(tmp_path):
    (tmp_path / "private.txt").write_text("The private subagent value.")
    specialist = ScriptedModel(
        responses=[call("read_file", file_path="/private.txt"), AIMessage("The file access was denied.")]
    )
    parent = ScriptedModel(
        responses=[
            call("task", subagent_type="specialist", description="Read the requested file."),
            AIMessage("The specialist completed the attempt."),
        ]
    )
    graph = build_graph(
        model=parent,
        backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
        permissions=[FilesystemPermission(operations=["read"], paths=["/private.txt"], mode="deny")],
        subagents=[{"name": "specialist", "description": "Inspect requested files.", "model": specialist}],
    )

    completed = await graph.ainvoke({"messages": [HumanMessage("Delegate the file read.")]})

    reads = tool_results(specialist.seen[-1], "read_file")
    assert len(reads) == 1
    assert reads[0].status == "error"
    assert "The private subagent value." not in str(parent.seen) + str(specialist.seen)
    assert completed["messages"][-1].content == "The specialist completed the attempt."


async def test_skills_and_memory_use_the_selected_backend(tmp_path):
    skill = tmp_path / "skills" / "report-check"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: report-check\ndescription: Check report totals before delivery.\n---\n"
        "Read the report and check each total.\n"
    )
    (tmp_path / "AGENTS.md").write_text("The report currency is EUR.")
    model = ScriptedModel(responses=[AIMessage("Context loaded.")])
    graph = build_graph(
        model=model,
        backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
        skills=["/skills/"],
        memory=["/AGENTS.md"],
    )

    result = await graph.ainvoke({"messages": [HumanMessage("Prepare the report.")]})

    system_prompt = str(model.seen[0][0].content)
    assert "The report currency is EUR." in system_prompt
    assert "Check report totals before delivery." in system_prompt
    assert "/skills/report-check/SKILL.md" in system_prompt
    assert result["messages"][-1].content == "Context loaded."


async def test_async_subagent_registration_does_not_connect_to_a_remote_server():
    model = ScriptedModel(responses=[AIMessage("No background work is needed.")])
    graph = build_graph(
        model=model,
        subagents=[
            {
                "name": "remote-review",
                "description": "Review a report in the background.",
                "graph_id": "report-review",
                "url": "http://127.0.0.1:1",
                "mode": "isolated",
                "update_strategy": "enqueue",
                "notify_parent": False,
            }
        ],
    )

    result = await graph.ainvoke({"messages": [HumanMessage("Reply without delegation.")]})

    expected = {"start_async_task", "check_async_task", "update_async_task", "cancel_async_task", "list_async_tasks"}
    assert model.bound_tool_names and all(expected <= names for names in model.bound_tool_names)
    assert all("execute" not in names for names in model.bound_tool_names)
    assert result["messages"][-1].content == "No background work is needed."
