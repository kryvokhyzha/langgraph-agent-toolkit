import functools
import importlib
import inspect
import os
import traceback
from contextlib import aclosing
from copy import copy
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple, TypeVar
from uuid import UUID, uuid4

import joblib
from langchain_core.exceptions import ModelAuthenticationError
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
    convert_to_messages,
)
from langchain_core.runnables import RunnableConfig
from langgraph._internal._constants import PREVIOUS
from langgraph.constants import END, START
from langgraph.errors import GraphRecursionError
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.pregel import Pregel
from langgraph.types import Command, Interrupt

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.core.memory.concurrency import ConversationCoordinator, serialize_execution
from langgraph_agent_toolkit.core.observability.empty import EmptyObservability
from langgraph_agent_toolkit.core.settings import settings
from langgraph_agent_toolkit.helper.constants import get_default_agent, set_default_agent
from langgraph_agent_toolkit.helper.exceptions import InputValidationError
from langgraph_agent_toolkit.helper.logging import logger
from langgraph_agent_toolkit.helper.utils import (
    convert_message_content_to_string,
    create_ai_message,
    langchain_to_chat_message,
    remove_tool_calls,
)
from langgraph_agent_toolkit.schema import AgentInfo, ChatMessage


_HITL_APPROVE = {"approve", "yes", "y", "ok", "accept", "approved"}
_HITL_REJECT = {"reject", "no", "n", "deny", "rejected"}


def _uses_functional_history(graph: Pregel) -> bool:
    """Identify the Functional API checkpoint layout.

    LangGraph stores `entrypoint.final.save` in its private `PREVIOUS` channel.
    Keep this dependency contract covered when the LangGraph version changes.
    """
    return graph.input_channels == START and graph.output_channels == END and PREVIOUS in graph.channels


async def _graph_history_values(graph: Pregel, config: RunnableConfig) -> dict[str, Any]:
    """Read saved conversation state without changing the graph output."""
    state = await graph.aget_state(config)
    if not _uses_functional_history(graph):
        return state.values
    checkpoint = await graph.checkpointer.aget_tuple(state.config)
    values = checkpoint.checkpoint["channel_values"].get(PREVIOUS) if checkpoint else None
    if values is None:
        return {}
    if not isinstance(values, dict) or not isinstance(values.get("messages", []), list):
        raise InputValidationError("Functional agent history must save an object with a messages list.")
    return values


async def get_graph_history(graph: Pregel, config: RunnableConfig) -> list[BaseMessage]:
    """Read messages from StateGraph state or Functional API saved state."""
    values = await _graph_history_values(graph, config)
    return convert_to_messages(values.get("messages", []))


async def add_graph_history(graph: Pregel, config: RunnableConfig, messages: list[Any]) -> None:
    """Append messages while preserving Functional API saved state."""
    if _uses_functional_history(graph):
        values = dict(await _graph_history_values(graph, config))
        values["messages"] = add_messages(values.get("messages", []), messages)
    else:
        values = {"messages": messages}
    await graph.aupdate_state(config=config, values=values)


def interrupt_value_to_content(value: Any) -> Any:
    """Convert an interrupt payload to valid ``AIMessage`` content.

    Custom ``interrupt()`` blueprints pass a string. The function returns that
    string unchanged. ``HumanInTheLoopMiddleware`` passes a request dictionary.
    The function joins the action descriptions and adds reply instructions. This
    prevents an invalid dictionary value in ``AIMessage(content=...)``.
    """
    if isinstance(value, (str, list)):
        return value
    if _is_mcp_elicitation(value):
        lines = []
        for request in value.get("requests", []):
            lines.append(str(request.get("message", "The MCP tool needs more information.")))
            if request.get("mode") == "url":
                lines.append(str(request.get("url", "")))
        lines.append("\nSend answers in input.responses. Use input.resume for multiple pending interrupts.")
        return "\n".join(lines)
    if isinstance(value, dict) and value.get("action_requests"):
        lines = []
        for req in value["action_requests"]:
            description = req.get("description")
            lines.append(
                str(description) if description else f"Approve `{req.get('name')}` with args {req.get('args')}?"
            )
        lines.append("\nReply 'approve' to proceed, 'reject: <reason>' to decline, or send other instructions.")
        return "\n".join(lines)
    return str(value)


def _is_mcp_elicitation(value: Any) -> bool:
    """Check the MCP elicitation discriminator without loading optional packages."""
    return isinstance(value, dict) and value.get("type") == "mcp_elicitation"


def interrupts_to_chat_message(interrupts: list[Interrupt]) -> ChatMessage:
    """Keep every interrupt ID and payload in the response."""
    contents = [interrupt_value_to_content(interrupt.value) for interrupt in interrupts]
    content = contents[0] if len(contents) == 1 else "\n\n".join(str(value) for value in contents)
    return ChatMessage(
        type="ai",
        content=content,
        custom_data={
            "interrupts": [
                {
                    "id": interrupt.id if isinstance(getattr(interrupt, "id", None), str) else None,
                    "value": interrupt.value,
                }
                for interrupt in interrupts
            ]
        },
    )


def _validate_mcp_resume(value: dict[str, Any], resume: Any) -> dict[str, Any]:
    """Validate the answer envelope and each MCP form value."""
    if not isinstance(resume, dict) or set(resume) != {"responses"}:
        raise InputValidationError("Each MCP resume value must contain only a responses object.")
    responses = resume["responses"]
    requests = value.get("requests")
    if (
        not isinstance(requests, list)
        or not requests
        or any(not isinstance(request, dict) or not isinstance(request.get("key"), str) for request in requests)
    ):
        raise InputValidationError("The pending MCP interrupt has invalid request keys.")
    expected_keys = {request["key"] for request in requests}
    if not isinstance(responses, dict) or set(responses) != expected_keys:
        raise InputValidationError("MCP responses must contain one answer for each pending request key.")

    for request in requests:
        answer = responses[request["key"]]
        if not isinstance(answer, dict) or set(answer) - {"action", "content"}:
            raise InputValidationError("Each MCP answer must contain an action and optional content.")
        action = answer.get("action")
        if not isinstance(action, str) or action not in {"accept", "decline", "cancel"}:
            raise InputValidationError("An MCP answer action must be accept, decline, or cancel.")
        content = answer.get("content")
        if action != "accept" or request.get("mode") == "url":
            if content is not None:
                raise InputValidationError("Only an accepted MCP form answer can contain content.")
            continue
        if not isinstance(content, dict) or any(
            not isinstance(key, str)
            or not (
                item is None
                or isinstance(item, (str, int, float, bool))
                or isinstance(item, list)
                and all(isinstance(element, str) for element in item)
            )
            for key, item in content.items()
        ):
            raise InputValidationError("An accepted MCP form answer must contain an object with valid form values.")
    return resume


def build_resume_command(interrupted_tasks: list, user_input: Dict[str, Any]) -> Command:
    """Build the ``Command(resume=...)`` for an interrupted run.

    ``HumanInTheLoopMiddleware`` expects ``{"decisions": [...]}``. Other
    ``interrupt()`` blueprints read the input dictionary. For a HITL tool
    approval request, translate the user's reply to a decision for each pending
    tool call. Otherwise, return the input dictionary unchanged.
    """
    pending = [interrupt for task in interrupted_tasks for interrupt in (getattr(task, "interrupts", None) or [])]
    mcp_pending = [interrupt for interrupt in pending if _is_mcp_elicitation(interrupt.value)]
    if mcp_pending:
        if "resume" in user_input:
            resume_map = user_input["resume"]
            pending_ids = {getattr(interrupt, "id", None) for interrupt in pending}
            if "responses" in user_input:
                raise InputValidationError("Send input.resume or input.responses, not both.")
            if not all(isinstance(interrupt_id, str) and interrupt_id for interrupt_id in pending_ids):
                raise InputValidationError("The pending interrupts do not have valid IDs.")
            if not isinstance(resume_map, dict) or set(resume_map) != pending_ids:
                raise InputValidationError("input.resume must contain one value for each pending interrupt ID.")
            for interrupt in mcp_pending:
                _validate_mcp_resume(interrupt.value, resume_map[interrupt.id])
            return Command(resume=resume_map)
        if len(pending) != 1:
            raise InputValidationError("Multiple interrupts are pending. Send answers in input.resume by interrupt ID.")
        return Command(resume=_validate_mcp_resume(mcp_pending[0].value, {"responses": user_input.get("responses")}))

    interrupt_value = pending[0].value if pending else None

    if isinstance(interrupt_value, dict) and interrupt_value.get("action_requests"):
        count = len(interrupt_value["action_requests"]) or 1
        raw = user_input.get("message")
        message = raw.strip() if isinstance(raw, str) else ""
        lowered = message.lower()
        if lowered in _HITL_APPROVE:
            decision: Dict[str, Any] = {"type": "approve"}
        elif lowered in _HITL_REJECT or lowered.startswith("reject"):
            reason = message.split(":", 1)[1].strip() if ":" in message else "User rejected the action."
            decision = {"type": "reject", "message": reason}
        else:
            decision = {"type": "respond", "message": message}
        return Command(resume={"decisions": [decision] * count})

    return Command(resume=user_input)


T = TypeVar("T")


class AgentExecutor:
    """Load, run, and save LangGraph agents."""

    def __init__(self, *args):
        """Initialize the `AgentExecutor` and import agents.

        Args:
            *args: Import strings for the agents. Example:
                "langgraph_agent_toolkit.agents.blueprints.react.agent:react_agent".

        Raises:
            ValueError: If no agents are provided.

        """
        self.agents: Dict[str, Agent] = {}
        self.concurrency = ConversationCoordinator()

        if not args:
            raise ValueError("At least one agent must be provided to AgentExecutor.")

        self.load_agents_from_imports(args)
        self._validate_default_agent_loaded()

    def load_agents_from_imports(self, args: tuple) -> None:
        """Import agents from the specified import strings."""
        errors = []
        for import_str in args:
            try:
                module_path, object_name = import_str.split(":")
                module = importlib.import_module(module_path)
                agent_obj = getattr(module, object_name)

                if isinstance(agent_obj, (CompiledStateGraph, Pregel)):
                    agent = Agent(
                        name=object_name, description=f"Dynamically loaded {object_name}", graph=copy(agent_obj)
                    )
                    self.agents[agent.name] = agent
                elif isinstance(agent_obj, Agent):
                    agent = copy(agent_obj)
                    agent.graph = copy(agent_obj.graph)
                    self.agents[agent.name] = agent
                else:
                    raise ValueError(f"Object '{object_name}' is neither a graph nor an Agent instance")
            except (ImportError, AttributeError, ValueError) as e:
                logger.error(f"Error loading agent from '{import_str}': {e}")
                errors.append(import_str)
        if errors:
            raise ValueError(f"Required agents failed to load: {', '.join(errors)}")

    def _validate_default_agent_loaded(self) -> None:
        """Validate the configured default agent.

        Use the first loaded agent if the configured default is unavailable.
        """
        if not self.agents:
            raise ValueError("No agents were loaded. Please check your imports.")

        configured_default = get_default_agent()

        if configured_default in self.agents:
            logger.debug(f"Default agent '{configured_default}' is available in loaded agents.")
            return

        new_default = list(self.agents.keys())[0]
        logger.warning(
            f"Default agent '{configured_default}' not found in loaded agents. Using '{new_default}' as default."
        )
        set_default_agent(new_default)

    def get_agent(self, agent_id: str) -> Agent:
        """Return the agent with the specified ID.

        Args:
            agent_id: The ID of the agent.

        Returns:
            The requested `Agent` instance.

        Raises:
            KeyError: The agent ID is not found.

        """
        if agent_id not in self.agents:
            raise KeyError(f"Agent '{agent_id}' not found")
        return self.agents[agent_id]

    def get_all_agent_info(self) -> list[AgentInfo]:
        """Return information about all available agents.

        Returns:
            `AgentInfo` objects with agent IDs and descriptions.

        """
        return [AgentInfo(key=agent_id, description=agent.description) for agent_id, agent in self.agents.items()]

    def add_agent(self, agent_id: str, agent: Agent) -> None:
        """Add an agent to the executor.

        Args:
            agent_id: The ID for the agent.
            agent: The `Agent` instance.

        """
        self.agents[agent_id] = agent

    @staticmethod
    def handle_agent_errors(func: Callable[..., T]) -> Callable[..., T]:
        """Handle errors during agent execution.

        Handle `GraphRecursionError` and other exceptions.

        Args:
            func: The function to decorate.

        Returns:
            The decorated function.

        """

        def _handle_error(e: Exception):
            """Log an error and raise it again."""
            if isinstance(e, ModelAuthenticationError):
                logger.warning("The model provider credentials were rejected")
                raise e
            tb_str = traceback.format_exc()

            if isinstance(e, GraphRecursionError):
                logger.error(f"GraphRecursionError occurred: {e}\n\nFull traceback:\n{tb_str}")
            else:
                logger.error(f"Error during agent execution: {e}\n\nFull traceback:\n{tb_str}")

            raise e

        @functools.wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            try:
                return await func(self, *args, **kwargs)
            except Exception as e:
                return _handle_error(e)

        @functools.wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                return _handle_error(e)

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    async def _setup_agent_execution(
        self,
        agent_id: str,
        input: Dict[str, Any],
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
        model_name: Optional[str] = None,
        model_provider: Optional[str] = None,
        model_config_key: Optional[str] = None,
        agent_config: Optional[Dict[str, Any]] = None,
        recursion_limit: Optional[int] = None,
    ) -> Tuple[Agent, Any, Any, UUID]:
        """Set up an agent run for `invoke` and `stream`.

        Args:
            agent_id: ID of the agent to run.
            input: User message for the agent.
            thread_id: Optional conversation thread ID.
            user_id: Optional user ID.
            model_name: Optional replacement model name.
            model_provider: Optional replacement model provider.
            model_config_key: Optional replacement model configuration key.
            agent_config: Optional agent configuration.
            recursion_limit: Optional limit for graph recursion.

        Returns:
            A tuple that contains:
                - agent: The `Agent` instance.
                - input_data: Formatted input for the agent.
                - config: The `RunnableConfig` for the agent.
                - run_id: The UUID for this run.

        """
        agent = self.get_agent(agent_id)
        agent_graph = agent.graph

        run_id = uuid4()
        thread_id = thread_id or str(uuid4())

        recursion_limit = recursion_limit or settings.DEFAULT_RECURSION_LIMIT

        configurable = {
            "thread_id": thread_id,
            "user_id": user_id,
        }

        if model_config_key and model_config_key in settings.MODEL_CONFIGS:
            configurable["model_config_key"] = model_config_key

            model_config = settings.MODEL_CONFIGS[model_config_key]
            if "provider" in model_config:
                configurable["model_provider"] = model_config["provider"]
            if "name" in model_config:
                configurable["model_name"] = model_config["name"]
        else:
            if model_name:
                configurable["model_name"] = model_name

            if model_provider:
                configurable["model_provider"] = model_provider

        if agent_config:
            reserved = {"thread_id", "user_id", "checkpoint_id", "checkpoint_ns"}
            if reserved.intersection(agent_config) or any(key.startswith("__") for key in agent_config):
                raise ValueError("agent_config cannot override identity or checkpoint fields")
            configurable.update(agent_config)

        if agent.observability is None:
            agent.observability = EmptyObservability()
        callback = agent.observability.get_callback_handler(
            run_id=str(run_id), user_id=user_id, session_id=thread_id, update_trace=True
        )

        config = RunnableConfig(
            configurable=configurable,
            run_id=run_id,
            callbacks=[callback] if callback else None,
            recursion_limit=recursion_limit,
            metadata={
                "langfuse_session_id": thread_id,
                "langfuse_user_id": user_id,
                "langfuse_tags": [agent.name],
            },
        )

        _input = input.model_dump() if hasattr(input, "model_dump") else dict(input)
        input_data: Command | dict[str, Any]

        interrupted_tasks = []
        if settings.CHECK_INTERRUPTS and agent_graph.checkpointer is not None:
            state = await agent_graph.aget_state(config=config)
            interrupted_tasks = [task for task in state.tasks if hasattr(task, "interrupts") and task.interrupts]

        if interrupted_tasks:
            input_data = build_resume_command(interrupted_tasks, _input)
        else:
            if "message" in _input:
                message = _input.pop("message", "") or ""
                input_data = {"messages": [HumanMessage(content=message)], **_input}
            else:
                input_data = _input

        return agent, input_data, config, run_id

    @handle_agent_errors
    @serialize_execution
    async def invoke(
        self,
        agent_id: str,
        input: Dict[str, Any],
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
        model_name: Optional[str] = None,
        model_provider: Optional[str] = None,
        model_config_key: Optional[str] = None,
        agent_config: Optional[Dict[str, Any]] = None,
        recursion_limit: Optional[int] = None,
    ) -> ChatMessage:
        """Run an agent with a message and return its response.

        Args:
            agent_id: ID of the agent to run.
            input: User message for the agent.
            thread_id: Optional conversation thread ID.
            user_id: Optional user ID.
            model_name: Optional replacement model name.
            model_provider: Optional replacement model provider.
            model_config_key: Optional replacement model configuration key.
            agent_config: Optional agent configuration.
            recursion_limit: Optional limit for graph recursion.

        Returns:
            The agent response as a `ChatMessage`.

        """
        agent, input_data, config, run_id = await self._setup_agent_execution(
            agent_id=agent_id,
            input=input,
            thread_id=thread_id,
            user_id=user_id,
            model_name=model_name,
            model_provider=model_provider,
            model_config_key=model_config_key,
            agent_config=agent_config,
            recursion_limit=recursion_limit,
        )

        with agent.observability.trace_context(
            run_id=run_id,
            user_id=user_id,
            session_id=thread_id,
            input=input_data,
            agent_name=agent.name,
        ) as trace_span:
            response_type = None
            response = None
            pending_interrupts = []
            seen_interrupts = set()
            # Keep only the latest state. List-mode ainvoke retains every state.
            async with aclosing(
                agent.graph.astream(
                    input=input_data,
                    config=config,
                    stream_mode=["values"],
                    output_keys=agent.graph.output_channels,
                )
            ) as events:
                async for response_type, response in events:
                    if response_type != "values":
                        continue
                    for pending_interrupt in response.get("__interrupt__", []):
                        key = getattr(pending_interrupt, "id", None) or id(pending_interrupt)
                        if key not in seen_interrupts:
                            seen_interrupts.add(key)
                            pending_interrupts.append(pending_interrupt)

            if response_type is None:
                raise ValueError("Agent returned no response events")

            if pending_interrupts:
                output = interrupts_to_chat_message(pending_interrupts)
            elif response_type == "values" and "__interrupt__" not in response:
                generated_message = response.get("structured_response")
                if not generated_message:
                    messages = response.get("messages") or []
                    if not messages:
                        raise ValueError("Agent response contains no messages")
                    generated_message = messages[-1]

                output = langchain_to_chat_message(generated_message)
            else:
                raise ValueError(f"Unexpected response type: {response_type}")

            output.run_id = str(run_id)
            agent.observability.update_trace(trace_span, output=output.content)
            return output

    @handle_agent_errors
    @serialize_execution
    async def stream(
        self,
        agent_id: str,
        input: Dict[str, Any],
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
        model_name: Optional[str] = None,
        model_provider: Optional[str] = None,
        model_config_key: Optional[str] = None,
        stream_tokens: bool = True,
        agent_config: Optional[Dict[str, Any]] = None,
        recursion_limit: Optional[int] = None,
    ) -> AsyncGenerator[str | ChatMessage, None]:
        """Stream an agent response as tokens or messages.

        Args:
            agent_id: ID of the agent to run.
            input: User message for the agent.
            thread_id: Optional conversation thread ID.
            user_id: Optional user ID.
            model_name: Optional replacement model name.
            model_provider: Optional replacement model provider.
            model_config_key: Optional replacement model configuration key.
            stream_tokens: Stream individual tokens when true.
            agent_config: Optional agent configuration.
            recursion_limit: Optional limit for graph recursion.

        Yields:
            Full `ChatMessage` objects or token strings.

        """
        agent, input_data, config, run_id = await self._setup_agent_execution(
            agent_id=agent_id,
            input=input,
            thread_id=thread_id,
            user_id=user_id,
            model_name=model_name,
            model_provider=model_provider,
            model_config_key=model_config_key,
            agent_config=agent_config,
            recursion_limit=recursion_limit,
        )

        with agent.observability.trace_context(
            run_id=run_id,
            user_id=user_id,
            session_id=thread_id,
            input=input_data,
            agent_name=agent.name,
        ) as trace_span:
            stream_mode = ["updates", "messages", "custom"] if stream_tokens else ["updates"]
            final_output: str | None = None
            pending_interrupts = []
            seen_interrupts = set()

            # Close graph tasks and checkpoint writes before releasing the conversation.
            async with aclosing(
                agent.graph.astream(input=input_data, config=config, stream_mode=stream_mode)
            ) as events:
                async for stream_event in events:
                    if not isinstance(stream_event, tuple):
                        continue

                    stream_mode, event = stream_event
                    new_messages = []

                    if stream_mode == "updates":
                        for node, updates in event.items():
                            # Summary updates replace context and can contain retained answers.
                            if node == "SummarizationMiddleware.before_model":
                                continue
                            if node == "__interrupt__":
                                for pending_interrupt in updates:
                                    key = getattr(pending_interrupt, "id", None) or id(pending_interrupt)
                                    if key not in seen_interrupts:
                                        seen_interrupts.add(key)
                                        pending_interrupts.append(pending_interrupt)
                                continue

                            update_messages = (updates or {}).get("messages", [])

                            if node == "supervisor":
                                ai_messages = [msg for msg in update_messages if isinstance(msg, AIMessage)]
                                if ai_messages:
                                    update_messages = [ai_messages[-1]]

                            if node in ("research_expert", "math_expert"):
                                if update_messages:
                                    msg = ToolMessage(
                                        content=update_messages[0].content,
                                        name=node,
                                        tool_call_id="",
                                    )
                                    update_messages = [msg]
                            new_messages.extend(update_messages)

                            structured_response = (updates or {}).get("structured_response")
                            if structured_response is not None:
                                new_messages.append(structured_response)

                    elif stream_mode == "custom":
                        new_messages = [event]

                    elif stream_mode == "messages" and stream_tokens:
                        msg, metadata = event
                        if "skip_stream" in metadata.get("tags", []) or metadata.get("lc_source") == "summarization":
                            continue
                        if not isinstance(msg, AIMessageChunk):
                            continue
                        content = remove_tool_calls(msg.content)
                        if content:
                            yield convert_message_content_to_string(content)

                    processed_messages = []
                    current_message: dict[str, Any] = {}
                    for msg in new_messages:
                        if isinstance(msg, tuple):
                            key, value = msg
                            current_message[key] = value
                        else:
                            if current_message:
                                processed_messages.append(create_ai_message(current_message))
                                current_message = {}
                            processed_messages.append(msg)

                    if current_message:
                        processed_messages.append(create_ai_message(current_message))

                    for msg in processed_messages:
                        if isinstance(msg, RemoveMessage):
                            continue
                        try:
                            chat_message = langchain_to_chat_message(msg)
                            chat_message.run_id = str(run_id)
                            if chat_message.type == "human":
                                continue
                            if chat_message.type == "ai" and chat_message.content:
                                _content = chat_message.content
                                final_output = (
                                    _content
                                    if isinstance(_content, str)
                                    else convert_message_content_to_string(_content)
                                    if isinstance(_content, list)
                                    else str(_content)
                                )
                            yield chat_message
                        except Exception as e:
                            logger.error(f"Error parsing message: {e}")
                            continue

            if pending_interrupts:
                chat_message = interrupts_to_chat_message(pending_interrupts)
                chat_message.run_id = str(run_id)
                final_output = (
                    chat_message.content
                    if isinstance(chat_message.content, str)
                    else convert_message_content_to_string(chat_message.content)
                )
                yield chat_message

            agent.observability.update_trace(trace_span, output=final_output)

    def save(self, path: str, agent_ids: Optional[List[str]] = None) -> None:
        """Save agents to disk with `joblib`.

        Args:
            path: Directory path for the agent files.
            agent_ids: Agent IDs to save. Save all agents when this is `None`.

        """
        _path = Path(path)
        _path.mkdir(exist_ok=True, parents=True)

        agents_to_save = self.agents
        if agent_ids:
            agents_to_save = {k: v for k, v in self.agents.items() if k in agent_ids}

        for agent_id, agent in agents_to_save.items():
            joblib.dump(agent, _path / f"{agent_id}.joblib")

    def load_saved_agents(self, path: str) -> None:
        """Load agents from `joblib` files on disk.

        Args:
            path: Directory path for the agent files.

        """
        for filename in os.listdir(path):
            if filename.endswith(".joblib"):
                agent = joblib.load(os.path.join(path, filename))
                self.agents[agent.name] = agent

        self._validate_default_agent_loaded()
