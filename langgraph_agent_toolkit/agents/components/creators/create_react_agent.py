from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    Sequence,
    Type,
    Union,
    cast,
    get_type_hints,
)

from langchain.chat_models.base import _ConfigurableModel
from langchain_core.language_models import (
    BaseChatModel,
    LanguageModelInput,
    LanguageModelLike,
)
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import (
    Runnable,
    RunnableBinding,
    RunnableConfig,
    RunnableSequence,
)
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt.chat_agent_executor import (
    AgentState,
    AgentStateWithStructuredResponse,
    StructuredResponseSchema,
    _get_prompt_runnable,
    _get_state_value,
    _should_bind_tools,
    _validate_chat_history,
)
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer, Send
from langgraph.utils.runnable import RunnableCallable, RunnableLike
from pydantic import BaseModel

from langgraph_agent_toolkit.agents.components.utils import default_pre_model_hook
from langgraph_agent_toolkit.helper.utils import sanitize_chat_history


def _get_model(model: LanguageModelLike, config: RunnableConfig) -> BaseChatModel:
    """Get the underlying model from a RunnableBinding or return the model itself."""
    if isinstance(model, _ConfigurableModel):
        return model._model(config)

    if isinstance(model, RunnableSequence):
        model = next(
            (step for step in model.steps if isinstance(step, (RunnableBinding, BaseChatModel))),
            model,
        )

    if isinstance(model, RunnableBinding):
        model = model.bound

    if not isinstance(model, BaseChatModel):
        raise TypeError(
            f"Expected `model` to be a ChatModel or RunnableBinding (e.g. model.bind_tools(...)), got {type(model)}"
        )

    return model


def create_react_agent(
    model: Union[str, LanguageModelLike],
    tools: Union[Sequence[Union[BaseTool, Callable]], ToolNode],
    *,
    prompt: Optional[
        Union[SystemMessage, str, Callable[[Any], LanguageModelInput], Runnable[Any, LanguageModelInput]]
    ] = None,
    response_format: Optional[Union[StructuredResponseSchema, tuple[str, StructuredResponseSchema]]] = None,
    pre_model_hook: Optional[RunnableLike] = None,
    state_schema: Optional[Type[Any]] = None,
    config_schema: Optional[Type[Any]] = None,
    checkpointer: Optional[Checkpointer] = None,
    store: Optional[BaseStore] = None,
    interrupt_before: Optional[list[str]] = None,
    interrupt_after: Optional[list[str]] = None,
    debug: bool = False,
    version: Literal["v1", "v2"] = "v1",
    name: Optional[str] = None,
    immediate_step_threshold: int = 5,
    immediate_generation_prompt: Optional[str] = None,
) -> CompiledStateGraph:
    """Create a graph that works with a chat model that utilizes tool calling with an additional router.

    This implementation extends the original create_react_agent by adding a router node
    that checks remaining steps and routes to either the agent or an immediate generation
    node when the remaining steps are below a threshold.

    Args:
        model: The `LangChain` chat model that supports tool calling.
        tools: A list of tools or a ToolNode instance.
        prompt: An optional prompt for the LLM.
        response_format: An optional schema for the final agent output.
        pre_model_hook: An optional node to add before the `agent` node.
        state_schema: An optional state schema that defines graph state.
        config_schema: An optional schema for configuration.
        checkpointer: An optional checkpoint saver object.
        store: An optional store object.
        interrupt_before: An optional list of node names to interrupt before.
        interrupt_after: An optional list of node names to interrupt after.
        debug: A flag indicating whether to enable debug mode.
        version: Determines the version of the graph to create ('v1' or 'v2').
        name: An optional name for the CompiledStateGraph.
        immediate_step_threshold: Number of remaining steps below which the router will use immediate generation.
        immediate_generation_prompt: Optional custom prompt for the immediate generation mode.
            If not provided, a default prompt will be used instructing the model to generate a direct answer.

    Returns:
        A compiled LangChain runnable that can be used for chat interactions.

    """
    if version not in ("v1", "v2"):
        raise ValueError(f"Invalid version {version}. Supported versions are 'v1' and 'v2'.")

    if state_schema is not None:
        required_keys = {"messages", "remaining_steps"}
        if response_format is not None:
            required_keys.add("structured_response")

        schema_keys = set(get_type_hints(state_schema))
        if missing_keys := required_keys - set(schema_keys):
            raise ValueError(f"Missing required key(s) {missing_keys} in state_schema")

    if state_schema is None:
        state_schema = AgentStateWithStructuredResponse if response_format is not None else AgentState

    if isinstance(tools, ToolNode):
        tool_classes = list(tools.tools_by_name.values())
        tool_node = tools
    else:
        tool_node = ToolNode(tools)
        # get the tool functions wrapped in a tool class from the ToolNode
        tool_classes = list(tool_node.tools_by_name.values())

    if isinstance(model, str):
        try:
            from langchain.chat_models import (  # type: ignore[import-not-found]
                init_chat_model,
            )
        except ImportError:
            raise ImportError(
                "Please install langchain (`pip install langchain`) to use '<provider>:<model>' "
                "string syntax for `model` parameter."
            )

        model = cast(BaseChatModel, init_chat_model(model))

    tool_calling_enabled = len(tool_classes) > 0

    if _should_bind_tools(model, tool_classes) and tool_calling_enabled:
        model = cast(BaseChatModel, model).bind_tools(tool_classes)

    model_runnable = _get_prompt_runnable(prompt) | model

    # If any of the tools are configured to return_directly after running,
    # our graph needs to check if these were called
    should_return_direct = {t.name for t in tool_classes if t.return_direct}

    def _are_more_steps_needed(state: Any, response: BaseMessage) -> bool:
        has_tool_calls = isinstance(response, AIMessage) and response.tool_calls
        all_tools_return_direct = (
            all(call["name"] in should_return_direct for call in response.tool_calls)
            if isinstance(response, AIMessage)
            else False
        )
        remaining_steps = _get_state_value(state, "remaining_steps", None)
        is_last_step = _get_state_value(state, "is_last_step", False)
        return (
            (remaining_steps is None and is_last_step and has_tool_calls)
            or (remaining_steps is not None and remaining_steps < 1 and all_tools_return_direct)
            or (remaining_steps is not None and remaining_steps < 2 and has_tool_calls)
        )

    def _get_model_input_state(state: Any) -> Any:
        if pre_model_hook is not None:
            messages = (_get_state_value(state, "llm_input_messages")) or _get_state_value(state, "messages")
            error_msg = f"Expected input to call_model to have 'llm_input_messages' or 'messages' key, but got {state}"
        else:
            messages = _get_state_value(state, "messages")
            error_msg = f"Expected input to call_model to have 'messages' key, but got {state}"

        if messages is None:
            raise ValueError(error_msg)

        # Sanitize chat history to remove incomplete tool calls before validation
        # This handles cases where previous executions were interrupted
        messages = sanitize_chat_history(messages)

        _validate_chat_history(messages)
        # we're passing messages under `messages` key, as this is expected by the prompt
        if isinstance(state_schema, type) and issubclass(state_schema, BaseModel):
            state.messages = messages  # type: ignore
        else:
            state["messages"] = messages  # type: ignore

        return state

    # Define the function that calls the model
    def call_model(state: Any, config: RunnableConfig) -> Any:
        state = _get_model_input_state(state)
        response = cast(AIMessage, model_runnable.invoke(state, config))
        # add agent name to the AIMessage
        response.name = name

        if _are_more_steps_needed(state, response):
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="Sorry, need more steps to process this request.",
                    )
                ]
            }
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}

    async def acall_model(state: Any, config: RunnableConfig) -> Any:
        state = _get_model_input_state(state)
        response = cast(AIMessage, await model_runnable.ainvoke(state, config))
        # add agent name to the AIMessage
        response.name = name
        if _are_more_steps_needed(state, response):
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="Sorry, need more steps to process this request.",
                    )
                ]
            }
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}

    # Define the immediate generation function - similar to call_model but with a prompt
    # that instructs the model to avoid tool calls and generate a direct response
    def immediate_generation(state: Any, config: RunnableConfig) -> Any:
        state = _get_model_input_state(state)
        # Create a special system message that instructs the model to give a direct answer
        default_prompt = (
            "You need to generate a direct answer based on the information you already have. "
            "DO NOT make any tool calls. Synthesize what you know and respond directly."
        )
        prompt_content = immediate_generation_prompt or default_prompt
        immediate_prompt = SystemMessage(content=prompt_content)

        messages = _get_state_value(state, "messages")
        prompt_with_instruction = [immediate_prompt] + list(messages)

        # Use the model directly without tool calling capabilities
        base_model = _get_model(model, config)
        response = cast(AIMessage, base_model.invoke(prompt_with_instruction, config))
        response.name = name

        return {"messages": [response]}

    async def aimmediate_generation(state: Any, config: RunnableConfig) -> Any:
        state = _get_model_input_state(state)
        default_prompt = (
            "You need to generate a direct answer based on the information you already have. "
            "DO NOT make any tool calls. Synthesize what you know and respond directly."
        )
        prompt_content = immediate_generation_prompt or default_prompt
        immediate_prompt = SystemMessage(content=prompt_content)

        messages = _get_state_value(state, "messages")
        prompt_with_instruction = [immediate_prompt] + list(messages)

        base_model = _get_model(model, config)
        # Fix: Use ainvoke instead of invoke for async function
        response = cast(AIMessage, await base_model.ainvoke(prompt_with_instruction, config))
        response.name = name

        return {"messages": [response]}

    # Define the router function that checks remaining steps
    def router_condition(state: Any) -> str:
        remaining_steps = _get_state_value(state, "remaining_steps", None)

        # If remaining_steps is below threshold and not None, route to immediate generation
        if remaining_steps is not None and remaining_steps < immediate_step_threshold:
            return "immediate_generation"

        # Otherwise, continue with normal agent flow
        return "agent"

    input_schema = state_schema
    if pre_model_hook is not None:
        # Dynamically create a schema that inherits from state_schema and adds 'llm_input_messages'
        if isinstance(state_schema, type) and issubclass(state_schema, BaseModel):
            # For Pydantic schemas
            from pydantic import create_model

            input_schema = create_model(
                "CallModelInputSchema",
                llm_input_messages=(list[AnyMessage], ...),
                __base__=state_schema,
            )
        else:
            # For TypedDict schemas
            class CallModelInputSchema(state_schema):  # type: ignore
                llm_input_messages: list[AnyMessage]

            input_schema = CallModelInputSchema

    def generate_structured_response(state: Any, config: RunnableConfig) -> Any:
        messages = _get_state_value(state, "messages")
        structured_response_schema = response_format
        if isinstance(response_format, tuple):
            system_prompt, structured_response_schema = response_format
            messages = [SystemMessage(content=system_prompt)] + list(messages)

        model_with_structured_output = _get_model(model, config).with_structured_output(
            cast(StructuredResponseSchema, structured_response_schema),
            strict=True,
            # include_raw=True,
        )
        response = model_with_structured_output.invoke(messages, config)
        return {"structured_response": response}

    async def agenerate_structured_response(state: Any, config: RunnableConfig) -> Any:
        messages = _get_state_value(state, "messages")
        structured_response_schema = response_format
        if isinstance(response_format, tuple):
            system_prompt, structured_response_schema = response_format
            messages = [SystemMessage(content=system_prompt)] + list(messages)

        model_with_structured_output = _get_model(model, config).with_structured_output(
            cast(StructuredResponseSchema, structured_response_schema),
            strict=True,
            # include_raw=True,
        )
        response = await model_with_structured_output.ainvoke(messages, config)
        return {"structured_response": response}

    # Use default_pre_model_hook if pre_model_hook is None
    if pre_model_hook is None:
        pre_model_hook = default_pre_model_hook

    if not tool_calling_enabled:
        # Define a new graph
        workflow = StateGraph(state_schema, config_schema=config_schema)

        # Add nodes for agent and immediate generation
        workflow.add_node(
            "agent",
            RunnableCallable(call_model, acall_model),
            input=input_schema,
        )

        workflow.add_node(
            "immediate_generation",
            RunnableCallable(immediate_generation, aimmediate_generation),
            input=input_schema,
        )

        # Always add pre_model_hook
        workflow.add_node("pre_model_hook", pre_model_hook)

        # Route pre_model_hook directly to either agent or immediate_generation based on condition
        workflow.add_conditional_edges("pre_model_hook", router_condition, ["agent", "immediate_generation"])

        # Always set START as entry point
        workflow.add_edge(START, "pre_model_hook")

        # Connect both agent and immediate_generation to END or structured response
        if response_format is not None:
            workflow.add_node(
                "generate_structured_response",
                RunnableCallable(generate_structured_response, agenerate_structured_response),
            )
            workflow.add_edge("agent", "generate_structured_response")
            workflow.add_edge("immediate_generation", "generate_structured_response")
            workflow.add_edge("generate_structured_response", END)
        else:
            workflow.add_edge("agent", END)
            workflow.add_edge("immediate_generation", END)

        return workflow.compile(
            checkpointer=checkpointer,
            store=store,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            debug=debug,
            name=name,
        )

    # Define the function that determines whether to continue or not
    def should_continue(state: Any) -> Union[str, list]:
        messages = _get_state_value(state, "messages")

        if not messages:
            return END if response_format is None else "generate_structured_response"

        last_message = messages[-1]
        # If there is no function call, then we finish
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return END if response_format is None else "generate_structured_response"
        # Otherwise if there is, we continue
        else:
            if version == "v1":
                return "tools"
            elif version == "v2":
                tool_calls = [tool_node.inject_tool_args(call, state, store) for call in last_message.tool_calls]
                return [Send("tools", [tool_call]) for tool_call in tool_calls]

    # Define a new graph
    workflow = StateGraph(state_schema, config_schema=config_schema)

    # Define the nodes
    workflow.add_node("agent", RunnableCallable(call_model, acall_model), input=input_schema)
    workflow.add_node(
        "immediate_generation", RunnableCallable(immediate_generation, aimmediate_generation), input=input_schema
    )
    workflow.add_node("tools", tool_node)

    # Always add pre_model_hook node
    workflow.add_node("pre_model_hook", pre_model_hook)

    # Route pre_model_hook to either agent or immediate_generation based on condition
    workflow.add_conditional_edges("pre_model_hook", router_condition, ["agent", "immediate_generation"])

    # Set START as the entry point and route to pre_model_hook
    workflow.add_edge(START, "pre_model_hook")

    # Add structured output node if response_format is provided
    if response_format is not None:
        workflow.add_node(
            "generate_structured_response",
            RunnableCallable(generate_structured_response, agenerate_structured_response),
        )
        workflow.add_edge("generate_structured_response", END)
        workflow.add_edge("immediate_generation", "generate_structured_response")
        should_continue_destinations = ["tools", "generate_structured_response"]
    else:
        workflow.add_edge("immediate_generation", END)
        should_continue_destinations = ["tools", END]

    # Add conditional edges from agent
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        should_continue_destinations,
    )

    def route_tool_responses(state: Any) -> str:
        for m in reversed(_get_state_value(state, "messages")):
            if not isinstance(m, ToolMessage):
                break
            if m.name in should_return_direct:
                return END

        # After tools, always go to pre_model_hook
        return "pre_model_hook"

    if should_return_direct:
        workflow.add_conditional_edges("tools", route_tool_responses, ["pre_model_hook", END])
    else:
        # After tools, always go to pre_model_hook
        workflow.add_edge("tools", "pre_model_hook")

    # Finally, we compile it!
    return workflow.compile(
        checkpointer=checkpointer,
        store=store,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
        debug=debug,
        name=name,
    )
