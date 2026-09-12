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
    HumanMessage,
    SystemMessage,
    ToolMessage,
    convert_to_messages,
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
from langgraph.prebuilt.tool_node import ToolCallWithContext, ToolNode
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer, Send
from langgraph.utils.runnable import RunnableCallable, RunnableLike
from pydantic import BaseModel

from langgraph_agent_toolkit.agents.components.utils import default_pre_model_hook
from langgraph_agent_toolkit.helper.utils import sanitize_chat_history


def _get_model(model: LanguageModelLike, config: RunnableConfig) -> BaseChatModel:
    """Return the model in a ``RunnableBinding`` or the model itself."""
    if isinstance(model, _ConfigurableModel):
        model = model._model(config)

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
    """Create a tool-calling chat-model graph with a router.

    The router checks the remaining steps. It routes to the agent or the
    immediate-generation node.

    Args:
        model: The `LangChain` chat model that supports tool calling.
        tools: A list of tools or a ToolNode instance.
        prompt: An optional prompt for the model.
        response_format: An optional schema for the final agent output.
        pre_model_hook: An optional node before the `agent` node.
        state_schema: An optional schema that defines the graph state.
        config_schema: An optional schema for configuration.
        checkpointer: An optional checkpoint saver object.
        store: An optional store object.
        interrupt_before: An optional list of node names to interrupt before.
        interrupt_after: An optional list of node names to interrupt after.
        debug: Enable debug mode.
        version: The graph version: `v1` or `v2`.
        name: An optional name for the `CompiledStateGraph`.
        immediate_step_threshold: Use immediate generation below this number of remaining steps.
        immediate_generation_prompt: An optional prompt for immediate generation.
            The default prompt tells the model to give a direct answer.

    Returns:
        A compiled LangChain runnable for chat interactions.

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

    prompt_runnable = _get_prompt_runnable(prompt)
    model_runnable = prompt_runnable | model

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

        messages = sanitize_chat_history(messages)

        _validate_chat_history(messages)
        if isinstance(state_schema, type) and issubclass(state_schema, BaseModel):
            state.messages = messages  # type: ignore
        else:
            state["messages"] = messages  # type: ignore

        return state

    def call_model(state: Any, config: RunnableConfig) -> Any:
        state = _get_model_input_state(state)
        response = cast(AIMessage, model_runnable.invoke(state, config))
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
        return {"messages": [response]}

    async def acall_model(state: Any, config: RunnableConfig) -> Any:
        state = _get_model_input_state(state)
        response = cast(AIMessage, await model_runnable.ainvoke(state, config))
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
        return {"messages": [response]}

    def add_immediate_instruction(model_input: LanguageModelInput) -> list[BaseMessage]:
        """Keep the configured prompt and add the final-answer instruction."""
        default_prompt = (
            "You need to generate a direct answer based on the information you already have. "
            "DO NOT make any tool calls. Synthesize what you know and respond directly."
        )
        prompt_content = immediate_generation_prompt or default_prompt
        messages = (
            [HumanMessage(content=model_input)] if isinstance(model_input, str) else convert_to_messages(model_input)
        )
        if messages and isinstance(messages[0], SystemMessage):
            content = messages[0].content
            merged = (
                f"{content}\n\n{prompt_content}"
                if isinstance(content, str)
                else [*content, {"type": "text", "text": prompt_content}]
            )
            return [messages[0].model_copy(update={"content": merged}), *messages[1:]]
        return [SystemMessage(content=prompt_content), *messages]

    def immediate_generation(state: Any, config: RunnableConfig) -> Any:
        state = _get_model_input_state(state)
        prompt_with_instruction = add_immediate_instruction(prompt_runnable.invoke(state, config))

        base_model = _get_model(model, config)
        response = cast(AIMessage, base_model.invoke(prompt_with_instruction, config))
        response.name = name

        return {"messages": [response]}

    async def aimmediate_generation(state: Any, config: RunnableConfig) -> Any:
        state = _get_model_input_state(state)
        prompt_with_instruction = add_immediate_instruction(await prompt_runnable.ainvoke(state, config))

        base_model = _get_model(model, config)
        response = cast(AIMessage, await base_model.ainvoke(prompt_with_instruction, config))
        response.name = name

        return {"messages": [response]}

    def router_condition(state: Any) -> str:
        remaining_steps = _get_state_value(state, "remaining_steps", None)

        if remaining_steps is not None and remaining_steps < immediate_step_threshold:
            return "immediate_generation"

        return "agent"

    input_schema = state_schema
    if pre_model_hook is not None:
        if isinstance(state_schema, type) and issubclass(state_schema, BaseModel):
            from pydantic import create_model

            input_schema = create_model(
                "CallModelInputSchema",
                llm_input_messages=(list[AnyMessage], ...),
                __base__=state_schema,
            )
        else:

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
        )
        response = await model_with_structured_output.ainvoke(messages, config)
        return {"structured_response": response}

    if pre_model_hook is None:
        pre_model_hook = default_pre_model_hook

    if not tool_calling_enabled:
        workflow = StateGraph(state_schema, context_schema=config_schema)

        workflow.add_node(
            "agent",
            RunnableCallable(call_model, acall_model),
            input_schema=input_schema,
        )

        workflow.add_node(
            "immediate_generation",
            RunnableCallable(immediate_generation, aimmediate_generation),
            input_schema=input_schema,
        )

        workflow.add_node("pre_model_hook", pre_model_hook)

        workflow.add_conditional_edges("pre_model_hook", router_condition, ["agent", "immediate_generation"])

        workflow.add_edge(START, "pre_model_hook")

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

    def should_continue(state: Any) -> Union[str, list]:
        messages = _get_state_value(state, "messages")

        if not messages:
            return END if response_format is None else "generate_structured_response"

        last_message = messages[-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return END if response_format is None else "generate_structured_response"
        else:
            if version == "v1":
                return "tools"
            elif version == "v2":
                return [
                    Send(
                        "tools",
                        ToolCallWithContext(__type="tool_call_with_context", tool_call=call, state=state),
                    )
                    for call in last_message.tool_calls
                ]

    workflow = StateGraph(state_schema, context_schema=config_schema)

    workflow.add_node("agent", RunnableCallable(call_model, acall_model), input_schema=input_schema)
    workflow.add_node(
        "immediate_generation", RunnableCallable(immediate_generation, aimmediate_generation), input_schema=input_schema
    )
    workflow.add_node("tools", tool_node)

    workflow.add_node("pre_model_hook", pre_model_hook)

    workflow.add_conditional_edges("pre_model_hook", router_condition, ["agent", "immediate_generation"])

    workflow.add_edge(START, "pre_model_hook")

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

        return "pre_model_hook"

    if should_return_direct:
        workflow.add_conditional_edges("tools", route_tool_responses, ["pre_model_hook", END])
    else:
        workflow.add_edge("tools", "pre_model_hook")

    return workflow.compile(
        checkpointer=checkpointer,
        store=store,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
        debug=debug,
        name=name,
    )
