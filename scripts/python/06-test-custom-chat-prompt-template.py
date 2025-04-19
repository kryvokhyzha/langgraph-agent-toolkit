import time
from typing import List

from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import create_react_agent

from langgraph_agent_toolkit.core.observability.factory import ObservabilityFactory
from langgraph_agent_toolkit.core.observability.types import ChatMessageDict, ObservabilityBackend
from langgraph_agent_toolkit.core.prompts.chat_prompt_template import ObservabilityChatPromptTemplate


def format_time(seconds: float) -> str:
    """Format time in ms with 2 decimal places."""
    return f"{seconds * 1000:.2f}ms"


def print_separator(title: str = None):
    """Print a separator line with optional title."""
    width = 80
    if title:
        print(f"\n{'=' * 5} {title} {'=' * (width - 7 - len(title))}")
    else:
        print(f"\n{'=' * width}")


# Load environment variables
load_dotenv(find_dotenv())

if __name__ == "__main__":
    print_separator("ObservabilityChatPromptTemplate Demo")

    # Step 1: Create observability platforms
    os_platform = ObservabilityFactory.create(ObservabilityBackend.EMPTY)
    # os_platform = ObservabilityFactory.create(ObservabilityBackend.LANGFUSE)
    # os_platform = ObservabilityFactory.create(ObservabilityBackend.LANGSMITH)

    print("✓ Created observability platforms")

    # Step 2: Create and store some sample prompts
    # Basic system+human prompt
    basic_chat_messages: List[ChatMessageDict] = [
        {"role": "system", "content": "You are an AI assistant specialized in {{ domain }}."},
        {"role": "human", "content": "I need help with {{ question }} related to {{ topic }}."},
    ]
    os_platform.push_prompt("basic-assistant", basic_chat_messages)
    print("✓ Stored basic chat prompt")

    # Comprehensive agent prompt
    agent_chat_messages: List[ChatMessageDict] = [
        {"role": "system", "content": "You are an AI agent that can use tools. Available tools: {{ tools }}"},
        {"role": "human", "content": "I want to {{ task }} with the following context: {{ context }}"},
    ]
    os_platform.push_prompt("tool-using-agent", agent_chat_messages)
    print("✓ Stored tool-using agent prompt")

    # Step 3: Create ObservabilityChatPromptTemplate instances with different configurations
    print_separator("Creating prompt templates")

    # Example 1: Basic template with initialization-time loading
    print("\n1. Template with initialization-time loading:")
    start_time = time.time()
    init_template = ObservabilityChatPromptTemplate.from_observability_platform(
        prompt_name="basic-assistant",
        observability_platform=os_platform,
        load_at_runtime=False,  # Load during initialization
        template_format="jinja2",  # Use jinja2 templating
        input_variables=["domain", "question", "topic"],  # Explicitly define input variables
    )
    load_time = time.time() - start_time
    print(f"   ✓ Created template (load time: {format_time(load_time)})")
    print(f"   ✓ Template has {len(init_template.messages)} message(s)")

    # Example 2: Template with runtime loading
    print("\n2. Template with runtime loading:")
    start_time = time.time()
    runtime_template = ObservabilityChatPromptTemplate.from_observability_platform(
        prompt_name="tool-using-agent",
        observability_platform=os_platform,
        load_at_runtime=True,  # Load during invoke
        template_format="jinja2",  # Use jinja2 templating
        input_variables=["tools", "task", "context"],  # Explicitly define input variables
    )
    create_time = time.time() - start_time
    print(f"   ✓ Created template (create time: {format_time(create_time)})")

    # Example 3: Using factory method with backend
    print("\n3. Template using backend factory method:")
    backend_template = ObservabilityChatPromptTemplate.from_observability_backend(
        prompt_name="basic-assistant",
        observability_backend=ObservabilityBackend.EMPTY,
        load_at_runtime=True,
        template_format="jinja2",  # Use jinja2 templating
        input_variables=["domain", "question", "topic"],  # Explicitly define input variables
    )
    print("   ✓ Created template using backend factory")

    # Example 4: Direct initialization
    print("\n4. Template with direct initialization:")
    direct_template = ObservabilityChatPromptTemplate(
        prompt_name="tool-using-agent",
        observability_backend=ObservabilityBackend.EMPTY,
        load_at_runtime=True,
        template_format="jinja2",  # Specify template format
        cache_ttl_seconds=120,  # Custom cache TTL
        input_variables=["tools", "task", "context"],  # Explicitly define input variables
    )
    print("   ✓ Created template with direct initialization")

    # Step 4: Invoking the templates
    print_separator("Invoking templates")

    # Test the initialization-time loaded template
    print("\n1. Invoking initialization-loaded template:")
    start_time = time.time()
    result1 = init_template.invoke(
        input=dict(
            domain="machine learning",
            question="hyperparameter tuning",
            topic="neural networks",
        )
    )
    invoke_time = time.time() - start_time
    print(f"   ✓ Invoke completed in {format_time(invoke_time)}")
    print("\n   Messages:")
    for msg in result1.to_messages():
        print(f"   - [{msg.type}]: {msg.content}")

    # Test the runtime-loaded template
    print("\n2. Invoking runtime-loaded template:")
    print("   First invocation (should load the prompt):")
    start_time = time.time()
    result2 = runtime_template.invoke(
        input=dict(
            tools="search, calculator, weather",
            task="analyze market trends",
            context="technology sector in 2023",
        )
    )
    first_invoke_time = time.time() - start_time
    print(f"   ✓ First invoke completed in {format_time(first_invoke_time)}")

    print("\n   Messages:")
    for msg in result2.to_messages():
        print(f"   - [{msg.type}]: {msg.content}")

    # Example 5: Create a custom prompt template and add it to a larger template
    print_separator("Combining with standard ChatPromptTemplate")

    # Create a combined template
    observability_template = ObservabilityChatPromptTemplate.from_observability_platform(
        prompt_name="basic-assistant",
        observability_platform=os_platform,
        load_at_runtime=True,
        template_format="jinja2",  # Use jinja2 templating
        input_variables=["domain", "question", "topic"],  # Explicitly define input variables
    )

    # Create a standard ChatPromptTemplate
    standard_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content="Additional context: This is a follow-up question."),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="Can you elaborate on {{ specific_point }}?"),
        ],
        template_format="jinja2",  # Use jinja2 templating
    )

    # Add them together
    combined_template = observability_template + standard_template

    print("\nInvoking combined template:")
    result3 = combined_template.invoke(
        input=dict(
            domain="programming",
            question="debugging techniques",
            topic="Python",
            chat_history=[
                HumanMessage(content="What are common debugging approaches?"),
                SystemMessage(content="Let me explain some debugging techniques."),
            ],
            specific_point="using breakpoints effectively",
        )
    )

    print("\n   Combined template messages:")
    for msg in result3.to_messages():
        print(f"   - [{msg.type}]: {msg.content}")

    # Example 6: Demonstrate platform switching
    print_separator("Platform switching")

    switchable_template = ObservabilityChatPromptTemplate.from_observability_platform(
        prompt_name="basic-assistant",
        observability_platform=os_platform,
        template_format="jinja2",  # Use jinja2 templating
        input_variables=["domain", "question", "topic"],  # Explicitly define input variables
    )

    print("\n1. Using original platform:")
    result4 = switchable_template.invoke(
        input=dict(
            domain="physics",
            question="relativity",
            topic="spacetime",
        )
    )

    # Create a new platform instance and switch
    new_platform = ObservabilityFactory.create(ObservabilityBackend.EMPTY)
    new_platform.push_prompt(
        "basic-assistant",
        [
            {"role": "system", "content": "You are a MODIFIED assistant specializing in {{ domain }}."},
            {"role": "human", "content": "I need help with {{ question }} related to {{ topic }}."},
        ],
    )

    print("\n2. Switching platforms (should reset loaded prompt):")
    switchable_template.observability_platform = new_platform

    result5 = switchable_template.invoke(
        input=dict(
            domain="physics",
            question="relativity",
            topic="spacetime",
        )
    )

    print("\n   Before platform switch:")
    print(f"   - [{result4.to_messages()[0].type}]: {result4.to_messages()[0].content}")

    print("\n   After platform switch:")
    print(f"   - [{result5.to_messages()[0].type}]: {result5.to_messages()[0].content}")

    print_separator("Performance comparison")

    # Compare performance of runtime vs initialization loading
    print("\nCreating templates:")

    # Init-time template
    start_time = time.time()
    init_template = ObservabilityChatPromptTemplate.from_observability_platform(
        prompt_name="basic-assistant",
        observability_platform=os_platform,
        load_at_runtime=False,
        template_format="jinja2",  # Use jinja2 templating
        input_variables=["domain", "question", "topic"],  # Explicitly define input variables
    )
    init_create_time = time.time() - start_time

    # Runtime template
    start_time = time.time()
    runtime_template = ObservabilityChatPromptTemplate.from_observability_platform(
        prompt_name="basic-assistant",
        observability_platform=os_platform,
        load_at_runtime=True,
        template_format="jinja2",  # Use jinja2 templating
        input_variables=["domain", "question", "topic"],  # Explicitly define input variables
    )
    runtime_create_time = time.time() - start_time

    print(f"   Init-time loading creation: {format_time(init_create_time)}")
    print(f"   Runtime loading creation: {format_time(runtime_create_time)}")

    print("\nInvoking templates (5 times each):")

    # Test init-time template
    init_times = []
    for i in range(5):
        start_time = time.time()
        init_template.invoke(input=dict(domain="test", question="question", topic="topic"))
        init_times.append(time.time() - start_time)

    # Test runtime template
    runtime_times = []
    for i in range(5):
        start_time = time.time()
        runtime_template.invoke(input=dict(domain="test", question="question", topic="topic"))
        runtime_times.append(time.time() - start_time)

    avg_init_time = sum(init_times) / len(init_times)
    avg_runtime_time = sum(runtime_times) / len(runtime_times)

    print(f"   Init-time loading average invoke: {format_time(avg_init_time)}")
    print(f"   Runtime loading average invoke: {format_time(avg_runtime_time)}")
    print(f"   Difference: {format_time(abs(avg_runtime_time - avg_init_time))}")

    if avg_init_time < avg_runtime_time:
        print("   ✓ Init-time loading is faster for repeated invocations")
    else:
        print("   ✓ Runtime loading is faster (unusual, might be affected by other factors)")

    print_separator("Demo Complete")
