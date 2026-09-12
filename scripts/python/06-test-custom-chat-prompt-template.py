import time
from typing import List

import rootutils
from dotenv import find_dotenv, load_dotenv


_ = rootutils.setup_root(
    search_from=__file__,
    indicator=".project-root",
    pythonpath=True,
    dotenv=False,
)
load_dotenv(find_dotenv(".local.env"), override=True)

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langgraph_agent_toolkit.core.observability.factory import ObservabilityFactory
from langgraph_agent_toolkit.core.observability.types import ChatMessageDict, ObservabilityBackend
from langgraph_agent_toolkit.core.prompts.chat_prompt_template import ObservabilityChatPromptTemplate


def format_time(seconds: float) -> str:
    """Format seconds as milliseconds with two decimal places."""
    return f"{seconds * 1000:.2f}ms"


def print_separator(title: str = None):
    """Print a separator line with an optional title."""
    width = 80
    if title:
        print(f"\n{'=' * 5} {title} {'=' * (width - 7 - len(title))}")
    else:
        print(f"\n{'=' * width}")


if __name__ == "__main__":
    print_separator("ObservabilityChatPromptTemplate Demo")

    # Create the observability platform.
    os_platform = ObservabilityFactory.create(ObservabilityBackend.EMPTY)

    print("✓ Created observability platforms")

    # Create and store sample prompts.
    basic_chat_messages: List[ChatMessageDict] = [
        {"role": "system", "content": "You are an AI assistant specialized in {{ domain }}."},
        {"role": "human", "content": "I need help with {{ question }} related to {{ topic }}."},
    ]
    os_platform.push_prompt("basic-assistant", basic_chat_messages)
    print("✓ Stored basic chat prompt")

    # Define a tool-using agent prompt.
    agent_chat_messages: List[ChatMessageDict] = [
        {"role": "system", "content": "You are an AI agent that can use tools. Available tools: {{ tools }}"},
        {"role": "human", "content": "I want to {{ task }} with the following context: {{ context }}"},
    ]
    os_platform.push_prompt("tool-using-agent", agent_chat_messages)
    print("✓ Stored tool-using agent prompt")

    # Define a prompt with Jinja2 conditions.
    conditional_chat_messages: List[ChatMessageDict] = [
        {"role": "system", "content": "You are an AI assistant specialized in {{ domain }}."},
        {
            "role": "human",
            "content": """
            {% if expertise_level == 'beginner' %}
            Please explain {{ topic }} in simple terms suitable for beginners.
            Use basic examples and avoid technical jargon.
            {% elif expertise_level == 'intermediate' %}
            Provide a comprehensive explanation of {{ topic }} with some technical details.
            Include practical examples that demonstrate the concepts.
            {% else %}
            Give an advanced analysis of {{ topic }} with in-depth technical details.
            Include complex examples and reference relevant research or methodologies.
            {% endif %}

            Additional context: {{ context }}
        """,
        },
    ]
    os_platform.push_prompt("conditional-assistant", conditional_chat_messages)
    print("✓ Stored conditional chat prompt with Jinja2 if-else statements")

    # Create templates with different configurations.
    print_separator("Creating prompt templates")

    # Load this template during initialization.
    print("\n1. Template with initialization-time loading:")
    start_time = time.time()
    init_template = ObservabilityChatPromptTemplate.from_observability_platform(
        prompt_name="basic-assistant",
        observability_platform=os_platform,
        load_at_runtime=False,
        template_format="jinja2",
        input_variables=["domain", "question", "topic"],
    )
    load_time = time.time() - start_time
    print(f"   ✓ Created template (load time: {format_time(load_time)})")
    print(f"   ✓ Template has {len(init_template.messages)} message(s)")

    # Load this template when it runs.
    print("\n2. Template with runtime loading:")
    start_time = time.time()
    runtime_template = ObservabilityChatPromptTemplate.from_observability_platform(
        prompt_name="tool-using-agent",
        observability_platform=os_platform,
        load_at_runtime=True,
        template_format="jinja2",
        input_variables=["tools", "task", "context"],
    )
    create_time = time.time() - start_time
    print(f"   ✓ Created template (create time: {format_time(create_time)})")

    # Create a template from a backend.
    print("\n3. Template using backend factory method:")
    backend_template = ObservabilityChatPromptTemplate.from_observability_backend(
        prompt_name="basic-assistant",
        observability_backend=ObservabilityBackend.EMPTY,
        load_at_runtime=True,
        template_format="jinja2",
        input_variables=["domain", "question", "topic"],
    )
    print("   ✓ Created template using backend factory")

    # Create a template directly.
    print("\n4. Template with direct initialization:")
    direct_template = ObservabilityChatPromptTemplate(
        prompt_name="tool-using-agent",
        observability_backend=ObservabilityBackend.EMPTY,
        load_at_runtime=True,
        template_format="jinja2",
        cache_ttl_seconds=120,
        input_variables=["tools", "task", "context"],
    )
    print("   ✓ Created template with direct initialization")

    # Run the templates.
    print_separator("Invoking templates")

    # Run the initialization-loaded template.
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

    # Run the runtime-loaded template.
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

    # Run the conditional template.
    print("\nInvoking conditional template with different expertise levels:")

    # Create the conditional template.
    conditional_template = ObservabilityChatPromptTemplate.from_observability_platform(
        prompt_name="conditional-assistant",
        observability_platform=os_platform,
        load_at_runtime=True,
        template_format="jinja2",
        input_variables=["domain", "expertise_level", "topic", "context"],
    )

    # Run the template at each expertise level.
    expertise_levels = ["beginner", "intermediate", "advanced"]

    for level in expertise_levels:
        print(f"\n   Testing with expertise_level='{level}':")
        result = conditional_template.invoke(
            input=dict(
                domain="data science",
                expertise_level=level,
                topic="neural networks",
                context="For a recommendation system project",
            )
        )

        print(f"   Messages for {level} level:")
        for msg in result.to_messages():
            if msg.type == "human":
                # Show the first 100 characters of the content.
                content = msg.content.strip()[:100] + "..." if len(msg.content) > 100 else msg.content.strip()
                print(f"   - [{msg.type}]: {content}")
            else:
                print(f"   - [{msg.type}]: {msg.content}")

    # Add a custom prompt template to a larger template.
    print_separator("Combining with standard ChatPromptTemplate")

    # Create the combined template.
    observability_template = ObservabilityChatPromptTemplate.from_observability_platform(
        prompt_name="basic-assistant",
        observability_platform=os_platform,
        load_at_runtime=True,
        template_format="jinja2",
        input_variables=["domain", "question", "topic"],
    )

    # Create a standard `ChatPromptTemplate`.
    standard_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content="Additional context: This is a follow-up question."),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="Can you elaborate on {{ specific_point }}?"),
        ],
        template_format="jinja2",  # Use jinja2 templating
    )

    # Combine the templates.
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

    # Switch the observability platform.
    print_separator("Platform switching")

    switchable_template = ObservabilityChatPromptTemplate.from_observability_platform(
        prompt_name="basic-assistant",
        observability_platform=os_platform,
        template_format="jinja2",
        input_variables=["domain", "question", "topic"],
    )

    print("\n1. Using original platform:")
    result4 = switchable_template.invoke(
        input=dict(
            domain="physics",
            question="relativity",
            topic="spacetime",
        )
    )

    # Create and use a new platform instance.
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

    # Compare runtime and initialization loading.
    print("\nCreating templates:")

    # Create the initialization-loaded template.
    start_time = time.time()
    init_template = ObservabilityChatPromptTemplate.from_observability_platform(
        prompt_name="basic-assistant",
        observability_platform=os_platform,
        load_at_runtime=False,
        template_format="jinja2",
        input_variables=["domain", "question", "topic"],
    )
    init_create_time = time.time() - start_time

    # Create the runtime-loaded template.
    start_time = time.time()
    runtime_template = ObservabilityChatPromptTemplate.from_observability_platform(
        prompt_name="basic-assistant",
        observability_platform=os_platform,
        load_at_runtime=True,
        template_format="jinja2",
        input_variables=["domain", "question", "topic"],
    )
    runtime_create_time = time.time() - start_time

    print(f"   Init-time loading creation: {format_time(init_create_time)}")
    print(f"   Runtime loading creation: {format_time(runtime_create_time)}")

    print("\nInvoking templates (5 times each):")

    # Run the initialization-loaded template.
    init_times = []
    for i in range(5):
        start_time = time.time()
        init_template.invoke(input=dict(domain="test", question="question", topic="topic"))
        init_times.append(time.time() - start_time)

    # Run the runtime-loaded template.
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
