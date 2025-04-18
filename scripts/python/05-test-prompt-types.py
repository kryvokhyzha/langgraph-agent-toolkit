from dotenv import find_dotenv, load_dotenv

from langgraph_agent_toolkit.core.observability.factory import ObservabilityFactory
from langgraph_agent_toolkit.core.observability.types import ChatMessageDict, ObservabilityBackend


load_dotenv(find_dotenv())

if __name__ == "__main__":
    # Create observability instance using the factory
    observability = ObservabilityFactory.create(ObservabilityBackend.EMPTY)

    print("=== Testing ChatMessageDict Format ===\n")

    # Test with string template (basic format for comparison)
    string_template = "Hello, {{ name }}! How can I help you with {{ topic }} today?"
    observability.push_prompt("string-template", string_template)
    print("✓ String template saved (for comparison)")

    # Test with List[ChatMessageDict] - system and human roles
    message_list_1: list[ChatMessageDict] = [
        {"role": "system", "content": "You are an AI assistant specialized in {{ domain }}."},
        {"role": "human", "content": "I need help with {{ problem }} in {{ context }}."},
    ]
    observability.push_prompt("chat-messages-1", message_list_1)
    print("✓ List[ChatMessageDict] with system+human saved")

    # Test with just human message
    message_list_2: list[ChatMessageDict] = [
        {"role": "human", "content": "Explain {{ topic }} as if I'm {{ audience }}."}
    ]
    observability.push_prompt("chat-messages-2", message_list_2)
    print("✓ List[ChatMessageDict] with just human saved")

    # Test with multiple alternating messages
    message_list_3: list[ChatMessageDict] = [
        {"role": "system", "content": "You are an AI tutor specialized in {{ subject }}."},
        {"role": "human", "content": "Can you help me understand {{ concept }}?"},
        {"role": "system", "content": "Focus on {{ focus_area }} when explaining."},
        {"role": "human", "content": "Make sure to include examples related to {{ example_context }}."},
    ]
    observability.push_prompt("chat-messages-3", message_list_3)
    print("✓ List[ChatMessageDict] with multiple alternating messages saved")

    # Test with multiple human messages in sequence
    message_list_4: list[ChatMessageDict] = [
        {"role": "system", "content": "You are an AI coding assistant."},
        {"role": "human", "content": "I'm working on a {{ language }} project."},
        {"role": "human", "content": "I need help with {{ feature }}."},
        {"role": "human", "content": "My current code structure is {{ structure }}."},
    ]
    observability.push_prompt("chat-messages-4", message_list_4)
    print("✓ List[ChatMessageDict] with multiple human messages saved")

    print("\n=== Retrieving Templates ===")

    # Retrieve and check each type
    templates = {
        "string-template": "String template",
        "chat-messages-1": "ChatMessageDict (system+human)",
        "chat-messages-2": "ChatMessageDict (human only)",
        "chat-messages-3": "ChatMessageDict (alternating)",
        "chat-messages-4": "ChatMessageDict (multiple human)",
    }

    for name, description in templates.items():
        result = observability.pull_prompt(name)
        print(f"✓ {description} retrieved: {type(result).__name__}")

    print("\n=== Rendering Templates with Variables ===")

    # Render string template for comparison
    print("\n1. String template (for comparison):")
    rendered = observability.render_prompt(prompt_name="string-template", name="User", topic="AI prompt management")
    print(f"   {rendered}")

    # Render ChatMessageDict example
    print("\n2. System+Human messages:")
    rendered = observability.render_prompt(
        prompt_name="chat-messages-1", domain="data science", problem="data visualization", context="financial analysis"
    )
    print(f"   {rendered}")

    # Render Human-only example
    print("\n3. Human-only message:")
    rendered = observability.render_prompt(
        prompt_name="chat-messages-2", topic="machine learning", audience="a five-year-old"
    )
    print(f"   {rendered}")

    # Render alternating messages example
    print("\n4. Alternating messages:")
    rendered = observability.render_prompt(
        prompt_name="chat-messages-3",
        subject="physics",
        concept="quantum entanglement",
        focus_area="practical applications",
        example_context="computing",
    )
    print(f"   {rendered}")

    # Render multiple human messages example
    print("\n5. Multiple human messages:")
    rendered = observability.render_prompt(
        prompt_name="chat-messages-4", language="Python", feature="async functions", structure="modular with classes"
    )
    print(f"   {rendered}")

    print("\n=== Testing Complete ===")
