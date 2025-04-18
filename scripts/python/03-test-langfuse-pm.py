from dotenv import find_dotenv, load_dotenv

from langgraph_agent_toolkit.core.observability.langfuse import LangfuseObservability


load_dotenv(find_dotenv())


if __name__ == "__main__":
    # Create a Langfuse observability instance
    observability = LangfuseObservability()

    # Define a joke generator prompt template
    joke_template = """Tell me a {{ joke_type }} joke about {{ topic }}.

Additional context:
- Audience: {{ audience }}
- Keep it {{ tone }}
- Length preference: {{ length }}"""

    # Push the prompt to Langfuse
    observability.push_prompt(
        name="joke-generator",
        prompt_template=joke_template,
    )
    print("Prompt pushed to Langfuse")

    # Retrieve the prompt from Langfuse
    retrieved_prompt = observability.pull_prompt("joke-generator")
    print(f"\nRetrieved prompt type: {type(retrieved_prompt)}")

    # Get the raw template
    template_str = observability.get_template("joke-generator")
    print("\nTemplate content:")
    print(template_str)

    # Render the prompt with variables
    rendered = observability.render_prompt(
        prompt_name="joke-generator",
        joke_type="pun",
        topic="programming",
        audience="developers",
        tone="light-hearted",
        length="short",
    )
    print("\nRendered prompt:")
    print(rendered)
