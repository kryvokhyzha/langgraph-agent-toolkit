import rootutils
from dotenv import find_dotenv, load_dotenv

from langgraph_agent_toolkit.core.observability.empty import EmptyObservability


load_dotenv(find_dotenv())
custom_dir = rootutils.find_root(__file__, indicator=".project-root") / "data" / "prompts"


if __name__ == "__main__":
    # Create an observability instance
    observability = EmptyObservability(prompts_dir=str(custom_dir))

    # Define a joke generator prompt template
    joke_template = """Tell me a {{ joke_type }} joke about {{ topic }}.

Additional context:
- Audience: {{ audience }}
- Keep it {{ tone }}
- Length preference: {{ length }}"""

    # Push the prompt
    observability.push_prompt(
        name="joke-generator",
        prompt_template=joke_template,
    )
    print("Prompt pushed to local storage")

    # Retrieve the prompt
    retrieved_prompt = observability.pull_prompt("joke-generator")
    print(f"\nRetrieved prompt type: {type(retrieved_prompt)}")

    # Display the template
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
