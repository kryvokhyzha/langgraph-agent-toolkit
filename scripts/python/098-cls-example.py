import os
from enum import StrEnum, auto
from typing import List

from dotenv import find_dotenv, load_dotenv
from langchain.chat_models.base import init_chat_model
from langchain_core.prompts import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from pydantic import BaseModel, Field


load_dotenv(find_dotenv(".local.env"), override=True)


class IntentType(StrEnum):
    GENERAL_QUESTION = auto()
    OUT_OF_SCOPE = auto()
    ASKS_FOR_HUMAN_SUPPORT = auto()


class ThoughtsSchema(BaseModel):
    thoughts_on_input_query: str = Field(
        description="The response on user query.",
    )
    reflection_on_previous_thoughts: str = Field(
        description="The alternative response on user query.",
    )


class ResponseSchema(BaseModel):
    thoughts: List[ThoughtsSchema] = Field(
        description="Explanation of the response.",
        max_length=3,
        min_length=1,
    )
    classifyied_intent_type: IntentType = Field(
        description="Classified intent type of the user query.",
    )


if __name__ == "__main__":
    DEFAULT_CONFIG_PREFIX = "agent"
    DEFAULT_CONFIGURABLE_FIELDS = ("temperature", "max_tokens", "top_p", "streaming")
    DEFAULT_MODEL_PARAMETER_VALUES = dict(
        temperature=0.1,
        max_tokens=2048,
        top_p=0.95,
        streaming=False,
    )

    model_so = init_chat_model(
        model=os.getenv("AZURE_OPENAI_MODEL_NAME"),
        model_provider="azure_openai",
        config_prefix=DEFAULT_CONFIG_PREFIX,
        configurable_fields=DEFAULT_CONFIGURABLE_FIELDS,
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        **DEFAULT_MODEL_PARAMETER_VALUES,
    ).with_structured_output(ResponseSchema)

    # Few-shot examples for better classification
    few_shot_examples = [
        {"query": "What features are included in your service?", "intent": "GENERAL_QUESTION"},
        {"query": "How do I reset my password?", "intent": "GENERAL_QUESTION"},
        {"query": "What's the weather like today?", "intent": "OUT_OF_SCOPE"},
        {"query": "Can you help me cook dinner?", "intent": "OUT_OF_SCOPE"},
        {"query": "I need to speak with a human agent", "intent": "ASKS_FOR_HUMAN_SUPPORT"},
        {"query": "Can I talk to someone from your support team?", "intent": "ASKS_FOR_HUMAN_SUPPORT"},
    ]

    input_message = ChatPromptTemplate(
        [
            SystemMessagePromptTemplate.from_template(
                (
                    "You are a helpful assistant that classifies user queries into different intent types. "
                    "The possible intent types are:\n"
                    "{% for intent in intent_types %}"
                    "- {{ intent.value }}: {{ intent.name }}\n"
                    "{% endfor %}\n\n"
                    "Here are some examples of how to classify queries:\n"
                    "{% for example in examples %}"
                    '- **Query**: "{{ example.query }}"\n'
                    "  **Intent**: {{ example.intent }}\n"
                    "{% endfor %}"
                ),
                template_format="jinja2",
            ),
            HumanMessagePromptTemplate.from_template(
                "Classify the following user query (encapsulated in triple quotes) "
                "into one of the provided intent types:\n"
                "'''{query}'''"
            ),
            # AIMessagePromptTemplate.from_template(
            #     "Here is the classification result:\n"
            #     "Thoughts: {{ thoughts.thoughts_on_input_query }}\n"
            #     "Reflection: {{ thoughts.reflection_on_previous_thoughts }}\n"
            #     "Classified Intent Type: {{ classifyied_intent_type }}"
            # ),
            # HumanMessagePromptTemplate.from_template(
            #     "Classify the following user query (encapsulated in triple quotes) "
            #     "into one of the provided intent types:\n"
            #     "'''{query}'''"
            # ),
        ]
    )

    _rendered_msgs = input_message.invoke(
        {
            "intent_types": list(IntentType),
            "examples": few_shot_examples,
            "query": "Who is the president of Brasil?",
        }
    )

    print("=" * 25)
    print(f"Input message: {_rendered_msgs.to_string()}")
    print("=" * 25)

    chain = input_message | model_so

    result = chain.invoke(
        {"query": "Who is the president of Brasil?", "intent_types": list(IntentType), "examples": few_shot_examples}
    )

    print(f"Result type: {type(result)}")
    print(f"Result: {result}")
    print("=" * 25)
