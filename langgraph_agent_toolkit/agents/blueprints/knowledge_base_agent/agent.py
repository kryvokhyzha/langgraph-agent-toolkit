import os
from typing import Any

from langchain_aws import AmazonKnowledgeBasesRetriever
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langchain_core.runnables.base import RunnableSequence
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps

from langgraph_agent_toolkit.agents.agent import Agent
from langgraph_agent_toolkit.core import settings
from langgraph_agent_toolkit.core.models.factory import CompletionModelFactory
from langgraph_agent_toolkit.helper.logging import logger
from langgraph_agent_toolkit.schema.models import ModelProvider


class AgentState(MessagesState, total=False):
    """State for the knowledge-base agent."""

    remaining_steps: RemainingSteps
    retrieved_documents: list[dict[str, Any]]
    kb_documents: str


def get_kb_retriever():
    """Create and return a knowledge-base retriever."""
    kb_id = os.environ.get("AWS_KB_ID", "")
    if not kb_id:
        raise ValueError("AWS_KB_ID environment variable must be set")

    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id=kb_id,
        retrieval_config={
            "vectorSearchConfiguration": {
                "numberOfResults": 3,
            }
        },
    )
    return retriever


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    """Add the knowledge-base system prompt to the model."""

    def create_system_message(state):
        base_prompt = """You are a helpful assistant that provides accurate information based on retrieved documents.

        You will receive a query along with relevant documents retrieved from a knowledge base.
        Use these documents to inform your response.

        Follow these guidelines:
        1. Base your answer primarily on the retrieved documents
        2. If the documents contain the answer, provide it clearly and concisely
        3. If the documents are insufficient, state that you don't have enough information
        4. Never make up facts or information not present in the documents
        5. Always cite the source documents when referring to specific information
        6. If the documents contradict each other, acknowledge this and explain the different perspectives

        Format your response in a clear, conversational manner. Use markdown formatting when appropriate.
        """

        if state.get("kb_documents"):
            document_prompt = (
                f"\n\nI've retrieved the following documents that may be relevant to the query:"
                f"\n\n{state['kb_documents']}\n\n"
                "Please use these documents to inform your response to the user's query. "
                "Only use information from these documents and clearly indicate when you are unsure."
            )
            return [SystemMessage(content=base_prompt + document_prompt)] + state["messages"]
        else:
            no_docs_prompt = "\n\nNo relevant documents were found in the knowledge base for this query."
            return [SystemMessage(content=base_prompt + no_docs_prompt)] + state["messages"]

    preprocessor = RunnableLambda(
        create_system_message,
        name="StateModifier",
    )
    return RunnableSequence(preprocessor, model)


def _retrieval_query(content: str | list[str | dict[str, Any]]) -> str:
    """Extract text for retrieval without sending attached media."""
    if isinstance(content, str):
        return content.strip()
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts).strip()


def _document_source(metadata: dict[str, Any], source_metadata: dict[str, Any]) -> str:
    """Read custom sources and Bedrock document locations."""
    source = metadata.get("source") or source_metadata.get("source")
    if source:
        return source

    location = metadata.get("location")
    if isinstance(location, dict):
        for location_key, source_key in (
            ("s3Location", "uri"),
            ("webLocation", "url"),
            ("confluenceLocation", "url"),
            ("salesforceLocation", "url"),
            ("sharePointLocation", "url"),
            ("kendraDocumentLocation", "uri"),
            ("customDocumentLocation", "id"),
        ):
            details = location.get(location_key)
            if isinstance(details, dict) and isinstance(details.get(source_key), str) and details[source_key]:
                return details[source_key]

    return source_metadata.get("x-amz-bedrock-kb-source-uri") or "Unknown"


async def retrieve_documents(state: AgentState, config: RunnableConfig) -> AgentState:
    """Retrieve documents relevant to the latest user message."""
    human_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    if not human_messages:
        return {"messages": [], "retrieved_documents": []}

    query = _retrieval_query(human_messages[-1].content)
    if not query:
        return {"messages": [], "retrieved_documents": []}

    try:
        retriever = get_kb_retriever()

        retrieved_docs = await retriever.ainvoke(query)

        document_summaries = []
        for i, doc in enumerate(retrieved_docs, 1):
            source_metadata = doc.metadata.get("source_metadata")
            if not isinstance(source_metadata, dict):
                source_metadata = {}
            summary = {
                "id": doc.metadata.get("id", source_metadata.get("id", f"doc-{i}")),
                "source": _document_source(doc.metadata, source_metadata),
                "title": doc.metadata.get("title", source_metadata.get("title", f"Document {i}")),
                "content": doc.page_content,
                "relevance_score": doc.metadata.get("score", 0),
            }
            document_summaries.append(summary)

        logger.info(f"Retrieved {len(document_summaries)} documents for query: {query[:50]}...")

        return {"retrieved_documents": document_summaries, "messages": []}

    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        return {"retrieved_documents": [], "messages": []}


async def prepare_augmented_prompt(state: AgentState, config: RunnableConfig) -> AgentState:
    """Add retrieved document content to the prompt state."""
    documents = state.get("retrieved_documents", [])

    if not documents:
        return {"kb_documents": "", "messages": []}

    formatted_docs = "\n\n".join(
        [
            f"--- Document {i + 1} ---\n"
            f"Source: {doc.get('source', 'Unknown')}\n"
            f"Title: {doc.get('title', 'Unknown')}\n\n"
            f"{doc.get('content', '')}"
            for i, doc in enumerate(documents)
        ]
    )

    return {"kb_documents": formatted_docs, "messages": []}


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    """Generate a response from retrieved documents."""
    model_config_key = config["configurable"].get("model_config_key") or config["configurable"].get(
        "agent_config", {}
    ).get("model_config")

    if model_config_key and model_config_key in settings.MODEL_CONFIGS:
        model_config = settings.MODEL_CONFIGS[model_config_key]
        model = CompletionModelFactory.get_model_from_config(model_config)
    else:
        model = CompletionModelFactory.create(
            model_provider=config["configurable"].get(
                "model_provider", ModelProvider.FAKE if settings.USE_FAKE_MODEL else ModelProvider.OPENAI
            ),
            model_name=config["configurable"].get("model_name", settings.OPENAI_MODEL_NAME),
            openai_api_base=settings.OPENAI_API_BASE_URL,
            openai_api_key=settings.OPENAI_API_KEY,
        )

    model_runnable = wrap_model(model)

    response = await model_runnable.ainvoke(state, config)

    return {"messages": [response]}


agent = StateGraph(AgentState)

agent.add_node("retrieve_documents", retrieve_documents)
agent.add_node("prepare_augmented_prompt", prepare_augmented_prompt)
agent.add_node("model", acall_model)

agent.set_entry_point("retrieve_documents")

agent.add_edge("retrieve_documents", "prepare_augmented_prompt")
agent.add_edge("prepare_augmented_prompt", "model")
agent.add_edge("model", END)

kb_agent = Agent(
    name="kb-agent",
    description="A retrieval-augmented generation agent using Amazon Bedrock Knowledge Base.",
    graph=agent.compile(checkpointer=None),
)
