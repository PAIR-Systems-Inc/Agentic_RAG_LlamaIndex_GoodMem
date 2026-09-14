"""Scoped LlamaIndex tools built from the shared GoodMem retriever."""

from llama_index.core.tools import RetrieverTool
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters
from llama_index.tools.goodmem import GoodMemRetriever


def make_tools(client=None, state=None, *, async_client=None, rerank=True):
    descriptions = {
        "langgraph": "Search LangGraph docs for StateGraph, nodes, edges, reducers, persistence and workflows.",
        "langchain": "Search LangChain docs for models, tools, prompts, built-in agents and tool-calling loops.",
    }
    return [
        RetrieverTool.from_defaults(
            GoodMemRetriever(
                client=client,
                async_client=async_client,
                space_ids=[state["spaces"][collection]],
                filters=MetadataFilters(
                    filters=[
                        MetadataFilter(key="application", value="agentic-rag-llamaindex-goodmem")
                    ]
                ),
                reranker_id=state.get("reranker_id") if rerank else None,
            ),
            name=f"{collection}_docs_tool",
            description=description,
        )
        for collection, description in descriptions.items()
    ]
