"""The demo configures native retrieval tools without exposing collection selection."""

from goodmem_rag.retrieval import make_tools


def test_search_tools_keep_their_scopes_and_filter():
    state = {"spaces": {"langgraph": "graph", "langchain": "chain"}, "reranker_id": "reranker"}
    tools = make_tools(state=state)
    assert [t.metadata.name for t in tools] == ["langgraph_docs_tool", "langchain_docs_tool"]
    for collection, tool in zip(["langgraph", "langchain"], tools):
        assert set(tool.metadata.get_parameters_dict()["properties"]) == {"input"}
        assert tool.retriever.space_ids == (state["spaces"][collection],)
        assert "agentic-rag-llamaindex-goodmem" in tool.retriever.filter
        assert tool.retriever.reranker_id == "reranker"
