"""Deterministic checks of the workflow's control flow and evidence handling."""

import asyncio
from types import SimpleNamespace

from llama_index.core.llms import ChatMessage
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.tools import ToolOutput

from goodmem_rag.agents import AgenticRAGWorkflow, Answer, Grade


class Model:
    def __init__(self, decisions, grades=None):
        self.decisions = iter(decisions)
        self.grades = iter(grades or [])
        self.inputs = []

    async def achat_with_tools(self, *, tools, chat_history, allow_parallel_tool_calls):
        self.inputs.append([m.content for m in chat_history])
        calls = next(self.decisions)
        return SimpleNamespace(
            message=ChatMessage(role="assistant", content="Answer from evidence."), calls=calls
        )

    def get_tool_calls_from_response(self, response, **kwargs):
        return response.calls

    async def astructured_predict(self, cls, prompt, **kwargs):
        self.inputs.append(kwargs)
        return Grade(relevant=next(self.grades))

    async def achat(self, messages):
        self.inputs.append([m.content for m in messages])
        return SimpleNamespace(message=SimpleNamespace(content="Answer from evidence."))


def choice(collection, query):
    return [
        ToolSelection(
            tool_id="call-" + query,
            tool_name=collection + "_docs_tool",
            tool_kwargs={"input": query},
        )
    ]


def tools():
    result = []
    for collection in ["langgraph", "langchain"]:
        node = NodeWithScore(
            node=TextNode(
                text=collection + " evidence",
                metadata={"source": f"https://example.org/{collection}"},
            )
        )

        async def search(input: str, *, _node=node):
            return ToolOutput(
                content=_node.text,
                raw_input={"input": input},
                raw_output=[_node],
                tool_name="search",
            )

        result.append(
            SimpleNamespace(metadata=SimpleNamespace(name=f"{collection}_docs_tool"), acall=search)
        )
    return result


def test_direct_answer_does_not_search():
    async def run():
        result = await AgenticRAGWorkflow(Model([[]]), tools()).run(question="hello")
        assert result.calls == [] and result.steps == ["decide"]

    asyncio.run(run())


def test_irrelevant_evidence_rewrites_and_budget_stops_the_loop():
    async def run():
        model = Model([choice("langgraph", "first")], grades=[False])
        result = await AgenticRAGWorkflow(model, tools(), max_searches=1).run(
            question="original question"
        )
        assert len(result.calls) == 1
        assert result.steps == ["decide", "retrieve", "grade", "rewrite", "decide"]
        assert result.sources == ["https://example.org/langgraph"]
        assert "original question" in str(model.inputs[-1])

    asyncio.run(run())


def test_dependent_search_retains_evidence_from_both_collections():
    async def run():
        model = Model(
            [choice("langgraph", "first"), choice("langchain", "follow-up"), []],
            grades=[True, True],
        )
        result = await AgenticRAGWorkflow(model, tools()).run(question="compare")
        assert [c["name"] for c in result.calls] == ["langgraph_docs_tool", "langchain_docs_tool"]
        assert len(result.sources) == 2
        assert "langgraph evidence" in str(model.inputs[-1]) and "langchain evidence" in str(
            model.inputs[-1]
        )

    asyncio.run(run())


def test_consulted_sources_do_not_fabricate_urls():
    result = Answer(
        "Partial answer",
        nodes=[NodeWithScore(node=TextNode(text="note", metadata={"title": "No source"}))],
    )
    assert result.rendered() == "Partial answer"
