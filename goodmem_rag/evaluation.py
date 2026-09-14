"""Comparable retrieval, provenance and agent-routing acceptance checks."""

import importlib.metadata
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from .agents import ask
from .config import chat_model, save_json
from .retrieval import make_tools

RETRIEVAL_CASES = [
    (
        "state",
        "langgraph",
        "What are state, nodes and edges in StateGraph?",
        "graph-api",
        ["state", "nodes", "edges"],
    ),
    (
        "reducers",
        "langgraph",
        "How does the add_messages reducer handle message updates?",
        "graph-api",
        ["add_messages"],
    ),
    (
        "persistence",
        "langgraph",
        "What does a checkpointer do when compiling a graph?",
        "graph-api",
        ["checkpointer"],
    ),
    ("fanout", "langgraph", "How does the Send API implement map-reduce?", "graph-api", ["Send"]),
    (
        "tools",
        "langchain",
        "How do I define a custom tool with the @tool decorator?",
        "tools",
        ["@tool"],
    ),
    (
        "binding",
        "langchain",
        "How do bind_tools and tool_calls work on a chat model?",
        "models",
        ["bind_tools", "tool_calls"],
    ),
    (
        "agents",
        "langchain",
        "How do I create an agent using create_agent?",
        "overview",
        ["create_agent"],
    ),
    (
        "structured",
        "langchain",
        "How do I get structured output from a chat model?",
        "models",
        ["with_structured_output"],
    ),
]
COMPARISON_QUESTION = (
    "Using details from both docs, explain how a LangGraph StateGraph differs from "
    "a LangChain agent's tool-calling loop. Cite both sources."
)
SEQUENTIAL_QUESTION = (
    "First search the LangGraph overview to identify the higher-level framework it recommends "
    "for prebuilt agent architectures. After reading that result, search the recommended "
    "framework's documentation to identify its agent constructor and explain how its "
    "tool-calling loop works. Cite both sources."
)


def _citations(text):
    return {
        url.rstrip(".,;:!?") for url in re.findall(r"https://docs\.langchain\.com/[^\s)\]>]+", text)
    }


async def evaluate(settings, state, output, retrieval_only=False):
    model = None if retrieval_only else chat_model()
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "packages": {
            p: importlib.metadata.version(p)
            for p in ["goodmem", "llamaindex-goodmem", "llama-index-core", "llama-index-workflows"]
        },
        "chat_model": getattr(model, "model", None),
        "documents": [{"source": d["source"], "sha256": d["sha256"]} for d in state["documents"]],
        "retrieval_configuration": {"base_top_k": 5, "rerank_candidates": 20, "rerank_top_k": 5},
        "retrieval": [],
        "agents": [],
        "limitations": "Small acceptance suite, not a comparative answer-quality benchmark. "
        "Citation checks establish provenance, not entailment of every claim. "
        "Timing includes network and provider variation.",
    }
    async with settings.async_client() as client:
        report["server"] = (await client.system.info()).model_dump(mode="json", exclude_none=True)
        report["embedder_model"] = (
            await client.embedders.get(id=state["embedder_id"])
        ).model_identifier
        report["reranker_model"] = (
            (await client.rerankers.get(id=state["reranker_id"])).model_identifier
            if state.get("reranker_id")
            else None
        )
        for rerank in [False, True] if state.get("reranker_id") else [False]:
            tools = {
                t.metadata.name: t
                for t in make_tools(async_client=client, state=state, rerank=rerank)
            }
            for name, collection, query, expected_page, keywords in RETRIEVAL_CASES:
                start = time.monotonic()
                output_ = await tools[f"{collection}_docs_tool"].acall(input=query)
                nodes = output_.raw_output
                text = "\n".join(n.text for n in nodes).lower()
                sources = [n.metadata.get("source", "") for n in nodes]
                passed = (
                    bool(nodes)
                    and any(s.endswith("/" + expected_page) for s in sources)
                    and all(k.lower() in text for k in keywords)
                    and all(
                        n.metadata["_goodmem"]["space_id"] == state["spaces"][collection]
                        for n in nodes
                    )
                )
                report["retrieval"].append(
                    {
                        "case": name,
                        "reranked": rerank,
                        "passed": passed,
                        "seconds": round(time.monotonic() - start, 3),
                        "sources": sources,
                        "scores": [n.score for n in nodes],
                        "expected_page": expected_page,
                        "keywords": keywords,
                    }
                )
                save_json(Path(output), report)
                print(
                    f"Retrieval {name} rerank={rerank}: {'PASS' if passed else 'FAIL'}", flush=True
                )
        if model:
            cases = [
                ("direct", "What is the capital of France?", set()),
                (
                    "single_space",
                    "What is a checkpointer used for in LangGraph? Cite the docs.",
                    {"langgraph"},
                ),
                ("cross_space", COMPARISON_QUESTION, {"langgraph", "langchain"}),
                ("sequential", SEQUENTIAL_QUESTION, {"langgraph", "langchain"}),
            ]
            for agent in ["workflow", "react"]:
                for name, question, expected in cases:
                    start = time.monotonic()
                    case = {"agent": agent, "case": name, "question": question}
                    try:
                        result = await ask(
                            model,
                            make_tools(async_client=client, state=state),
                            question,
                            agent=agent,
                        )
                        used = {c["name"].removesuffix("_docs_tool") for c in result.calls}
                        citations = _citations(result.text)
                        citations_ok = not expected or (
                            bool(citations)
                            and citations <= set(result.sources)
                            and all(any(f"/{s}/" in u for u in citations) for s in expected)
                        )
                        passed = (
                            used == expected
                            and citations_ok
                            and len(result.calls) <= 4
                            and (
                                name != "sequential" or len({c["round"] for c in result.calls}) >= 2
                            )
                            and (name != "direct" or "paris" in result.text.lower())
                        )
                        case.update(
                            passed=passed,
                            tool_calls=result.calls,
                            answer=result.text,
                            citations=sorted(citations),
                            sources=result.sources,
                            steps=result.steps,
                        )
                    except Exception as exc:
                        case.update(passed=False, error=f"{type(exc).__name__}: {exc}")
                    case["seconds"] = round(time.monotonic() - start, 3)
                    report["agents"].append(case)
                    save_json(Path(output), report)
                    print(
                        f"Agent {agent}/{name}: {'PASS' if case['passed'] else 'FAIL'}", flush=True
                    )
    report["passed"] = all(c["passed"] for c in report["retrieval"] + report["agents"])
    save_json(Path(output), report)
    return report["passed"]
