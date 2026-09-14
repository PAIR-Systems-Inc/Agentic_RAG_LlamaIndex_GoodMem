# Agentic RAG with LlamaIndex and GoodMem

Build a documentation assistant that decides when to search, which collection to consult, and whether another lookup is needed. Four notebooks take you from a single workflow step to an agent that answers questions across two documentation collections and cites its sources.

This is a LlamaIndex adaptation of [Chandula Senevirathna’s Agentic_RAG](https://github.com/ChandulaSenevirathna/Agentic_RAG), following our [LangChain + GoodMem port](https://github.com/PAIR-Systems-Inc/Agentic_RAG_GoodMem). It keeps the same six LangGraph and LangChain documentation pages so both implementations can face the same questions.

## The notebooks

| Notebook | What you’ll learn |
| --- | --- |
| [1. Workflows](1_LlamaIndex_Workflows.ipynb) | Connect steps with typed events, run two LLM calls concurrently, and join their results. |
| [2. Conditional routing](2_LlamaIndex_Conditional_Routing.ipynb) | Classify a question and send it to the appropriate step. |
| [3. Agentic RAG](3_LlamaIndex_Agentic_RAG.ipynb) | Build an explicit retrieve, grade, draft and rewrite workflow. |
| [4. ReAct multi-hop RAG](4_LlamaIndex_ReAct_MultiHop_RAG.ipynb) | Use LlamaIndex’s prebuilt agent to perform dependent searches across collections. |

GoodMem stores the documents and handles chunking, embeddings, search and optional reranking. LlamaIndex runs the agent. No local embedding model or vector database is needed in the Python process. Reranking itself needs no LLM.

## Get started

You need Python 3.11+, [uv](https://docs.astral.sh/uv/), a Cohere API key, and a running GoodMem server. Docker Compose is included if you want a local server.

Clone this repository, then install its dependencies, including the published [GoodMem integration for LlamaIndex](https://pypi.org/project/llamaindex-goodmem/0.2.0/):

```bash
git clone https://github.com/PAIR-Systems-Inc/Agentic_RAG_LlamaIndex_GoodMem.git
cd Agentic_RAG_LlamaIndex_GoodMem
uv sync --locked
cp .env.example .env
```

Set `COHERE_API_KEY` in `.env`. The defaults use Cohere for chat, embeddings and reranking. Notebooks 1–2 are ready to run with just that key.

For notebooks 3–4, start a new local server and load the documentation:

```bash
docker compose up -d --wait
uv run llamaindex-rag setup --init --with-reranker
uv run llamaindex-rag ask "What is a checkpointer used for in LangGraph? Cite the docs."
```

For an existing server, set `GOODMEM_BASE_URL` and `GOODMEM_API_KEY`, then omit `--init`. You can supply existing embedder and reranker IDs too.

Try a question that needs both collections:

```bash
uv run llamaindex-rag ask "Compare LangGraph StateGraph with the LangChain agent loop. Cite both docs."
```

Add `--agent workflow` to use notebook 3’s explicit workflow. The default is notebook 4’s ReAct agent. Both commands print the searches they made.

## Open the notebooks

In VS Code, select this project’s `.venv` as the notebook kernel. For JupyterLab:

```bash
uv run python -m ipykernel install --sys-prefix --name llamaindex-goodmem-rag
uv run --group notebooks jupyter lab
```

Run cells from top to bottom. See [running and validation](docs/running-the-demo.md) and the [integration findings](docs/llamaindex-integration-review.md) for further details.

Original attribution and terms are preserved in [LICENSE.md](LICENSE.md); see [provenance](docs/upstream.md).
