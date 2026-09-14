# Running and validating the LlamaIndex port

Use the same Python environment for the CLI and notebooks. `uv sync --locked` installs LlamaIndex, its Cohere and Groq adapters, the official GoodMem SDK, and `llamaindex-goodmem 0.2.0` from PyPI. The lockfile contains no LangChain, LangGraph or Chroma dependencies.

## Existing GoodMem server

Set `GOODMEM_BASE_URL` to the REST server root, without `/v1` or `/mcp`, and supply its `GOODMEM_API_KEY`. Set `GOODMEM_EMBEDDER_ID` and optionally `GOODMEM_RERANKER_ID` to reuse registered models. Run `uv run llamaindex-rag setup` without `--init`.

The LlamaIndex port uses the namespace `agentic-rag-llamaindex-goodmem`, so its spaces are separate from the earlier port. Setup downloads the same six official pages, uses deterministic memory IDs for each content version, and waits for the memories it wrote. Re-running setup reuses unchanged documents. A changed page replaces the previous version only after the replacement is indexed.

For a new Docker server, run `docker compose up -d --wait`, then `uv run llamaindex-rag setup --init --with-reranker`. The Compose project has its own database volume. Its default host port is 8088; set `GOODMEM_PORT` and update `GOODMEM_BASE_URL` if another demo already uses that port.

## Chat providers

Cohere is the tested default: `CHAT_PROVIDER=cohere`, `CHAT_MODEL=command-a-03-2025`, and `COHERE_API_KEY`. Groq is also configurable through `CHAT_PROVIDER=groq`, `CHAT_MODEL=openai/gpt-oss-120b`, and `GROQ_API_KEY`; this provider path has not been live-tested in this exercise.

For Cohere, the routing tutorial and relevance grader use LlamaIndex’s text-based Pydantic program. The tested Cohere adapter omits enum values when converting tool schemas, so enum-based classification can produce invalid labels. The explicit agent workflow uses native tool calls with simple query parameters; ReAct uses ordinary chat.

## Validation commands

```bash
uv run pytest -q
uv run ruff check goodmem_rag tests scripts
uv run python scripts/run_notebooks.py
uv run llamaindex-rag evaluate --retrieval-only
uv run llamaindex-rag evaluate --output .runtime/evaluation.json
```

The live evaluation has eight retrieval questions with and without reranking, then four questions for each agent: general knowledge, a single collection, a comparison, and a dependent second search. It checks tool use, source provenance, collection scope and the search budget. It does not judge whether every generated statement follows from its cited passage.

The notebook runner starts a fresh kernel for each notebook and writes executed copies to `.runtime/executed`. Use `--output` to choose another directory. Source notebooks keep their outputs empty.

Stop the server with `docker compose stop`; the corpus persists in its volume. Do not delete the volume unless you intend to discard that server’s data.
