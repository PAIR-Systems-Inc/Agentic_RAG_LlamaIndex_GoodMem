# What the LlamaIndex port found

The exercise improved the shared integration as well as producing a working demo. GoodMem fits LlamaIndex’s retriever interface naturally, and its server-managed ingestion avoids loading an embedding model in every application. Version 0.1 provided administrative tools but missed that central interface; version 0.2 adds it.

The port has a separate public GitHub home at [PAIR-Systems-Inc/Agentic_RAG_LlamaIndex_GoodMem](https://github.com/PAIR-Systems-Inc/Agentic_RAG_LlamaIndex_GoodMem). General integration changes are released in [llamaindex-goodmem 0.2.0](https://pypi.org/project/llamaindex-goodmem/0.2.0/), with [source and migration notes](https://github.com/PAIR-Systems-Inc/goodmem-llamaindex/releases/tag/v0.2.0). This demo installs the published package from PyPI.

## Reproduced problems in version 0.1

We started with the then-current PyPI version, 0.1.0. Its `base.py` exactly matches the repository baseline at `2a6f352766c58dfd7a2a3c851dfa1e62915953b5`; see the [wheel verification](validation/llamaindex-proposal/published-baseline.json). All 58 existing unit tests passed before these changes.

| Finding | Evidence and consequence | Change in 0.2 |
| --- | --- | --- |
| Get-memory cannot read ordinary text | Live get-memory raised `JSONDecodeError`; the code calls `.json()` on a text content response. | Use SDK content bytes and decode according to the content type. Keep metadata when downloading content fails. |
| Source metadata disappears | Live search returned chunks, but omitted stored `source` and `title`. The converter copies a few SDK fields rather than user metadata. | Return native `NodeWithScore` objects with stored metadata, stable chunk IDs and SOURCE relationships. |
| Retrieval failures are hidden | A live invalid-reranker request returned five chunks without its failure notice. The NDJSON parser ignores all status events. | Keep statuses and `partial` in administrative results; raise known failures in the standard retriever. Unknown codes never abort retrieval. |
| Empty search means a minute of retries | A mock HTTP test observed repeated requests and sleep after a legitimate empty response. | Search once. Wait explicitly for batches of the memory IDs that were written. |
| Pagination stops after the first page | A mock response with `nextToken` caused no second request. Name-based reuse can consequently miss spaces; it also echoes the requested embedder even when reusing another configuration. | Use SDK pagination; creation always creates and surfaces duplicate-name conflicts. |
| Malformed responses look successful | Invalid NDJSON is silently discarded and becomes an empty result. | Delegate stream parsing and HTTP errors to the official SDK. |
| Basic framework interfaces are absent | Only `BaseToolSpec` is exported; no retriever, Document ingestion or retrieval metadata filters. The README’s `ReActAgent.from_tools` does not exist in the tested framework version. | Add `GoodMemRetriever`, Document ingestion, supported native filters, and current workflow-based examples. |
| Obsolete and overly broad tools | The update schema still exposes `public_read`; create-memory accepts arbitrary local paths. | Remove obsolete fields; make file upload an explicit opt-in restricted to a configured directory. |

The baseline [mock report](validation/llamaindex-proposal/baseline.json) and [live report](validation/llamaindex-proposal/baseline-live.json) preserve the observations.

## What typical users gain

The shared retriever works with LlamaIndex’s `RetrieverTool`, `RetrieverQueryEngine`, callbacks and native async calls. Developers configure collection scope and filters; the model sees only the query input. Reranking requests extra candidates and needs no GoodMem LLM. Both the retriever and ingestor use the official synchronous or asynchronous SDK directly.

Document ingestion writes whole documents through the batch API. Accepted IDs remain available after partial failures. Waiting is a separate operation with a caller-selected budget, and completed memories disappear from later status requests. No empty-search polling or compatibility client remains.

These changes belong to the reusable integration. Production Python source fell from **1,102 to 1,018 lines, 7.6% smaller**, including comments, blank lines and all review fixes. The demo’s retrieval configuration is 30 lines using native framework wrappers. See the [release comparison](validation/release-0.2.0/comparison.json).

## Fixes found during review of the proposal

Two review rounds reproduced seven defects introduced or left incomplete in the proposed integration. All seven are fixed in 0.2.0:

- Injecting one SDK client can no longer make calls in the other mode silently use a different server or credential from environment settings. Both modes require their corresponding injected clients.
- Vector scores now follow LlamaIndex's higher-is-better convention; successful reranker scores retain their direction. Regression tests cover standard fusion and similarity postprocessing.
- If indexing confirmation fails after a write succeeds, administrative tools return the accepted ID, an unconfirmed indexing outcome and recovery guidance. The agent can check the existing memory without uploading it again.
- Content download transport failures preserve metadata in both sync and async calls with caller-supplied HTTP clients.
- Document LLM and embedding metadata exclusions survive ingestion and retrieval. Standard retrieval tools keep excluded fields out of their rendered output.
- Raw scores, score kind and statuses live on `GoodMemNodeWithScore`, outside node metadata and identity. Two-query tests verify that repeated chunks combine correctly in all four fusion modes, in sync and async calls.
- Inequality, excluded membership and negated comparisons explicitly handle missing/null fields. Generated filters are checked against LlamaIndex's evaluator and a live GoodMem server.

These cases were missing from the first proposal's 41 tests. The release has 119 offline tests, including the actual agent tool runner's recovery behavior. All 16 demo retrieval checks passed with the published wheel; see the [release retrieval report](validation/release-0.2.0/retrieval.json).

## What the rebuilt application establishes

The six source documents are byte-for-byte identical to the LangChain port’s corpus. The same embedder, reranker and chat model are used, with separate spaces and the same acceptance questions. Both plain and reranked retrieval passed all 16 checks.

All four notebooks executed successfully again with the published 0.2.0 package; see the [notebook results](validation/release-0.2.0/notebooks.json). The last full agent evaluation passed **7 of 8** cases. The explicit workflow passed all four cases. ReAct completed the dependent two-search case but cited a link inside a passage rather than a retrieved source, failing the citation gate. An earlier ReAct run passed all four, so the final result also exposes model variability. We retain the failure in the [evaluation report](validation/llamaindex-proposal/evaluation.json); this agent evaluation predates the integration review fixes.

The citation gate checks source provenance, not whether every generated claim is entailed. Some intermediate answers also made unsupported comparisons before consulting both collections. Agent behavior needs broader evaluation before claiming a quality advantage. GoodMem retrieval, metadata preservation and the shared SDK interfaces have separate deterministic and live coverage.

## LlamaIndex rough edges outside our integration

The tested Cohere adapter drops enum constraints when converting function schemas. JSON containing long Markdown answers was also fragile. The final explicit workflow uses native tool calls for search decisions and ordinary text for answers. Small classification and grading outputs use LlamaIndex’s text-based Pydantic program. ReAct uses its framework-provided loop.

The native `RetrieverTool` formats keyword queries as text such as `input is ...`. The port uses that standard wrapper as shipped; it does not introduce a custom tool solely to change this formatting. A no-text query-engine mode also attempted to resolve the default OpenAI LLM despite an explicit mock LLM; ordinary query-engine tests work when the model is supplied.

## Review and release scope

All 119 offline integration tests pass on Python 3.10 and 3.13, including unknown-code HTTP responses before and after valid chunks. The wheel downloaded from PyPI passes the same 119 tests outside the source checkout on Python 3.13. Four live tests pass, covering text round trips, ingestion, metadata exclusions, escaped filter values, missing/null comparisons, reranking without an LLM, score conversion and partial diagnostics. The demo's 10 unit tests pass, and its installed modules match the published wheel. LlamaIndex's agent initialization emits four upstream Pydantic deprecation warnings in the agent recovery test.

The clean API update removes old JSON shapes and obsolete flags. It supports common scalar metadata operators and nested conditions. Unsupported operators fail explicitly. URL downloading remains application code; this package supplies retrieval and ingestion rather than a VectorStore accepting precomputed embeddings.

The updated monthly harness passes 18 checks against the published wheel; its optional LLM-summary check is skipped because the test server has no LLM configured. Release artifacts and the remaining agent citation limitation are recorded in [release verification](validation/release-0.2.0/verification.json).
