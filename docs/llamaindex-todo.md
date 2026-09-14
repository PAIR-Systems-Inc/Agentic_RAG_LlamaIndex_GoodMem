# LlamaIndex port

- [x] Check the published integration and reproduce failures with mock HTTP and a live server.
- [x] Add reusable SDK-backed retrieval and Document ingestion to goodmem-llamaindex.
- [x] Replace obsolete tool behavior and add regression tests at the HTTP boundary.
- [x] Rebuild the four teaching stages using LlamaIndex workflows and agents.
- [x] Run the same retrieval, citation, routing and dependent-search acceptance cases.
- [x] Record the findings, migration requirements and limitations for review.
- [x] Prevent injected sync/async clients from silently falling back to other connection settings.
- [x] Correct framework score direction and verify standard fusion/postprocessing behavior.
- [x] Return accepted write receipts to agents after indexing confirmation failures.
- [x] Preserve metadata after custom HTTP client transport failures during content download.
- [x] Validate the review fixes on Python 3.10/3.13, the built wheel, the live server and the demo corpus.
- [x] Preserve Document LLM and embedding metadata exclusions through storage and retrieval.
- [x] Keep query-dependent diagnostics outside node identity and test two-query fusion in all four modes.
- [x] Match LlamaIndex's missing/null behavior for negated metadata filters.
- [x] Commit, push and publish llamaindex-goodmem 0.2.0 to PyPI.
- [x] Replace the demo's local dependency override with the published integration and verify its installed files.
- [x] Run all four notebooks, 16 retrieval checks and 10 demo unit tests using the release.
- [x] Update and validate the monthly e2e harness and the docs page in the GoodMem monorepo.

Working directories: `Agentic_RAG_LlamaIndex_GoodMem` and `goodmem-llamaindex`.
The demo uses the published `llamaindex-goodmem 0.2.0` package. See the [release verification](validation/release-0.2.0/verification.json).

The last full agent evaluation has a documented citation failure (7/8 cases). The release notebook runs passed, but do not replace that broader agent evaluation.
