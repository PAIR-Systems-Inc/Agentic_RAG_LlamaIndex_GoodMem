# Upstream provenance

Source: https://github.com/ChandulaSenevirathna/Agentic_RAG

Adaptation base: `6153d18a7922cefa9528086dbfaf8597a77bf74a` (`new patterns`). The original Git history is retained; local remote `upstream` points to the source repository.

Original author: Chandula Senevirathna. The unchanged `LICENSE.md` remains authoritative for upstream material. The upstream repository contains four notebooks, while its license text refers to older filenames and includes different terms for a free demo and paid materials. No paid notebooks were recreated or obtained separately.

The earlier LangChain + GoodMem adaptation kept the two graph tutorials, the explicit retrieve/grade/draft/rewrite pattern, and the prebuilt ReAct pattern. The two RAG notebooks now share executable modules with the CLI and tests. GoodMem replaces Chroma, local BGE-M3, and client-side splitting. The same six source page paths are used, with official Markdown preferred over full-page HTML scraping.

The original README is available in Git at the base commit. Descriptions of paid notebooks in that README are upstream documentation, not additional implemented features in this adaptation.


## LlamaIndex adaptation

This working tree starts from the public GoodMem port and retains its Git history and original license notice. The four notebooks have been rewritten around LlamaIndex typed workflows, native retrieval tools and its workflow-based ReAct agent. GoodMem still owns document chunking and embeddings. The source corpus and acceptance questions are retained for comparison.

Integration changes live in the separate [PAIR-Systems-Inc/goodmem-llamaindex](https://github.com/PAIR-Systems-Inc/goodmem-llamaindex) repository. This demo installs its published `llamaindex-goodmem 0.2.0` release from PyPI.
