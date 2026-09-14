"""Upload whole documentation pages; GoodMem owns chunking and embedding."""

from __future__ import annotations

import hashlib
import os
import uuid
from collections.abc import Callable

import httpx
from bs4 import BeautifulSoup
from goodmem import Goodmem
from goodmem.errors import ConflictError, NotFoundError
from llama_index.core.schema import Document
from llama_index.tools.goodmem import GoodMemDocumentIngestor, wait_for_memories

from .config import Settings, save_json

SOURCES = {
    "langgraph": [
        "https://docs.langchain.com/oss/python/langgraph/overview",
        "https://docs.langchain.com/oss/python/langgraph/workflows-agents",
        "https://docs.langchain.com/oss/python/langgraph/graph-api",
    ],
    "langchain": [
        "https://docs.langchain.com/oss/python/langchain/overview",
        "https://docs.langchain.com/oss/python/langchain/models",
        "https://docs.langchain.com/oss/python/langchain/tools",
    ],
}
CHUNKING = {
    "recursive": {
        "chunkSize": 1000,
        "chunkOverlap": 200,
        "lengthMeasurement": "CHARACTER_COUNT",
        "keepStrategy": "KEEP_END",
    }
}


def source_document(url: str, http: httpx.Client) -> dict:
    """Prefer the official Markdown representation; fall back to article HTML."""
    response = http.get(url + ".md")
    if response.status_code == 404:
        response = http.get(url)
    response.raise_for_status()
    if "text/html" in response.headers.get("content-type", ""):
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        article = soup.find("article") or soup.find("main")
        if article is None:
            raise ValueError(f"No article found at {url}; refusing to ingest a navigation page")
        text = article.get_text("\n", strip=True)
        title = soup.title.get_text(strip=True) if soup.title else url.rsplit("/", 1)[-1]
    else:
        text = response.text
        title = next((line[2:].strip() for line in text.splitlines() if line.startswith("# ")), url)
    if len(text.strip()) < 200:
        raise ValueError(f"Suspiciously short documentation response from {url}")
    return {
        "source": url,
        "title": title,
        "text": text,
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def _ensure(get: Callable, create: Callable):
    try:
        return get()
    except NotFoundError:
        try:
            return create()
        except ConflictError:
            # Only a conflict is recoverable. Auth/network/validation errors must propagate.
            return get()


def setup(settings: Settings, *, initialize: bool = False, with_reranker: bool = False) -> dict:
    if initialize:
        try:
            settings.key()
        except ValueError:
            with Goodmem(base_url=settings.base_url, timeout=settings.timeout) as client:
                initialized = client.system.init()
            if not initialized.root_api_key:
                raise ValueError("Server already initialized; supply its GOODMEM_API_KEY")
            save_json(
                settings.runtime / "credentials.json",
                {
                    "base_url": settings.base_url,
                    "api_key": initialized.root_api_key,
                },
            )
    try:
        previous = settings.state()
    except ValueError:
        previous = {}
    state = {
        "base_url": settings.base_url,
        "namespace": settings.namespace,
        "spaces": {},
        "documents": [],
        "chunking": CHUNKING,
    }
    identity = uuid.uuid5(uuid.NAMESPACE_URL, f"llamaindex-goodmem-rag:{settings.namespace}")
    with settings.client() as client:
        model = os.getenv("GOODMEM_EMBEDDING_MODEL") or "embed-v4.0"
        embedder_id = os.getenv("GOODMEM_EMBEDDER_ID")
        if embedder_id:
            client.embedders.get(id=embedder_id)
        else:
            embedder_id = str(uuid.uuid5(identity, f"embedder:{model}"))
            key = os.getenv("EMBEDDING_API_KEY") or os.getenv("COHERE_API_KEY")

            def create_embedder():
                if not key:
                    raise ValueError(
                        "Set GOODMEM_EMBEDDER_ID or EMBEDDING_API_KEY / COHERE_API_KEY"
                    )
                return client.embedders.create(
                    embedder_id=embedder_id,
                    display_name=f"{settings.namespace}: {model}",
                    model_identifier=model,
                    api_key=key,
                )

            _ensure(lambda: client.embedders.get(id=embedder_id), create_embedder)
        state["embedder_id"] = embedder_id
        reranker_id = os.getenv("GOODMEM_RERANKER_ID") or previous.get("reranker_id")
        if reranker_id:
            client.rerankers.get(id=reranker_id)
        elif with_reranker:
            rerank_model = os.getenv("GOODMEM_RERANKER_MODEL") or "rerank-v3.5"
            reranker_id = str(uuid.uuid5(identity, f"reranker:{rerank_model}"))
            _ensure(
                lambda: client.rerankers.get(id=reranker_id),
                lambda: client.rerankers.create(
                    reranker_id=reranker_id,
                    display_name=f"{settings.namespace}: {rerank_model}",
                    model_identifier=rerank_model,
                    api_key=os.environ["COHERE_API_KEY"],
                ),
            )
        state["reranker_id"] = reranker_id
        with httpx.Client(
            timeout=60, follow_redirects=True, headers={"User-Agent": "Agentic-RAG-GoodMem/0.1"}
        ) as http:
            for collection, urls in SOURCES.items():
                # Embedder changes must create a different space (embedders are immutable).
                space_id = str(uuid.uuid5(identity, f"{collection}:{embedder_id}"))
                _ensure(
                    lambda: client.spaces.get(id=space_id),
                    lambda: client.spaces.create(
                        space_id=space_id,
                        name=f"{settings.namespace}: {collection}",
                        space_embedders=[{"embedder_id": embedder_id}],
                        default_chunking_config=CHUNKING,
                        labels={
                            "application": "agentic-rag-llamaindex-goodmem",
                            "collection": collection,
                        },
                    ),
                )
                state["spaces"][collection] = space_id
                current_ids = set()
                for url in urls:
                    document = source_document(url, http)
                    memory_id = str(uuid.uuid5(uuid.UUID(space_id), url + document["sha256"]))
                    try:
                        client.memories.get(id=memory_id)
                    except NotFoundError:
                        GoodMemDocumentIngestor(client=client, space_id=space_id).add_documents(
                            [
                                Document(
                                    id_=memory_id,
                                    text=document["text"],
                                    metadata={k: v for k, v in document.items() if k != "text"}
                                    | {"application": "agentic-rag-llamaindex-goodmem"},
                                )
                            ]
                        )
                    wait_for_memories(client, [memory_id], settings.ingest_timeout)
                    current_ids.add(memory_id)
                    state["documents"].append(
                        {k: v for k, v in document.items() if k != "text"}
                        | {"memory_id": memory_id, "collection": collection}
                    )
                    print(f"Indexed {collection}: {document['title']}", flush=True)
                # Replace changed pages only after their replacements are searchable.
                for old in client.memories.list(space_id=space_id):
                    if (old.metadata or {}).get(
                        "application"
                    ) == "agentic-rag-llamaindex-goodmem" and (old.memory_id not in current_ids):
                        client.memories.delete(id=old.memory_id)
        save_json(settings.runtime / "state.json", state)
    return state
