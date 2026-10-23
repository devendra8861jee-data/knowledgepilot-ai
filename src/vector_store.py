"""ChromaDB-backed vector store.

We always pass our own embeddings to Chroma, so Chroma never uses its built-in
embedding function. The collection name includes a hash of the embedder name:
vectors from different embedding models are never mixed in one collection.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.models import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Raised for vector-store problems (user-facing message)."""


def collection_name_for(embedder_name: str, prefix: str = "kp") -> str:
    """Return a valid Chroma collection name unique to this embedder."""
    digest = hashlib.sha1(embedder_name.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"


class VectorStore:
    """Thin wrapper over a persistent Chroma collection."""

    def __init__(self, persist_dir: Path | str, embedder_name: str, prefix: str = "kp") -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name_for(embedder_name, prefix)
        try:
            self._client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name, metadata={"hnsw:space": "cosine"}
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Could not open the vector database at '{self.persist_dir}': {exc}"
            ) from exc

    # ------------------------------------------------------------------ writes
    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Insert chunks with their embeddings and metadata."""
        if len(chunks) != len(embeddings):
            raise VectorStoreError("Number of chunks and embeddings must match.")
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "doc_id": c.doc_id,
                    "filename": c.filename,
                    "page": c.page,
                    "page_label": c.page_label,
                    "chunk_index": c.chunk_index,
                }
                for c in chunks
            ],
        )

    def delete_document(self, doc_id: str) -> None:
        """Remove every chunk of a document."""
        self._collection.delete(where={"doc_id": doc_id})

    def clear(self) -> None:
        """Remove everything in this collection."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )

    # ------------------------------------------------------------------- reads
    def count(self) -> int:
        return self._collection.count()

    def has_document(self, doc_id: str) -> bool:
        found = self._collection.get(where={"doc_id": doc_id}, limit=1, include=[])
        return len(found["ids"]) > 0

    def list_documents(self) -> list[dict]:
        """Summaries of indexed documents: doc_id, filename, pages, chunks."""
        data = self._collection.get(include=["metadatas"])
        summary: dict[str, dict] = {}
        for meta in data["metadatas"] or []:
            entry = summary.setdefault(
                meta["doc_id"],
                {
                    "doc_id": meta["doc_id"],
                    "filename": meta["filename"],
                    "chunks": 0,
                    "pages": set(),
                },
            )
            entry["chunks"] += 1
            entry["pages"].add(int(meta["page"]))
        result = []
        for entry in summary.values():
            entry["pages"] = len(entry["pages"])
            result.append(entry)
        return sorted(result, key=lambda e: e["filename"].lower())

    def _chunk_from_row(self, chunk_id: str, text: str, meta: dict) -> Chunk:
        """Build a Chunk from a Chroma row."""
        return Chunk(
            chunk_id=chunk_id,
            doc_id=meta["doc_id"],
            filename=meta["filename"],
            page=int(meta["page"]),
            chunk_index=int(meta["chunk_index"]),
            text=text,
            page_label=meta.get("page_label", "") or "",
        )

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """Fetch a single indexed chunk by its exact ID."""
        try:
            result = self._collection.get(ids=[chunk_id], include=["documents", "metadatas"])
        except Exception as exc:
            raise VectorStoreError(f"Could not fetch source chunk: {exc}") from exc
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        if not ids or not documents or not metadatas:
            return None
        return self._chunk_from_row(ids[0], documents[0], metadatas[0])

    def get_page_chunks(self, doc_id: str, page: int) -> list[Chunk]:
        """Fetch chunks for one physical page of one document, ordered by chunk index."""
        try:
            result = self._collection.get(
                where={"$and": [{"doc_id": doc_id}, {"page": int(page)}]},
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            raise VectorStoreError(f"Could not fetch page chunks: {exc}") from exc
        chunks = [
            self._chunk_from_row(chunk_id, text, meta)
            for chunk_id, text, meta in zip(
                result.get("ids") or [],
                result.get("documents") or [],
                result.get("metadatas") or [],
            )
        ]
        return sorted(chunks, key=lambda chunk: (chunk.chunk_index, chunk.chunk_id))

    def query(
        self,
        query_embedding: list[float],
        top_k: int,
        doc_ids: Optional[list[str]] = None,
    ) -> list[RetrievedChunk]:
        """Return the ``top_k`` most similar chunks (optionally within some documents)."""
        total = self.count()
        if total == 0 or top_k <= 0:
            return []

        where = None
        if doc_ids:
            where = {"doc_id": doc_ids[0]} if len(doc_ids) == 1 else {"doc_id": {"$in": doc_ids}}

        try:
            res = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, total),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError(f"Vector search failed: {exc}") from exc

        results: list[RetrievedChunk] = []
        ids = res["ids"][0]
        for rank, (chunk_id, text, meta, dist) in enumerate(
            zip(ids, res["documents"][0], res["metadatas"][0], res["distances"][0]), start=1
        ):
            chunk = Chunk(
                chunk_id=chunk_id,
                doc_id=meta["doc_id"],
                filename=meta["filename"],
                page=int(meta["page"]),
                chunk_index=int(meta["chunk_index"]),
                text=text,
                page_label=meta.get("page_label", "") or "",
            )
            # Chroma cosine distance = 1 - cosine similarity
            results.append(RetrievedChunk(chunk=chunk, score=1.0 - float(dist), rank=rank))
        return results
