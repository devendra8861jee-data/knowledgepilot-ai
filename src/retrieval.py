"""Semantic retrieval with lightweight lexical reranking and qualifier guards."""

from __future__ import annotations

import logging
import re
from typing import Optional

from src.embeddings import Embedder
from src.models import RetrievedChunk
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Keep this intentionally small and transparent: it is a lexical tie-breaker,
# not a replacement for semantic embeddings.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,:][0-9]+)*", re.IGNORECASE)
_STOPWORDS = frozenset(
    "a an the of to in on for and or is are was were be been by with at as it its "
    "this that do does did how what which who when where why can i you we they from per "
    "about into have has had support supported get give tell explain please".split()
)
# Preserve distinctions such as 5G cellular vs 5 GHz Wi-Fi. Normalized anchors
# remove spaces, so "5 GHz" becomes "5ghz" while "5G" remains "5g".
_TECHNICAL_ANCHOR_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:g|ghz|mhz|khz|gbps|mbps|tbps|gb|mb|tb|lte)\b",
    re.IGNORECASE,
)


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text or "") if token.lower() not in _STOPWORDS}


def _technical_anchors(text: str) -> set[str]:
    return {re.sub(r"\s+", "", match.group(0).lower()) for match in _TECHNICAL_ANCHOR_RE.finditer(text or "")}


class Retriever:
    """Retrieve semantic candidates, then rerank using exact lexical overlap.

    Similarity scores remain the original vector-store cosine similarities so the
    evidence threshold stays interpretable. Lexical overlap affects ordering only.
    """

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        top_k: int = 4,
        min_relevance: float = 0.10,
        candidate_multiplier: int = 3,
        max_candidates: int = 24,
        lexical_weight: float = 0.25,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if candidate_multiplier < 1 or max_candidates < 1:
            raise ValueError("candidate limits must be at least 1.")
        if not 0.0 <= lexical_weight <= 1.0:
            raise ValueError("lexical_weight must be between 0 and 1.")
        self.store = store
        self.embedder = embedder
        self.top_k = top_k
        self.min_relevance = min_relevance
        self.candidate_multiplier = candidate_multiplier
        self.max_candidates = max_candidates
        self.lexical_weight = lexical_weight

    def retrieve(
        self,
        question: str,
        top_k: Optional[int] = None,
        doc_ids: Optional[list[str]] = None,
    ) -> list[RetrievedChunk]:
        """Return the best candidates, combining semantic relevance and token overlap."""
        question = (question or "").strip()
        if not question:
            return []

        final_k = top_k or self.top_k
        if final_k < 1:
            return []
        query_vec = self.embedder.embed_query(question)
        candidate_k = min(max(final_k, final_k * self.candidate_multiplier), self.max_candidates)
        candidates = self.store.query(query_vec, candidate_k, doc_ids=doc_ids)
        query_tokens = _tokens(question)

        def hybrid_score(item: RetrievedChunk) -> float:
            doc_tokens = _tokens(item.chunk.text)
            coverage = len(query_tokens & doc_tokens) / max(len(query_tokens), 1)
            return (1.0 - self.lexical_weight) * item.score + self.lexical_weight * coverage

        ranked = sorted(candidates, key=hybrid_score, reverse=True)[:final_k]
        return [
            RetrievedChunk(chunk=item.chunk, score=item.score, rank=rank)
            for rank, item in enumerate(ranked, start=1)
        ]

    def select_evidence(
        self,
        retrieved: list[RetrievedChunk],
        question: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        """Filter weak matches and reject mismatched numeric/technical qualifiers.

        If the query contains a precise technical quantity such as ``5G`` or
        ``5 GHz``, a candidate must contain that same normalized qualifier. This
        prevents semantically similar but materially different concepts from being
        passed to the generator as evidence.
        """
        anchors = _technical_anchors(question or "")
        selected = []
        for item in retrieved:
            if item.score < self.min_relevance:
                continue
            if anchors:
                passage_anchors = _technical_anchors(item.chunk.text)
                if not anchors.intersection(passage_anchors):
                    continue
            selected.append(item)
        return selected
