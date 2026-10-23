"""High-level workflows: ingest a PDF and answer a question."""

from __future__ import annotations

import logging
import time
from typing import Optional

from config import FALLBACK_MESSAGE
from src.chunking import chunk_document
from src.citations import build_citations, strip_invalid_markers
from src.embeddings import Embedder, EmbeddingError
from src.generation import (
    LLMClient,
    LLMError,
    build_messages,
    extractive_answer,
)
from src.ingestion import IngestionError, compute_doc_id, parse_pdf
from src.models import AnswerResult, IngestReport
from src.retrieval import Retriever
from src.vector_store import VectorStore, VectorStoreError

logger = logging.getLogger(__name__)


def ingest_pdf(
    data: bytes,
    filename: str,
    store: VectorStore,
    embedder: Embedder,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> IngestReport:
    """Parse, chunk, embed and index one PDF. Never raises for bad user files."""
    try:
        doc_id = compute_doc_id(data) if data else ""
        if doc_id and store.has_document(doc_id):
            return IngestReport(
                filename=filename,
                status="duplicate",
                message="Already indexed (identical file content). Skipped.",
                doc_id=doc_id,
            )

        document = parse_pdf(data, filename)
        chunks = chunk_document(document, chunk_size, chunk_overlap)
        if not chunks:
            raise IngestionError(f"'{document.filename}' produced no text chunks.")

        vectors = embedder.embed_documents([c.text for c in chunks])
        store.add_chunks(chunks, vectors)
        logger.info("Indexed %s: %d pages, %d chunks", document.filename, len(document.pages), len(chunks))
        return IngestReport(
            filename=document.filename,
            status="indexed",
            message=f"Indexed {len(chunks)} chunks from {len(document.pages)} page(s).",
            doc_id=document.doc_id,
            num_pages=document.num_pages,
            num_chunks=len(chunks),
            warnings=list(document.warnings),
        )
    except (IngestionError, EmbeddingError, VectorStoreError, ValueError) as exc:
        return IngestReport(filename=filename, status="failed", message=str(exc))
    except Exception as exc:  # last-resort guard so one bad file never crashes the app
        logger.exception("Unexpected ingestion failure for %s", filename)
        return IngestReport(
            filename=filename,
            status="failed",
            message=f"Unexpected error while indexing: {type(exc).__name__}.",
        )


def answer_question(
    question: str,
    retriever: Retriever,
    llm: Optional[LLMClient] = None,
    doc_ids: Optional[list[str]] = None,
    top_k: Optional[int] = None,
) -> AnswerResult:
    """Retrieve evidence and produce an answer with verified citations."""
    question = (question or "").strip()
    if not question:
        return AnswerResult(
            question=question,
            answer="Please type a question.",
            sufficient_evidence=False,
            mode="fallback",
        )

    t0 = time.perf_counter()
    try:
        retrieved = retriever.retrieve(question, top_k=top_k, doc_ids=doc_ids)
    except (EmbeddingError, VectorStoreError) as exc:
        return AnswerResult(
            question=question,
            answer="",
            sufficient_evidence=False,
            mode="error",
            error=str(exc),
            retrieval_latency_s=time.perf_counter() - t0,
        )
    retrieval_latency = time.perf_counter() - t0

    evidence = retriever.select_evidence(retrieved, question=question)
    if not evidence:
        return AnswerResult(
            question=question,
            answer=FALLBACK_MESSAGE,
            sufficient_evidence=False,
            retrieved=retrieved,
            mode="fallback",
            retrieval_latency_s=retrieval_latency,
        )

    if llm is None:
        answer = extractive_answer(evidence)
        citations, _ = build_citations(answer, evidence)
        return AnswerResult(
            question=question,
            answer=answer,
            sufficient_evidence=True,
            citations=citations,
            retrieved=retrieved,
            mode="extractive",
            retrieval_latency_s=retrieval_latency,
        )

    t1 = time.perf_counter()
    try:
        response = llm.generate(build_messages(question, evidence))
    except LLMError as exc:
        return AnswerResult(
            question=question,
            answer="",
            sufficient_evidence=True,
            retrieved=retrieved,
            mode="error",
            error=str(exc),
            retrieval_latency_s=retrieval_latency,
            generation_latency_s=time.perf_counter() - t1,
        )
    generation_latency = time.perf_counter() - t1

    text = response.text.strip()
    base = dict(
        question=question,
        retrieved=retrieved,
        retrieval_latency_s=retrieval_latency,
        generation_latency_s=generation_latency,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
    )

    if not text:
        return AnswerResult(
            answer="",
            sufficient_evidence=True,
            mode="error",
            error="The LLM returned an empty answer. Please try again.",
            **base,
        )

    if FALLBACK_MESSAGE.lower().rstrip(".") in text.lower():
        return AnswerResult(
            answer=FALLBACK_MESSAGE, sufficient_evidence=False, mode="fallback", **base
        )

    citations, invalid = build_citations(text, evidence)
    warnings: list[str] = []
    if invalid:
        text = strip_invalid_markers(text, invalid)
        warnings.append(
            "The model referred to source numbers that do not exist "
            f"({', '.join(str(n) for n in invalid)}). Those markers were removed."
        )
    if not citations:
        warnings.append(
            "The answer contains no valid citation. Treat it as unverified and check the "
            "retrieved passages below."
        )
    return AnswerResult(
        answer=text,
        sufficient_evidence=True,
        citations=citations,
        mode="llm",
        warnings=warnings,
        **base,
    )
