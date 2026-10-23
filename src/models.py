"""Shared data structures used across the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class PageText:
    """Text extracted from a single PDF page."""

    page_number: int  # 1-based physical page index in the PDF file
    text: str
    page_label: str = ""  # printed page label (e.g. "iv", "12") when the PDF defines one


@dataclass(frozen=True)
class ParsedDocument:
    """A PDF that was successfully parsed."""

    doc_id: str  # SHA-256 (first 16 hex chars) of the file bytes
    filename: str
    num_pages: int
    pages: list[PageText]
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Chunk:
    """A piece of text that stays inside exactly one page of one document."""

    chunk_id: str
    doc_id: str
    filename: str
    page: int
    chunk_index: int
    text: str
    page_label: str = ""


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by the vector store together with its similarity score."""

    chunk: Chunk
    score: float  # cosine similarity, higher is more similar
    rank: int  # 1-based position in the result list


@dataclass(frozen=True)
class Citation:
    """A citation that always points to a really retrieved chunk."""

    index: int  # the [n] marker shown in the answer
    filename: str
    page: int
    page_label: str
    chunk_id: str
    excerpt: str
    score: float


@dataclass
class IngestReport:
    """Outcome of trying to index one uploaded file."""

    filename: str
    status: str  # "indexed" | "duplicate" | "failed"
    message: str
    doc_id: Optional[str] = None
    num_pages: int = 0
    num_chunks: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class AnswerResult:
    """Everything the UI and the evaluator need to know about one answer."""

    question: str
    answer: str
    sufficient_evidence: bool
    citations: list[Citation] = field(default_factory=list)
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    mode: str = "extractive"  # "llm" | "extractive" | "fallback" | "error"
    warnings: list[str] = field(default_factory=list)
    error: Optional[str] = None
    retrieval_latency_s: float = 0.0
    generation_latency_s: float = 0.0
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

    @property
    def total_latency_s(self) -> float:
        return self.retrieval_latency_s + self.generation_latency_s
