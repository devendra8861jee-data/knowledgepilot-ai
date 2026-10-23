"""Split page text into overlapping chunks without crossing page boundaries."""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.models import Chunk, ParsedDocument


def validate_chunk_params(chunk_size: int, chunk_overlap: int) -> None:
    """Raise ``ValueError`` with a helpful message for invalid settings."""
    if chunk_size < 100:
        raise ValueError("chunk_size must be at least 100 characters.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")


def chunk_document(
    document: ParsedDocument, chunk_size: int = 800, chunk_overlap: int = 120
) -> list[Chunk]:
    """Chunk every page separately so each chunk has exactly one page number.

    Chunking per page costs a little context at page borders but guarantees
    that a citation's page number is always correct.
    """
    validate_chunk_params(chunk_size, chunk_overlap)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Chunk] = []
    for page in document.pages:
        for page_chunk_index, piece in enumerate(splitter.split_text(page.text)):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}-p{page.page_number}-c{page_chunk_index}",
                    doc_id=document.doc_id,
                    filename=document.filename,
                    page=page.page_number,
                    chunk_index=page_chunk_index,
                    text=piece,
                    page_label=page.page_label,
                )
            )
    return chunks
