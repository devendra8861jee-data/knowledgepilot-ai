"""Build citations that point to real retrieved chunks only.

The LLM sees numbered sources ``[1]``, ``[2]``... and is asked to cite with those
numbers. We never trust the model for filenames or pages: we parse the numbers
and look the metadata up in the chunks we actually retrieved. Numbers that do
not map to a retrieved chunk are removed and reported.
"""

from __future__ import annotations

import re

from src.models import Citation, RetrievedChunk

_MARKER_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
EXCERPT_CHARS = 300


def extract_citation_indices(answer: str) -> list[int]:
    """Return cited source numbers in order of first appearance, without duplicates."""
    seen: list[int] = []
    for match in _MARKER_RE.finditer(answer or ""):
        for part in match.group(1).split(","):
            number = int(part.strip())
            if number not in seen:
                seen.append(number)
    return seen


def make_excerpt(text: str, max_chars: int = EXCERPT_CHARS) -> str:
    """Return the beginning of ``text`` (a true substring) cut at a word boundary."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip() + "..."


def build_citations(
    answer: str, sources: list[RetrievedChunk]
) -> tuple[list[Citation], list[int]]:
    """Map ``[n]`` markers in ``answer`` to ``sources`` (1-based list positions).

    Returns:
        (citations, invalid_indices) where ``invalid_indices`` are numbers the model
        used that do not correspond to any provided source.
    """
    citations: list[Citation] = []
    invalid: list[int] = []
    for number in extract_citation_indices(answer):
        if 1 <= number <= len(sources):
            src = sources[number - 1]
            chunk = src.chunk
            citations.append(
                Citation(
                    index=number,
                    filename=chunk.filename,
                    page=chunk.page,
                    page_label=chunk.page_label,
                    chunk_id=chunk.chunk_id,
                    excerpt=make_excerpt(chunk.text),
                    score=src.score,
                )
            )
        else:
            invalid.append(number)
    return citations, invalid


def strip_invalid_markers(answer: str, invalid: list[int]) -> str:
    """Remove ``[n]`` markers that do not map to a real source."""
    if not invalid:
        return answer
    bad = set(invalid)

    def _replace(match: re.Match[str]) -> str:
        numbers = [int(p.strip()) for p in match.group(1).split(",")]
        kept = [n for n in numbers if n not in bad]
        return f"[{', '.join(str(n) for n in kept)}]" if kept else ""

    cleaned = _MARKER_RE.sub(_replace, answer)
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()


def format_citation_label(citation: Citation) -> str:
    """Human readable label, e.g. ``handbook.pdf, page 3 (printed label: 1)``."""
    label = f"{citation.filename}, page {citation.page}"
    if citation.page_label and citation.page_label != str(citation.page):
        label += f" (printed label: {citation.page_label})"
    return label
