"""PDF ingestion: extract text per page and keep document metadata.

Page numbers
------------
``PageText.page_number`` is the *physical* 1-based position of the page in the
PDF file (what a PDF viewer shows in its page box). Many PDFs have cover pages
or roman-numeral front matter, so the number printed on the page can differ.
When the PDF defines page labels we also store them in ``page_label``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re

import pymupdf

from src.models import PageText, ParsedDocument

logger = logging.getLogger(__name__)

MIN_CHARS_PER_PAGE = 20  # pages with fewer characters are treated as "no text"


class IngestionError(Exception):
    """Raised when a PDF cannot be turned into usable text.

    The message is written for end users and is safe to show in the UI.
    """


def compute_doc_id(data: bytes) -> str:
    """Return a stable id for a file based on its content (not its name)."""
    return hashlib.sha256(data).hexdigest()[:16]


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_pdf(data: bytes, filename: str) -> ParsedDocument:
    """Extract text from PDF bytes, page by page.

    Raises:
        IngestionError: for empty, malformed, encrypted or scanned (image-only) PDFs.
    """
    safe_name = os.path.basename(filename or "uploaded.pdf") or "uploaded.pdf"

    if not data:
        raise IngestionError(f"'{safe_name}' is empty (0 bytes).")

    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several exception types for bad files
        logger.warning("Could not open %s: %s", safe_name, exc)
        raise IngestionError(
            f"'{safe_name}' could not be opened. The file may be corrupted or not a real PDF."
        ) from exc

    try:
        if doc.needs_pass:
            raise IngestionError(
                f"'{safe_name}' is password protected. Remove the password and upload it again."
            )
        if doc.page_count == 0:
            raise IngestionError(f"'{safe_name}' has no pages.")

        pages: list[PageText] = []
        empty_pages: list[int] = []
        for index in range(doc.page_count):
            try:
                page = doc.load_page(index)
                text = _clean_text(page.get_text("text"))
                label = page.get_label() or ""
            except Exception as exc:
                logger.warning("Failed to read page %d of %s: %s", index + 1, safe_name, exc)
                empty_pages.append(index + 1)
                continue
            if len(text) < MIN_CHARS_PER_PAGE:
                empty_pages.append(index + 1)
                continue
            pages.append(PageText(page_number=index + 1, text=text, page_label=label))

        if not pages:
            raise IngestionError(
                f"'{safe_name}' contains no extractable text. It is probably a scanned PDF "
                "(images only). Run OCR first (for example with OCRmyPDF) and upload the "
                "searchable version."
            )

        warnings: list[str] = []
        if empty_pages:
            shown = ", ".join(str(p) for p in empty_pages[:10])
            more = "..." if len(empty_pages) > 10 else ""
            warnings.append(
                f"{len(empty_pages)} page(s) had no extractable text and were skipped "
                f"(pages: {shown}{more})."
            )

        return ParsedDocument(
            doc_id=compute_doc_id(data),
            filename=safe_name,
            num_pages=doc.page_count,
            pages=pages,
            warnings=warnings,
        )
    finally:
        doc.close()
