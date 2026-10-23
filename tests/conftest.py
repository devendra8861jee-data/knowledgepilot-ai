"""Shared test helpers. All tests run offline (no API key, no model download)."""

from __future__ import annotations

import sys
from pathlib import Path

import pymupdf
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.embeddings import HashingEmbedder  # noqa: E402
from src.vector_store import VectorStore  # noqa: E402


def make_pdf(pages: list[str]) -> bytes:
    """Build a small text PDF in memory. Empty strings create blank (image-less) pages."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(50, 50, 550, 780), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture
def store(tmp_path, embedder) -> VectorStore:
    return VectorStore(tmp_path / "chroma", embedder.name, prefix="test")


@pytest.fixture
def pdf_cats() -> bytes:
    return make_pdf(
        [
            "Cover page of the cats guide. Written for curious readers of all ages.",
            "Cats sleep about fourteen hours every day and are most active at dawn and dusk.",
            "A healthy cat diet needs animal protein. Taurine is an essential amino acid for cats.",
        ]
    )


@pytest.fixture
def pdf_rockets() -> bytes:
    return make_pdf(
        [
            "Rocket engineering handbook, internal edition, with a long enough cover text.",
            "Liquid rocket engines burn kerosene with liquid oxygen to produce enormous thrust.",
        ]
    )
