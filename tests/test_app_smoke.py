"""Smoke tests that run the real Streamlit script headlessly (offline, no API key)."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

from src.embeddings import HashingEmbedder  # noqa: E402
from src.pipeline import ingest_pdf  # noqa: E402
from src.vector_store import VectorStore  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def offline_env(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("EMBEDDING_BACKEND", "hashing")
    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("LLM_API_KEY", "")
    return tmp_path


def test_app_starts_without_errors_and_shows_empty_state(offline_env):
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value == "KnowledgePilot AI"
    assert any("No documents indexed yet" in i.value for i in at.info)


def test_chat_returns_cited_passages_and_fallback(offline_env, pdf_cats):
    embedder = HashingEmbedder()
    store = VectorStore(offline_env / "chroma", embedder.name)
    assert ingest_pdf(pdf_cats, "cats.pdf", store, embedder, 400, 50).status == "indexed"

    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception

    at.chat_input[0].set_value("How many hours do cats sleep each day?").run()
    assert not at.exception
    rendered = " ".join(e.label for e in at.expander)
    assert "cats.pdf, page 2" in rendered

    at.chat_input[0].set_value("quantum chromodynamics lagrangian").run()
    assert not at.exception
    assert any("could not find sufficient evidence" in w.value for w in at.warning)
