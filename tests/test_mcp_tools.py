"""Offline unit tests for the MCP-facing knowledge-base tools."""

from __future__ import annotations

from src.mcp_server.tools import KnowledgeBaseTools
from src.pipeline import ingest_pdf


def _tools(store, embedder, pdf_cats):
    report = ingest_pdf(
        pdf_cats,
        "cats-guide.pdf",
        store,
        embedder,
        chunk_size=800,
        chunk_overlap=80,
    )
    assert report.status == "indexed"
    return KnowledgeBaseTools(store, embedder), report.doc_id


def test_list_documents_returns_metadata(store, embedder, pdf_cats):
    tools, doc_id = _tools(store, embedder, pdf_cats)
    result = tools.list_documents()
    assert result["ok"] is True
    assert result["count"] == 1
    assert result["documents"][0]["doc_id"] == doc_id
    assert result["documents"][0]["filename"] == "cats-guide.pdf"
    assert result["documents"][0]["pages"] == 3


def test_search_returns_source_text_and_page(store, embedder, pdf_cats):
    tools, doc_id = _tools(store, embedder, pdf_cats)
    result = tools.search_documents("What do cats eat?", top_k=4, doc_ids=[doc_id])
    assert result["ok"] is True
    assert result["count"] > 0
    assert all("chunk_id" in row and "page" in row and "text" in row for row in result["results"])
    assert any("protein" in row["text"].lower() or "taurine" in row["text"].lower() for row in result["results"])


def test_search_rejects_bad_arguments(store, embedder):
    tools = KnowledgeBaseTools(store, embedder)
    assert tools.search_documents(" ")["error"]["code"] == "invalid_query"
    assert tools.search_documents("cats", top_k=11)["error"]["code"] == "invalid_top_k"
    assert tools.search_documents("cats", doc_ids=[])["error"]["code"] == "invalid_doc_ids"
    assert tools.search_documents("x" * 501)["error"]["code"] == "invalid_query"


def test_get_source_passage_returns_exact_chunk(store, embedder, pdf_cats):
    tools, doc_id = _tools(store, embedder, pdf_cats)
    results = tools.search_documents("cat diet taurine", doc_ids=[doc_id])
    chunk_id = results["results"][0]["chunk_id"]
    result = tools.get_source_passage(chunk_id)
    assert result["ok"] is True
    assert result["count"] == 1
    assert result["passages"][0]["is_requested_chunk"] is True
    assert result["passages"][0]["chunk_id"] == chunk_id


def test_get_source_passage_handles_unknown_id(store, embedder):
    tools = KnowledgeBaseTools(store, embedder)
    result = tools.get_source_passage("does-not-exist")
    assert result["ok"] is False
    assert result["error"]["code"] == "not_found"


def test_get_source_passage_validates_neighbor_count(store, embedder):
    tools = KnowledgeBaseTools(store, embedder)
    assert tools.get_source_passage("chunk-1", neighbors=3)["error"]["code"] == "invalid_neighbors"
