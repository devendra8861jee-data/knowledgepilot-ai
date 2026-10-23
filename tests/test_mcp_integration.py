"""End-to-end MCP integration test using a real stdio subprocess.

This test starts ``mcp_server.py`` as an MCP host would, discovers the tools,
then calls all three tools against an isolated temporary Chroma database. It
uses the hashing embedder so it needs no model download or API key.
"""

from __future__ import annotations

import asyncio
import gc
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp", reason="Install requirements.txt to run MCP integration tests")

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

from src.embeddings import HashingEmbedder  # noqa: E402
from src.pipeline import ingest_pdf  # noqa: E402
from src.vector_store import VectorStore  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_stdio_mcp_client_can_call_all_tools(tmp_path, pdf_cats):
    """Verify startup, discovery, document listing, search, and passage lookup."""
    data_dir = tmp_path / "data"
    embedder = HashingEmbedder()
    store = VectorStore(data_dir / "chroma", embedder.name)
    report = ingest_pdf(
        pdf_cats,
        "cats-guide.pdf",
        store,
        embedder,
        chunk_size=800,
        chunk_overlap=80,
    )
    assert report.status == "indexed"
    # Release the parent process database handles before the child opens the same DB.
    del store
    gc.collect()

    child_env = os.environ.copy()
    child_env.update(
        {
            "DATA_DIR": str(data_dir),
            "EMBEDDING_BACKEND": "hashing",
            "LLM_PROVIDER": "none",
            "PYTHONPATH": str(ROOT),
        }
    )
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(ROOT / "mcp_server.py")],
        cwd=str(ROOT),
        env=child_env,
    )

    async def exercise_server() -> None:
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                listed_tools = await session.list_tools()
                names = {tool.name for tool in listed_tools.tools}
                assert {
                    "list_documents",
                    "search_documents",
                    "get_source_passage",
                } <= names

                listed = await session.call_tool("list_documents", {})
                assert listed.is_error is False
                listing = listed.structured_content
                assert listing["ok"] is True
                assert listing["count"] == 1
                assert listing["documents"][0]["filename"] == "cats-guide.pdf"
                assert listing["documents"][0]["doc_id"] == report.doc_id

                searched = await session.call_tool(
                    "search_documents",
                    {"query": "What do cats eat? animal protein taurine", "top_k": 4},
                )
                assert searched.is_error is False
                search_result = searched.structured_content
                assert search_result["ok"] is True
                assert search_result["count"] > 0
                first_hit = search_result["results"][0]
                assert first_hit["chunk_id"]
                assert first_hit["filename"] == "cats-guide.pdf"
                assert isinstance(first_hit["page"], int)
                assert first_hit["text"]

                passage_result = await session.call_tool(
                    "get_source_passage", {"chunk_id": first_hit["chunk_id"]}
                )
                assert passage_result.is_error is False
                passage = passage_result.structured_content
                assert passage["ok"] is True
                assert passage["count"] == 1
                assert passage["passages"][0]["chunk_id"] == first_hit["chunk_id"]
                assert passage["passages"][0]["is_requested_chunk"] is True
                assert passage["passages"][0]["text"]

    asyncio.run(exercise_server())
