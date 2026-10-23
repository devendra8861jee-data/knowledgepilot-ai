"""In-memory MCP protocol smoke test (runs when the MCP SDK is installed)."""

from __future__ import annotations

import asyncio

import pytest

mcp = pytest.importorskip("mcp", reason="Install requirements.txt to run MCP protocol tests")

from mcp import Client  # noqa: E402

from src.mcp_server.server import create_mcp_server  # noqa: E402
from src.mcp_server.tools import KnowledgeBaseTools  # noqa: E402


def test_mcp_server_exposes_tools_in_memory(store, embedder, pdf_cats):
    from src.pipeline import ingest_pdf

    report = ingest_pdf(pdf_cats, "cats-guide.pdf", store, embedder)
    assert report.status == "indexed"
    server = create_mcp_server(KnowledgeBaseTools(store, embedder))

    async def exercise():
        async with Client(server) as client:
            listed = await client.list_tools()
            names = {tool.name for tool in listed.tools}
            assert {"list_documents", "search_documents", "get_source_passage"} <= names
            result = await client.call_tool("list_documents", {})
            assert result.structured_content["ok"] is True

    asyncio.run(exercise())
