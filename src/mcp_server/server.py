"""Official MCP Python SDK v2 server registration."""

from __future__ import annotations

from mcp.server import MCPServer

from src.mcp_server.tools import KnowledgeBaseTools


def create_mcp_server(tools: KnowledgeBaseTools) -> MCPServer:
    """Register KnowledgePilot read-only tools on an MCPServer instance."""
    server = MCPServer("KnowledgePilot AI")

    @server.tool()
    def list_documents() -> dict[str, object]:
        """List the PDFs already indexed in KnowledgePilot, with page and chunk counts."""
        return tools.list_documents()

    @server.tool()
    def search_documents(
        query: str,
        top_k: int = 4,
        doc_ids: list[str] | None = None,
    ) -> dict[str, object]:
        """Search indexed PDFs and return relevant source chunks with page citations.

        query must be 1–500 characters; top_k must be 1–10. Optionally restrict
        results to document IDs returned by list_documents. This tool does not
        generate an answer; it returns source evidence only.
        """
        return tools.search_documents(query=query, top_k=top_k, doc_ids=doc_ids)

    @server.tool()
    def get_source_passage(chunk_id: str, neighbors: int = 0) -> dict[str, object]:
        """Retrieve an exact indexed passage by chunk ID, optionally with up to two adjacent chunks."""
        return tools.get_source_passage(chunk_id=chunk_id, neighbors=neighbors)

    return server
