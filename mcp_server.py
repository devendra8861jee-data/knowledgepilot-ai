"""Run the KnowledgePilot AI MCP server over stdio.

Run from the repository root after installing requirements:
    python mcp_server.py

Keep stdout reserved for the MCP protocol. Application logs are sent to stderr.
"""

from __future__ import annotations

import logging
import sys

from config import configure_logging, load_settings
from src.embeddings import get_embedder
from src.mcp_server.server import create_mcp_server
from src.mcp_server.tools import KnowledgeBaseTools
from src.vector_store import VectorStore


def main() -> None:
    # stdio transport uses stdout for JSON-RPC; configure logs to stderr explicitly.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )
    settings = load_settings()
    embedder = get_embedder(settings.embedding_backend, settings.embedding_model)
    store = VectorStore(settings.chroma_dir, embedder.name)
    tools = KnowledgeBaseTools(store=store, embedder=embedder)
    server = create_mcp_server(tools)
    server.run()


if __name__ == "__main__":
    main()
