# KnowledgePilot AI — Portfolio Demo Workflow

Use this checklist for a 3–5 minute screen recording or live internship interview demo. Use only PDFs you are allowed to share; the repository's fictional sample PDFs are suitable for a public demo.

## Before recording

- [ ] Install dependencies using the README setup instructions.
- [ ] Run ` .\.venv\Scripts\python.exe -m pytest -v ` and capture the latest result. The currently reported local result is **70 passed in 23.26s** (2026-10-23); re-run before recording.
- [ ] Start the app with ` .\.venv\Scripts\python.exe -m streamlit run app.py `.
- [ ] Open the local app at `http://localhost:8501`.
- [ ] Keep API keys and `.env` values hidden. For a no-key demo, use retrieval-only mode and/or the hashing embedder as appropriate.

## Suggested 4-minute walkthrough

### 1. Problem and goal (20–30 seconds)

> “KnowledgePilot AI is a retrieval-augmented generation assistant for PDFs. It is designed to answer from document evidence and show where that evidence came from.”

### 2. Upload and index a PDF (45–60 seconds)

- Upload a sample PDF in the Streamlit app.
- Show the indexing status and document/page/chunk information available in the UI.
- Explain that page-aware chunking preserves page metadata for citations.

### 3. Ask a question with a clear answer (45–60 seconds)

- Ask a question whose answer is explicitly stated in the uploaded PDF.
- Show the answer and the associated filename/page/excerpt citation.
- Briefly explain the retrieval path: embeddings → ChromaDB similarity search → evidence context → answer/citation validation.

### 4. Demonstrate safe abstention (30–45 seconds)

- Ask a question that is clearly unrelated to the uploaded document.
- Show the app's insufficient-evidence behavior if the configured threshold produces abstention.
- Do not claim abstention is perfect; describe it as a safeguard controlled by retrieval scores and configuration.

### 5. Show MCP tools and automated tests (60–75 seconds)

- Explain that an MCP client can discover and call three read-only tools: `list_documents`, `search_documents`, and `get_source_passage`.
- Run the focused MCP checks: ` .\.venv\Scripts\python.exe -m pytest tests\test_mcp_tools.py tests\test_mcp_protocol.py tests\test_mcp_integration.py -v `.
- Then show the complete suite command and its latest output.
- Be clear that an MCP stdio server is normally launched by a compatible host/client; a silent server terminal alone does not demonstrate a successful client session.

## Architecture narration (short version)

> “PDF text is extracted page by page, split into overlapping chunks, embedded, and stored in ChromaDB. At question time, the system retrieves the most similar chunks and applies a relevance threshold. If the evidence is insufficient, it can abstain. Otherwise, it builds a source-grounded answer and validates citation markers against the retrieved metadata. The MCP server separately exposes read-only document listing, search, and passage retrieval tools.”

## Do not overclaim

- A passing test suite is evidence of tested behavior, not proof that every answer is correct.
- Report the measured evaluation dataset, embedder, and configuration alongside retrieval metrics.
- Do not claim an external MCP host or AI agent is configured unless you have actually connected and demonstrated it.
- Do not put secrets, private documents, local vector databases, or `.env` in a public repository.

## Portfolio assets to capture

- [ ] App screenshot showing an answer with a page citation.
- [ ] Screenshot showing the indexed document list.
- [ ] Terminal screenshot showing the latest full test summary.
- [ ] Architecture diagram from [`architecture.md`](architecture.md).
- [ ] A short README description and repository link on your resume/LinkedIn.
