"""FastAPI adapter for the React KnowledgePilot dashboard.

Run from the repository root:
    uvicorn api:app --reload --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import load_settings
from src.embeddings import get_embedder
from src.generation import LLMConfigError, get_llm_client
from src.pipeline import answer_question, ingest_pdf
from src.retrieval import Retriever
from src.vector_store import VectorStore

app = FastAPI(title="KnowledgePilot AI API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:4173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
_overrides: dict[str, Any] = {}


class SettingsUpdate(BaseModel):
    embedding_backend: str | None = None
    chunk_size: int | None = Field(default=None, ge=200, le=2000)
    chunk_overlap: int | None = Field(default=None, ge=0, le=500)
    top_k: int | None = Field(default=None, ge=1, le=10)
    min_relevance: float | None = Field(default=None, ge=0, le=1)
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_base_url: str | None = None


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=5000)
    doc_ids: list[str] | None = None


def settings():
    base = load_settings()
    valid = {k: v for k, v in _overrides.items() if v is not None}
    return base.with_overrides(**valid)


@lru_cache(maxsize=4)
def embedder_for(backend: str, model: str):
    return get_embedder(backend, model)


@lru_cache(maxsize=4)
def store_for(persist_dir: str, embedder_name: str):
    return VectorStore(persist_dir, embedder_name)


def runtime():
    cfg = settings()
    embedder = embedder_for(cfg.embedding_backend, cfg.embedding_model)
    store = store_for(str(cfg.chroma_dir), embedder.name)
    return cfg, embedder, store


def serialise_answer(result):
    return {
        "question": result.question,
        "answer": result.answer,
        "mode": result.mode,
        "sufficient_evidence": result.sufficient_evidence,
        "error": result.error,
        "warnings": result.warnings,
        "retrieval_latency_s": result.retrieval_latency_s,
        "generation_latency_s": result.generation_latency_s,
        "total_latency_s": result.total_latency_s,
        "citations": [
            {
                "index": c.index, "filename": c.filename, "page": c.page,
                "page_label": c.page_label, "excerpt": c.excerpt, "score": c.score,
            } for c in result.citations
        ],
        "retrieved": [
            {
                "rank": r.rank, "filename": r.chunk.filename, "page": r.chunk.page,
                "score": r.score, "text": r.chunk.text,
                "used_as_evidence": r.score >= settings().min_relevance,
            } for r in result.retrieved
        ],
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "app": "KnowledgePilot AI"}


@app.get("/api/config")
def get_config():
    cfg = settings()
    return {
        "embedding_backend": cfg.embedding_backend,
        "embedding_model": cfg.embedding_model,
        "chunk_size": cfg.chunk_size,
        "chunk_overlap": cfg.chunk_overlap,
        "top_k": cfg.top_k,
        "min_relevance": cfg.min_relevance,
        "llm_provider": cfg.llm_provider,
        "llm_model": cfg.llm_model,
        "llm_base_url": cfg.llm_base_url,
        "has_api_key": cfg.has_api_key,
    }


@app.put("/api/config")
def update_config(body: SettingsUpdate):
    global _overrides
    values = body.model_dump(exclude_unset=True, exclude_none=True)
    if "embedding_backend" in values and values["embedding_backend"] not in {"hashing", "sentence-transformers"}:
        raise HTTPException(400, "Unsupported embedding backend")
    if "llm_provider" in values and values["llm_provider"] not in {"none", "openai_compatible"}:
        raise HTTPException(400, "Unsupported LLM provider")
    _overrides.update(values)
    return get_config()


@app.get("/api/documents")
def list_documents():
    try:
        _, _, store = runtime()
        return {"documents": store.list_documents(), "total_chunks": store.count()}
    except Exception as exc:
        raise HTTPException(500, f"Could not load document index: {type(exc).__name__}: {exc}") from exc


@app.post("/api/documents")
async def upload_documents(files: list[UploadFile] = File(...)):
    cfg, embedder, store = runtime()
    reports = []
    for upload in files:
        name = Path(upload.filename or "upload.pdf").name
        if not name.lower().endswith(".pdf"):
            reports.append({"filename": name, "status": "failed", "message": "Only PDF files are supported."})
            continue
        data = await upload.read()
        if len(data) > 30 * 1024 * 1024:
            reports.append({"filename": name, "status": "failed", "message": "File exceeds the 30 MB limit."})
            continue
        report = ingest_pdf(data, name, store, embedder, cfg.chunk_size, cfg.chunk_overlap)
        reports.append({
            "filename": report.filename, "status": report.status, "message": report.message,
            "doc_id": report.doc_id, "pages": report.num_pages, "chunks": report.num_chunks,
            "warnings": report.warnings,
        })
    return {"reports": reports}


@app.delete("/api/documents/{doc_id}")
def remove_document(doc_id: str):
    _, _, store = runtime()
    if not any(d["doc_id"] == doc_id for d in store.list_documents()):
        raise HTTPException(404, "Document not found")
    store.delete_document(doc_id)
    return {"deleted": True, "doc_id": doc_id}


@app.post("/api/chat")
def chat(body: ChatRequest):
    cfg, embedder, store = runtime()
    if not store.list_documents():
        raise HTTPException(400, "Upload and index a PDF before asking questions.")
    try:
        llm = get_llm_client(cfg)
    except LLMConfigError as exc:
        raise HTTPException(400, str(exc)) from exc
    retriever = Retriever(store, embedder, cfg.top_k, cfg.min_relevance)
    result = answer_question(body.question, retriever, llm=llm, doc_ids=body.doc_ids or None)
    return serialise_answer(result)


@app.get("/api/evaluation")
def evaluation_report():
    path = Path(__file__).resolve().parent / "evaluation" / "results" / "latest.json"
    if not path.exists():
        return {"available": False, "message": "Run the evaluation from the existing Streamlit app to generate a report."}
    import json
    try:
        return {"available": True, "report": json.loads(path.read_text(encoding="utf-8"))}
    except (OSError, ValueError) as exc:
        raise HTTPException(500, "Could not read the evaluation report.") from exc
