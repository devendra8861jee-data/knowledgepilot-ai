"""FastAPI adapter for the React KnowledgePilot UI.

Run locally with: uvicorn api:app --reload
This API is intended for local development; do not expose it publicly without auth.
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import load_settings
from src.embeddings import EmbeddingError, get_embedder
from src.generation import LLMConfigError, get_llm_client
from src.pipeline import answer_question, ingest_pdf
from src.retrieval import Retriever
from src.vector_store import VectorStore, VectorStoreError

logger = logging.getLogger("knowledgepilot.api")
BASE_SETTINGS = load_settings()
OVERRIDES: dict[str, object] = {}
RESOURCE_LOCK = RLock()
_RESOURCE_KEY: tuple[str, str, str] | None = None
_EMBEDDER = None
_STORE: VectorStore | None = None

app = FastAPI(title="KnowledgePilot AI API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:4173"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type"],
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    doc_ids: Optional[list[str]] = None


class SettingsPatch(BaseModel):
    embedding_backend: Optional[str] = None
    chunk_size: Optional[int] = Field(default=None, ge=200, le=2000)
    chunk_overlap: Optional[int] = Field(default=None, ge=0, le=500)
    top_k: Optional[int] = Field(default=None, ge=1, le=10)
    min_relevance: Optional[float] = Field(default=None, ge=0, le=1)
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = Field(default=None, max_length=200)
    llm_base_url: Optional[str] = Field(default=None, max_length=500)


def current_settings():
    return BASE_SETTINGS.with_overrides(**OVERRIDES)


def resources():
    global _RESOURCE_KEY, _EMBEDDER, _STORE
    settings = current_settings()
    key = (settings.embedding_backend, settings.embedding_model, str(settings.chroma_dir))
    with RESOURCE_LOCK:
        if key != _RESOURCE_KEY:
            embedder = get_embedder(settings.embedding_backend, settings.embedding_model)
            store = VectorStore(settings.chroma_dir, embedder.name)
            _EMBEDDER, _STORE, _RESOURCE_KEY = embedder, store, key
        return settings, _EMBEDDER, _STORE


def public_settings(settings) -> dict:
    return {
        "embedding_backend": settings.embedding_backend,
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "top_k": settings.top_k,
        "min_relevance": settings.min_relevance,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "llm_base_url": settings.llm_base_url,
        "llm_configured": settings.llm_provider == "none" or settings.has_api_key,
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "app": "KnowledgePilot AI"}


@app.get("/api/settings")
def get_settings():
    return public_settings(current_settings())


@app.patch("/api/settings")
def patch_settings(patch: SettingsPatch):
    global OVERRIDES
    changes = patch.model_dump(exclude_none=True)
    if "embedding_backend" in changes and changes["embedding_backend"] not in {"sentence-transformers", "hashing"}:
        raise HTTPException(status_code=422, detail="Unsupported embedding backend.")
    if "llm_provider" in changes and changes["llm_provider"] not in {"none", "openai_compatible"}:
        raise HTTPException(status_code=422, detail="Unsupported LLM provider.")
    effective = {**OVERRIDES, **changes}
    chunk_size = int(effective.get("chunk_size", BASE_SETTINGS.chunk_size))
    overlap = int(effective.get("chunk_overlap", BASE_SETTINGS.chunk_overlap))
    if overlap >= chunk_size:
        raise HTTPException(status_code=422, detail="Chunk overlap must be smaller than chunk size.")
    OVERRIDES = effective
    return public_settings(current_settings())


@app.get("/api/documents")
def list_documents():
    try:
        _, _, store = resources()
        return {"documents": store.list_documents(), "chunk_count": store.count()}
    except (EmbeddingError, VectorStoreError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/documents")
async def upload_documents(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="Choose at least one PDF.")
    settings, embedder, store = resources()
    reports = []
    for file in files:
        filename = Path(file.filename or "document.pdf").name
        if not filename.lower().endswith(".pdf"):
            reports.append({"filename": filename, "status": "failed", "message": "Only PDF files are supported."})
            continue
        payload = await file.read()
        report = ingest_pdf(payload, filename, store, embedder, settings.chunk_size, settings.chunk_overlap)
        reports.append(asdict(report))
    return {"reports": reports, "documents": store.list_documents(), "chunk_count": store.count()}


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    try:
        _, _, store = resources()
        if not any(item["doc_id"] == doc_id for item in store.list_documents()):
            raise HTTPException(status_code=404, detail="Document not found.")
        store.delete_document(doc_id)
        return {"deleted": doc_id, "documents": store.list_documents(), "chunk_count": store.count()}
    except (EmbeddingError, VectorStoreError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/chat")
def chat(request: ChatRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Enter a question.")
    try:
        settings, embedder, store = resources()
        llm = get_llm_client(settings)
        retriever = Retriever(store, embedder, settings.top_k, settings.min_relevance)
        result = answer_question(question, retriever, llm=llm, doc_ids=request.doc_ids or None)
        return asdict(result)
    except LLMConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (EmbeddingError, VectorStoreError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/evaluation")
def get_evaluation():
    from src.evaluation import load_report
    path = Path(__file__).resolve().parent / "evaluation" / "results" / "latest.json"
    report = load_report(path)
    return {"report": report}


@app.post("/api/evaluation")
def run_evaluation(use_llm: bool = False, judge: bool = False):
    from evaluation.run_evaluation import run
    from src.evaluation import load_report
    settings = current_settings()
    if (use_llm or judge) and (settings.llm_provider == "none" or not settings.has_api_key):
        raise HTTPException(status_code=400, detail="Configure an LLM provider and API key in the local .env first.")
    try:
        run(
            embedder_backend=settings.embedding_backend,
            embedding_model=settings.embedding_model,
            k=settings.top_k,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            min_relevance=settings.min_relevance,
            use_llm=use_llm or judge,
            judge=judge,
        )
        path = Path(__file__).resolve().parent / "evaluation" / "results" / "latest.json"
        return {"report": load_report(path)}
    except Exception as exc:
        logger.exception("Evaluation failed")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {type(exc).__name__}: {exc}") from exc
