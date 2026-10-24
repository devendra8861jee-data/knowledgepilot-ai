"""KnowledgePilot AI - Streamlit user interface.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

import pymupdf

import pandas as pd
import streamlit as st

from config import FALLBACK_MESSAGE, SUPPORTED_EMBEDDING_BACKENDS, configure_logging, load_settings
from src.citations import format_citation_label
from src.embeddings import EmbeddingError, get_embedder
from src.generation import LLMConfigError, get_llm_client
from src.models import AnswerResult
from src.pipeline import answer_question, ingest_pdf
from src.retrieval import Retriever
from src.vector_store import VectorStore, VectorStoreError

configure_logging()
logger = logging.getLogger("knowledgepilot.app")

st.set_page_config(page_title="KnowledgePilot AI", page_icon="📚", layout="wide", initial_sidebar_state="expanded")

# Dark navy theme inspired by the supplied RAG dashboard and document workspace.
st.markdown("""
<style>
:root { --kp-bg:#080f1e; --kp-panel:#0e192c; --kp-panel-2:#111f35; --kp-line:#263650; --kp-text:#e6edf8; --kp-muted:#95a7c2; --kp-blue:#4b7cff; --kp-cyan:#4bc7e8; --kp-green:#37c878; }
.stApp { background:var(--kp-bg); color:var(--kp-text); }
[data-testid="stHeader"] { background:rgba(8,15,30,.92); }
[data-testid="stSidebar"] { background:#070d19; border-right:1px solid #1d2a40; }
[data-testid="stSidebar"] * { color:#dbe5f6; }
[data-testid="stSidebar"] input { color:#101827 !important; }
[data-testid="stSidebar"] [data-baseweb="select"] * { color:#101827; }
.block-container { padding-top:1.35rem; padding-bottom:2rem; max-width:1800px; }
h1,h2,h3 { color:#f2f6ff !important; letter-spacing:-.025em; }
h1 { font-weight:760 !important; }
p,label,li { color:#d3deee; }
[data-testid="stCaptionContainer"], .stCaption { color:var(--kp-muted) !important; }
[data-testid="stMetric"] { background:linear-gradient(145deg,#111f35,#0e192b); border:1px solid var(--kp-line); border-radius:13px; padding:15px 16px; min-height:106px; box-shadow:0 8px 24px rgba(0,0,0,.14); }
[data-testid="stMetricLabel"] { color:#9db0ca !important; font-size:.84rem; }
[data-testid="stMetricValue"] { color:#f4f7ff !important; font-weight:720; }
[data-testid="stMetricDelta"] { font-size:.75rem; }
[data-testid="stVerticalBlockBorderWrapper"] { background:#0e192b; border-color:var(--kp-line) !important; border-radius:14px; }
[data-testid="stChatMessage"] { background:#101c30; border:1px solid #263650; border-radius:14px; padding:13px 15px; margin-bottom:.7rem; }
[data-testid="stFileUploader"] { background:#0e192b; border:1.5px dashed #3b5277; border-radius:14px; padding:12px; }
[data-testid="stExpander"] { background:#0e192b; border:1px solid #263650; border-radius:11px; }
.stButton > button,.stDownloadButton > button,[data-testid="stFormSubmitButton"] > button { border-radius:9px; border:1px solid #314563; font-weight:650; min-height:2.45rem; }
.stButton > button[kind="primary"],[data-testid="stFormSubmitButton"] > button[kind="primary"] { background:#315ff4; border-color:#4875ff; color:white; }
.stButton > button:hover { border-color:#6191ff; color:white; }
[data-baseweb="input"] input,[data-baseweb="textarea"] textarea { background:#101c30 !important; color:#edf3ff !important; }
[data-baseweb="select"] > div { background:#101c30; border-color:#314563; }
hr { border-color:#24334b; }
.kp-brand { display:flex; align-items:center; gap:.65rem; font-weight:800; font-size:1.05rem; padding:.25rem 0 .35rem; }
.kp-brand-mark { display:inline-flex; align-items:center; justify-content:center; width:2rem; height:2rem; border-radius:.65rem; background:#203d78; color:#89b4ff; font-size:1.15rem; }
.kp-eyebrow { color:#76a5ff; font-size:.73rem; text-transform:uppercase; letter-spacing:.14em; font-weight:800; }
.kp-hero { padding:1.15rem 1.3rem; border-radius:16px; border:1px solid #263c60; background:linear-gradient(115deg,#101e35,#0b1527 72%); margin:.35rem 0 1.1rem; }
.kp-muted { color:#96a9c4; font-size:.9rem; }
.kp-status-dot { color:#43d58a; font-weight:800; }
.kp-blank { border:1px dashed #354967; border-radius:15px; padding:1.25rem; color:#9eb0c8; background:#0c1729; }
@media (max-width:850px) { .block-container { padding:1rem .75rem 2rem; } h1 { font-size:1.8rem !important; } [data-testid="stMetric"] { padding:10px; min-height:84px; } }
@media (max-width:560px) { [data-testid="stSidebar"] { min-width:0; } [data-testid="stMetricValue"] { font-size:1.15rem; } }
</style>
""", unsafe_allow_html=True)

EVAL_RESULTS_PATH = Path(__file__).resolve().parent / "evaluation" / "results" / "latest.json"


# ------------------------------------------------------------------ cached objects
@st.cache_resource(show_spinner="Loading embedding model (first run may download it)...")
def load_embedder(backend: str, model: str):
    """Create the embedder once per backend/model."""
    return get_embedder(backend, model)


@st.cache_resource
def load_store(persist_dir: str, embedder_name: str) -> VectorStore:
    """Open the persistent vector store once per embedder."""
    return VectorStore(persist_dir, embedder_name)


# ---------------------------------------------------------------------- navigation
APP_PAGES = ["Overview", "Documents", "AI Chat", "Evaluation", "Settings & Help"]


def render_sidebar() -> str:
    """Render compact primary navigation and return the active page."""
    st.sidebar.markdown(
        '<div class="kp-brand"><span class="kp-brand-mark">✳</span><span>KnowledgePilot AI</span></div>',
        unsafe_allow_html=True,
    )
    st.sidebar.caption("PRIVATE PDF KNOWLEDGE WORKSPACE")
    st.sidebar.divider()
    page = st.sidebar.radio(
        "Workspace navigation", APP_PAGES, key="kp_active_page", label_visibility="collapsed"
    )
    st.sidebar.divider()
    base = apply_user_settings(load_settings()) if "apply_user_settings" in globals() else load_settings()
    st.sidebar.markdown("#### System status")
    st.sidebar.markdown('<span class="kp-status-dot">●</span> Local index available', unsafe_allow_html=True)
    if base.llm_provider != "none" and base.has_api_key:
        st.sidebar.caption(f"LLM configured · {base.llm_model}")
    elif base.llm_provider == "none":
        st.sidebar.caption("Extractive answer mode")
    else:
        st.sidebar.caption("LLM selected · API key missing")
    st.sidebar.divider()
    st.sidebar.caption("PDFs and the vector index are stored in your configured local data directory.")
    return page


def apply_user_settings(base):
    """Apply only values explicitly saved from the settings page."""
    overrides = st.session_state.get("kp_settings_overrides", {})
    try:
        return base.with_overrides(**overrides)
    except (TypeError, ValueError) as exc:
        logger.warning("Ignoring invalid session settings: %s", exc)
        return base


def render_settings_page(base) -> None:
    """Accessible grouped configuration with explicit Apply behavior."""
    st.markdown('<div class="kp-eyebrow">PREFERENCES</div>', unsafe_allow_html=True)
    st.title("Settings & Help")
    st.caption("Tune retrieval and answer generation. API keys stay in your private .env file.")
    settings = apply_user_settings(base)

    with st.form("kp_settings_form", border=True):
        st.subheader("1. Document understanding")
        c1, c2 = st.columns(2)
        with c1:
            backend_options = list(SUPPORTED_EMBEDDING_BACKENDS)
            backend = st.selectbox(
                "Embedding backend", backend_options,
                index=backend_options.index(settings.embedding_backend)
                if settings.embedding_backend in backend_options else 0,
                help="Sentence Transformers gives semantic search; hashing is a lightweight offline fallback.",
            )
            embedding_model = st.text_input("Embedding model", value=settings.embedding_model)
            chunk_size = st.number_input("Chunk size (characters)", 200, 2000, int(settings.chunk_size), 50)
        with c2:
            chunk_overlap = st.number_input("Chunk overlap (characters)", 0, 500, int(min(settings.chunk_overlap, 500)), 10)
            top_k = st.number_input("Top-k passages", 1, 10, int(settings.top_k), 1)
            threshold = st.slider(
                "Minimum relevance score", 0.0, 1.0, float(settings.min_relevance), 0.01,
                help="Increase to reject weak matches; decrease if relevant answers are being missed.",
            )
        st.divider()
        st.subheader("2. Answer generation")
        c3, c4 = st.columns(2)
        with c3:
            provider_options = ["none", "openai_compatible"]
            provider = st.selectbox(
                "LLM provider", provider_options,
                index=provider_options.index(settings.llm_provider)
                if settings.llm_provider in provider_options else 0,
                format_func=lambda value: "Extractive (no API)" if value == "none" else "OpenAI-compatible (Groq, etc.)",
            )
            model = st.text_input("LLM model", value=settings.llm_model)
        with c4:
            base_url = st.text_input("LLM base URL", value=settings.llm_base_url)
            st.caption("API key: " + ("Found in environment (.env); value hidden." if settings.has_api_key else "Not found. Add LLM_API_KEY to .env."))

        apply = st.form_submit_button("Apply settings", type="primary", use_container_width=True)
    if apply:
        if int(chunk_overlap) >= int(chunk_size):
            st.error("Chunk overlap must be smaller than chunk size. Update the values and try again.")
        elif not embedding_model.strip():
            st.error("Embedding model cannot be empty.")
        elif provider == "openai_compatible" and not base_url.strip():
            st.error("LLM base URL cannot be empty.")
        else:
            st.session_state["kp_settings_overrides"] = {
                "embedding_backend": backend,
                "embedding_model": embedding_model.strip(),
                "chunk_size": int(chunk_size),
                "chunk_overlap": int(chunk_overlap),
                "top_k": int(top_k),
                "min_relevance": float(threshold),
                "llm_provider": provider,
                "llm_model": model.strip() or settings.llm_model,
                "llm_base_url": base_url.strip() or settings.llm_base_url,
            }
            st.session_state["kp_settings_saved_notice"] = True
            st.rerun()

    if st.session_state.pop("kp_settings_saved_notice", False):
        st.success("Settings applied to this session. Chunking changes affect newly indexed PDFs; re-index existing PDFs when required.")
    with st.expander("Troubleshooting and safe defaults"):
        st.markdown("""
- **Scanned PDF:** the document needs OCR before its text can be searched.
- **No answer found:** check the selected document, question wording, relevance threshold, and top-k.
- **Embedding model download fails:** check internet access or switch to hashing for a lightweight test.
- **Groq/API errors:** verify the private .env values and provider limits. A key being present does not prove the connection is valid.
- **Changed chunk size or overlap:** re-upload/re-index documents to apply the new chunking.
- **Privacy:** never share or commit .env, API keys, or private PDFs.
""")

# -------------------------------------------------------------------- documents
def pdf_source_path(settings, doc_id: str) -> Path:
    """Local path for the original PDF; doc IDs are content hashes."""
    return Path(settings.data_dir) / "documents" / f"{doc_id}.pdf"


def persist_pdf_source(settings, doc_id: str, data: bytes) -> None:
    target = pdf_source_path(settings, doc_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_bytes(data)


def _delete_pdf_source(settings, doc_id: str) -> None:
    source = pdf_source_path(settings, doc_id)
    try:
        source.unlink(missing_ok=True)
    except OSError:
        logger.exception("Could not remove original PDF for document %s", doc_id)


@st.cache_data(show_spinner=False, max_entries=64)
def _render_pdf_page(path: str, page_number: int, zoom: float = 1.15) -> bytes:
    """Render a locally stored PDF page as an image for Streamlit's viewer pane."""
    pdf = pymupdf.open(path)
    try:
        if page_number < 1 or page_number > pdf.page_count:
            raise ValueError("Page number is outside the PDF.")
        page = pdf.load_page(page_number - 1)
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
        return pixmap.tobytes("png")
    finally:
        pdf.close()


def _jump_to_source(doc_id: str, page: int, doc_label: str) -> None:
    """Callback so clicking a citation focuses the corresponding document page."""
    st.session_state["kp_selected_doc"] = doc_label
    st.session_state[f"kp_preview_page_{doc_id}"] = int(page)


def render_pdf_preview(settings, doc: dict | None) -> None:
    """Show a page-image preview, with a clear message for older indexed PDFs."""
    if doc is None:
        st.markdown('<div class="kp-blank">📄<br><br><b>Select a document</b><br>Choose one PDF to preview pages and inspect citations.</div>', unsafe_allow_html=True)
        return

    source = pdf_source_path(settings, doc["doc_id"])
    st.markdown(f"**{doc['filename']}**")
    st.caption(f"{doc['pages']} searchable page(s) · {doc['chunks']} chunks")
    if not source.exists():
        st.info("This PDF was indexed before document preview was enabled. Re-upload it in Documents to enable page previews.")
        return

    try:
        with pymupdf.open(str(source)) as pdf:
            page_count = pdf.page_count
        page_key = f"kp_preview_page_{doc['doc_id']}"
        if page_key not in st.session_state:
            st.session_state[page_key] = 1
        current_page = int(st.session_state[page_key])
        if current_page > page_count:
            st.session_state[page_key] = page_count
        toolbar_left, toolbar_right = st.columns([1.15, 1])
        with toolbar_left:
            st.number_input(
                "PDF page", min_value=1, max_value=max(1, page_count), step=1,
                key=page_key, label_visibility="collapsed",
            )
        with toolbar_right:
            st.caption(f"of {page_count} page(s)")
        page_number = int(st.session_state[page_key])
        page_png = _render_pdf_page(str(source), page_number)
        st.image(page_png, width="stretch")
        st.caption(f"PDF page {page_number} · Physical page numbering")
    except (OSError, ValueError, RuntimeError) as exc:
        logger.exception("PDF preview failed for %s", doc["filename"])
        st.warning(f"Could not preview this PDF page. Try re-uploading the PDF. ({type(exc).__name__})")


def render_documents_tab(settings, store: VectorStore, embedder) -> None:
    st.markdown('<div class="kp-eyebrow">KNOWLEDGE BASE</div>', unsafe_allow_html=True)
    st.title("Documents")
    st.caption("Upload PDFs to build your private, searchable knowledge base.")
    docs = store.list_documents()
    a, b, c = st.columns(3)
    a.metric("Indexed PDFs", len(docs))
    b.metric("Searchable pages", sum(int(d["pages"]) for d in docs))
    c.metric("Indexed chunks", sum(int(d["chunks"]) for d in docs))

    with st.container(border=True):
        st.subheader("Add documents")
        st.caption("Text-based PDFs work best. Scanned/image-only PDFs need OCR before upload.")
        uploads = st.file_uploader(
            "Drop PDF files here or browse", type=["pdf"], accept_multiple_files=True,
            help="Multiple PDF files can be uploaded together.",
        )
        if st.button("Index uploaded PDFs", disabled=not uploads, type="primary", use_container_width=True):
            reports = []
            progress = st.progress(0.0, text="Preparing documents…")
            for number, upload in enumerate(uploads, start=1):
                progress.progress((number - 1) / len(uploads), text=f"Indexing {upload.name}…")
                payload = upload.getvalue()
                report = ingest_pdf(
                    payload, upload.name, store, embedder, settings.chunk_size, settings.chunk_overlap
                )
                if report.status in {"indexed", "duplicate"} and report.doc_id:
                    try:
                        persist_pdf_source(settings, report.doc_id, payload)
                    except OSError:
                        logger.exception("Could not persist original PDF %s", upload.name)
                        report.warnings.append("Indexed successfully, but local PDF preview could not be saved.")
                reports.append(report)
            progress.progress(1.0, text="Processing complete")
            for report in reports:
                if report.status == "indexed":
                    st.success(f"✓ {report.filename} — {report.message}")
                elif report.status == "duplicate":
                    st.info(f"↻ {report.filename} — {report.message}")
                else:
                    st.error(f"Could not index {report.filename}: {report.message}")
                for warning in report.warnings:
                    st.warning(f"{report.filename}: {warning}")
            st.rerun()

    st.subheader("Your indexed documents")
    docs = store.list_documents()
    if not docs:
        st.markdown(
            '<div class="kp-blank">📂<br><br><b>Your knowledge base is empty</b><br>'
            'Upload your first PDF above. Once indexing finishes, open AI Chat to ask questions.</div>',
            unsafe_allow_html=True,
        )
        return

    filter_text = st.text_input("Search documents", placeholder="Search by file name…", key="kp_doc_search")
    shown_docs = [d for d in docs if filter_text.strip().lower() in d["filename"].lower()]
    for doc in shown_docs:
        with st.container(border=True):
            left, pages, chunks, action = st.columns([5, 1, 1, 1.2], vertical_alignment="center")
            with left:
                st.markdown(f"**📄 {doc['filename']}**")
                st.caption(f"ID · {doc['doc_id'][:10]}…")
            pages.metric("Pages", doc["pages"])
            chunks.metric("Chunks", doc["chunks"])
            with action:
                if st.button("Delete", key=f"request_delete_{doc['doc_id']}"):
                    st.session_state["kp_pending_delete"] = doc["doc_id"]
                    st.rerun()
            if st.session_state.get("kp_pending_delete") == doc["doc_id"]:
                st.warning(f"Remove **{doc['filename']}** and its indexed passages? This cannot be undone.")
                confirm, cancel = st.columns(2)
                with confirm:
                    if st.button("Confirm delete", key=f"confirm_delete_{doc['doc_id']}", type="primary"):
                        store.delete_document(doc["doc_id"])
                        _delete_pdf_source(settings, doc["doc_id"])
                        st.session_state.pop("kp_pending_delete", None)
                        st.session_state.pop(f"kp_preview_page_{doc['doc_id']}", None)
                        if st.session_state.get("kp_selected_doc", "").endswith(doc["doc_id"][:6]):
                            st.session_state["kp_selected_doc"] = "All indexed PDFs"
                        st.success(f"Removed {doc['filename']}.")
                        st.rerun()
                with cancel:
                    if st.button("Cancel", key=f"cancel_delete_{doc['doc_id']}"):
                        st.session_state.pop("kp_pending_delete", None)
                        st.rerun()
    if not shown_docs:
        st.info("No documents match that search. Try a shorter filename fragment.")


# ------------------------------------------------------------------------- chat
def render_answer(result: AnswerResult, threshold: float) -> None:
    """Show one assistant answer with citations and the retrieved context."""
    if result.error:
        st.error(result.error)
        if result.retrieved:
            st.caption("The retrieved passages are still available below.")
    elif result.mode == "fallback":
        st.warning(result.answer or FALLBACK_MESSAGE)
    else:
        st.markdown(result.answer)
    for warning in result.warnings:
        st.warning(warning)

    if result.citations:
        st.markdown("**Sources**")
        for citation in result.citations:
            title = f"[{citation.index}] {format_citation_label(citation)} - score {citation.score:.2f}"
            with st.expander(title):
                st.write(citation.excerpt)

    if result.retrieved:
        with st.expander(f"Retrieved context (debug view, {len(result.retrieved)} passages)"):
            rows = [
                {
                    "rank": r.rank,
                    "file": r.chunk.filename,
                    "page": r.chunk.page,
                    "score": round(r.score, 4),
                    "used_as_evidence": r.score >= threshold,
                }
                for r in result.retrieved
            ]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            for r in result.retrieved:
                st.markdown(f"**#{r.rank} - {r.chunk.filename}, page {r.chunk.page} - score {r.score:.3f}**")
                st.text(r.chunk.text)

    parts = [f"mode: {result.mode}", f"retrieval: {result.retrieval_latency_s * 1000:.0f} ms"]
    if result.generation_latency_s:
        parts.append(f"generation: {result.generation_latency_s:.2f} s")
    if result.prompt_tokens is not None:
        parts.append(f"tokens: {result.prompt_tokens} in / {result.completion_tokens} out")
    st.caption(" | ".join(parts))


def render_chat_tab(settings, store: VectorStore, embedder) -> None:
    docs = store.list_documents()
    if not docs:
        st.info("Index at least one PDF in the Documents tab before asking questions.")
        return

    options = {f"{d['filename']} ({d['doc_id'][:6]})": d["doc_id"] for d in docs}
    chosen = st.multiselect("Search only in (leave empty for all documents)", list(options))
    doc_ids = [options[name] for name in chosen] or None

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if st.session_state.messages:
        export_lines = []
        for item in st.session_state.messages:
            if item["role"] == "user":
                export_lines.extend([f"## You\n\n{item['content']}", ""])
            else:
                export_lines.extend([f"## KnowledgePilot AI\n\n{item['result'].answer or item['result'].error or ''}", ""])
                for citation in item["result"].citations:
                    export_lines.append(f"- Source: {format_citation_label(citation)} (page {citation.page})")
                export_lines.append("")
        st.download_button(
            "⬇️ Export chat",
            "\n".join(export_lines),
            file_name="knowledgepilot-chat.md",
            mime="text/markdown",
            use_container_width=True,
        )

    if st.session_state.messages and st.button("🧹 Clear chat"):
        st.session_state.messages = []
        st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                render_answer(message["result"], message["threshold"])

    question = st.chat_input("Ask a question about your documents")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            llm = get_llm_client(settings)
        except LLMConfigError as exc:
            st.error(str(exc))
            return
        retriever = Retriever(store, embedder, settings.top_k, settings.min_relevance)
        with st.spinner("Searching your documents..."):
            result = answer_question(question, retriever, llm=llm, doc_ids=doc_ids)
        render_answer(result, settings.min_relevance)
    st.session_state.messages.append(
        {"role": "assistant", "result": result, "threshold": settings.min_relevance}
    )


# ------------------------------------------------------------------- evaluation
def _pct(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def render_report(report: dict) -> None:
    """Display a saved evaluation report. Only measured values are shown."""
    summary, config = report["summary"], report["config"]
    k = config.get("k", 4)
    st.caption(
        f"Embedder: {config.get('embedder')} | chunk size {config.get('chunk_size')} / overlap "
        f"{config.get('chunk_overlap')} | threshold {config.get('min_relevance')} | "
        f"{summary['n_questions']} questions ({summary['n_answerable']} answerable, "
        f"{summary['n_unanswerable']} unanswerable)"
    )

    st.markdown("#### Ground-truth retrieval metrics")
    gt = summary["ground_truth_retrieval_metrics"]
    cols = st.columns(4)
    cols[0].metric(f"Recall@{k}", _pct(gt.get(f"recall@{k}")))
    cols[1].metric(f"Hit rate@{k}", _pct(gt.get(f"hit_rate@{k}")))
    cols[2].metric(f"Precision@{k}", _pct(gt.get(f"precision@{k}")))
    cols[3].metric("MRR", _pct(gt.get("mrr")))
    st.caption(gt["description"])
    retrieval_chart = pd.DataFrame(
        [
            {"Metric": f"Recall@{k}", "Score": gt.get(f"recall@{k}")},
            {"Metric": f"Hit rate@{k}", "Score": gt.get(f"hit_rate@{k}")},
            {"Metric": f"Precision@{k}", "Score": gt.get(f"precision@{k}")},
            {"Metric": "MRR", "Score": gt.get("mrr")},
        ]
    ).dropna()
    if not retrieval_chart.empty:
        st.markdown("##### Retrieval quality at a glance")
        st.bar_chart(retrieval_chart, x="Metric", y="Score")

    st.markdown("#### Latency and reliability")
    ops_chart = pd.DataFrame(
        [
            {"Metric": "Mean latency (s)", "Value": ops_value}
            for ops_value in [summary["operational_metrics"].get("mean_latency_s")]
            if ops_value is not None
        ]
        + [
            {"Metric": "P95 latency (s)", "Value": ops_value}
            for ops_value in [summary["operational_metrics"].get("p95_latency_s")]
            if ops_value is not None
        ]
    )
    if not ops_chart.empty:
        st.bar_chart(ops_chart, x="Metric", y="Value")

    judged_for_chart = summary["llm_judged_metrics"]
    judged_chart = pd.DataFrame(
        [
            {"Metric": label, "Score": judged_for_chart.get(key)}
            for label, key in [
                ("Faithfulness", "faithfulness"),
                ("Answer relevance", "answer_relevance"),
                ("Context relevance", "context_relevance"),
            ]
            if judged_for_chart.get(key) is not None
        ]
    )
    if not judged_chart.empty:
        st.markdown("##### LLM judge scores")
        st.bar_chart(judged_chart, x="Metric", y="Score")
    else:
        st.info("Faithfulness and relevance charts will appear after an evaluation run with the LLM judge enabled.")

    st.markdown("#### Abstention behaviour and lexical proxy")
    ab, proxy = summary["abstention_behaviour"], summary["proxy_metrics"]
    cols = st.columns(3)
    cols[0].metric("Correct abstention (unanswerable)", _pct(ab["correct_abstention_rate_unanswerable"]))
    cols[1].metric("False abstention (answerable)", _pct(ab["false_abstention_rate_answerable"]))
    cols[2].metric("Keyword coverage in context", _pct(proxy["context_keyword_coverage"]))

    st.markdown("#### LLM-judged metrics")
    judged = summary["llm_judged_metrics"]
    cols = st.columns(3)
    cols[0].metric("Faithfulness", _pct(judged["faithfulness"]))
    cols[1].metric("Answer relevance", _pct(judged["answer_relevance"]))
    cols[2].metric("Context relevance", _pct(judged["context_relevance"]))
    st.caption(judged["description"])

    st.markdown("#### Operational metrics")
    ops = summary["operational_metrics"]
    cols = st.columns(4)
    cols[0].metric("Mean latency (s)", _pct(ops["mean_latency_s"]))
    cols[1].metric("P95 latency (s)", _pct(ops["p95_latency_s"]))
    cols[2].metric("Error rate", _pct(ops["error_rate"]))
    cols[3].metric("Estimated cost (USD)", _pct(ops["estimated_cost_usd"]))

    st.markdown("#### Needs human review")
    for line in summary["human_review_required"]:
        st.write(f"- {line}")

    st.markdown("#### Per-question results")
    rows = [
        {
            "id": i["id"],
            "answerable": i["answerable"],
            "abstained": i["abstained"],
            "recall": i["recall"],
            "precision": i["precision"],
            "rr": i["reciprocal_rank"],
            "top_hit": (
                f"{i['retrieved'][0]['filename']} p{i['retrieved'][0]['page']}" if i["retrieved"] else ""
            ),
            "top_score": i["retrieved"][0]["score"] if i["retrieved"] else None,
            "latency_s": round(i["total_latency_s"], 4),
            "question": i["question"],
        }
        for i in report["items"]
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def render_evaluation_tab(settings) -> None:
    from evaluation.run_evaluation import run  # imported lazily to keep start-up fast

    st.subheader("RAG evaluation")
    st.caption(
        "Runs the labelled dataset in evaluation/sample_dataset.jsonl against the sample PDFs in "
        "evaluation/sample_docs using a separate temporary index, so your own documents are not "
        "touched. Results below are only what was actually measured."
    )
    llm_ready = settings.llm_provider != "none" and settings.has_api_key
    use_llm = st.checkbox(
        "Generate answers with the LLM (uses API calls)", value=False, disabled=not llm_ready
    )
    judge = st.checkbox(
        "Add LLM-judged metrics (faithfulness, relevance; more API calls)",
        value=False,
        disabled=not llm_ready,
    )
    if not llm_ready:
        st.caption("LLM options are disabled until an LLM provider and API key are configured.")

    if st.button("Run evaluation", type="primary"):
        try:
            with st.spinner("Indexing sample documents and evaluating..."):
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
            st.success("Evaluation finished.")
        except (EmbeddingError, LLMConfigError, FileNotFoundError, ValueError, RuntimeError) as exc:
            st.error(f"Evaluation could not run: {exc}")

    from src.evaluation import load_report

    report = load_report(EVAL_RESULTS_PATH)
    if report is None:
        st.info("No evaluation results yet. Click 'Run evaluation'.")
    else:
        render_report(report)


# ------------------------------------------------------------------------- main
def main() -> None:
    st.markdown('<div class="kp-eyebrow">Private knowledge workspace</div>', unsafe_allow_html=True)
    st.title("KnowledgePilot AI")
    st.markdown(
        '<div class="kp-hero"><strong>Turn your PDFs into answers.</strong><br>'
        '<span class="kp-muted">Search your documents, ask questions in natural language, and verify every answer with source citations.</span></div>',
        unsafe_allow_html=True,
    )

    settings = render_sidebar()

    try:
        embedder = load_embedder(settings.embedding_backend, settings.embedding_model)
        store = load_store(str(settings.chroma_dir), embedder.name)
    except (EmbeddingError, VectorStoreError, ValueError) as exc:
        st.error(str(exc))
        st.stop()

    docs_now = store.list_documents()
    total_pages = sum(int(d.get("pages", 0) or 0) for d in docs_now)
    total_chunks = sum(int(d.get("chunks", 0) or 0) for d in docs_now)
    dash, tab_docs, tab_chat, tab_eval, tab_help = st.tabs(
        ["Overview", "📄 Documents", "💬 AI Chat", "📊 Evaluation", "❔ Help"]
    )
    with dash:
        st.subheader("Workspace overview")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Indexed PDFs", len(docs_now), help="Unique PDF files in your local index")
        m2.metric("Pages indexed", total_pages)
        m3.metric("Searchable chunks", total_chunks)
        m4.metric("Answer engine", "Groq / LLM" if settings.llm_provider != "none" and settings.has_api_key else "Extractive")
        left, right = st.columns([1.3, 1])
        with left:
            st.markdown("#### Recent documents")
            if docs_now:
                st.dataframe(
                    pd.DataFrame([
                        {"Document": d["filename"], "Pages": d["pages"], "Chunks": d["chunks"]}
                        for d in docs_now
                    ]),
                    hide_index=True,
                    width="stretch",
                )
            else:
                st.info("No PDFs indexed yet. Open Documents to upload your first file.")
        with right:
            st.markdown("#### System status")
            st.success(f"Embedding backend: {settings.embedding_backend}")
            if settings.llm_provider != "none" and settings.has_api_key:
                st.success(f"LLM configured: {settings.llm_model}")
            elif settings.llm_provider == "none":
                st.info("LLM generation is off; extractive answers are available.")
            else:
                st.warning("LLM selected, but API key is missing. Add LLM_API_KEY to .env.")
            st.caption("Your PDFs and vector index are stored locally according to DATA_DIR.")
        st.markdown("#### Get started")
        st.write("1. Upload PDFs in **Documents**.  2. Ask questions in **AI Chat**.  3. Check retrieval quality in **Evaluation**.")
    with tab_docs:
        render_documents_tab(settings, store, embedder)
    with tab_chat:
        render_chat_tab(settings, store, embedder)
    with tab_eval:
        render_evaluation_tab(settings)
    with tab_help:
        st.markdown(
            """
**How it works:** PDFs are split into overlapping chunks, embedded, and stored in ChromaDB.
A question is embedded, the most similar chunks are retrieved, and the answer is built only from
those chunks. Citations are looked up from the retrieved chunks, never typed by the model.

**Common problems**
- *Scanned PDF*: the file has no text layer. Run OCR first, then upload it again.
- *Embedding model download fails*: the first run needs internet. Use the `hashing` backend to test offline.
- *No answer found*: lower the minimum relevance score, raise top-k, or check the question wording.
- *API key errors*: put `LLM_API_KEY` in `.env` (see README) and restart the app.
- *Changed chunk size*: remove and re-upload the documents so they are indexed with the new settings.
            """
        )


main()
