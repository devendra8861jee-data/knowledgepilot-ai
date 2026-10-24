"""KnowledgePilot AI - Streamlit user interface.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import logging
from pathlib import Path

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

# Lightweight responsive styling; the app remains native Streamlit.
st.markdown("""
<style>
  :root { --kp-ink: #172033; --kp-muted: #667085; --kp-line: #e6eaf0; --kp-accent: #4f46e5; }
  .stApp { background: linear-gradient(180deg, #f8faff 0%, #f7f8fc 240px, #ffffff 100%); color: var(--kp-ink); }
  [data-testid="stSidebar"] { background: #111827; border-right: 1px solid #263246; }
  [data-testid="stSidebar"] * { color: #f3f4f6; }
  [data-testid="stSidebar"] [data-baseweb="select"] * { color: #111827; }
  [data-testid="stSidebar"] input { color: #111827 !important; }
  h1 { letter-spacing: -0.035em; font-weight: 760 !important; }
  h2, h3 { letter-spacing: -0.02em; }
  [data-testid="stMetric"] { background: #fff; border: 1px solid var(--kp-line); border-radius: 16px; padding: 16px 18px; box-shadow: 0 4px 18px rgba(15,23,42,.035); }
  [data-testid="stMetricLabel"] { color: var(--kp-muted); }
  [data-testid="stMetricValue"] { color: var(--kp-ink); }
  [data-testid="stTabs"] button { border-radius: 10px 10px 0 0; font-weight: 650; }
  [data-testid="stChatMessage"] { border: 1px solid var(--kp-line); border-radius: 16px; padding: 14px; background: rgba(255,255,255,.86); }
  [data-testid="stFileUploader"] { border: 1.5px dashed #a5b4fc; border-radius: 16px; padding: 10px; background: #fafaff; }
  div.stButton > button[kind="primary"], div.stDownloadButton > button { border-radius: 10px; font-weight: 650; }
  [data-testid="stExpander"] { border: 1px solid var(--kp-line); border-radius: 12px; background: rgba(255,255,255,.78); }
  .kp-eyebrow { color: #4f46e5; font-size: .78rem; text-transform: uppercase; letter-spacing: .12em; font-weight: 800; }
  .kp-hero { padding: 1.15rem 1.3rem; border-radius: 18px; border: 1px solid #e0e7ff; background: linear-gradient(120deg,#eef2ff,#ffffff 70%); margin: .4rem 0 1.2rem 0; }
  .kp-muted { color: #667085; font-size: .92rem; }
  @media (max-width: 700px) {
    .block-container { padding: 1rem .8rem 2rem .8rem; }
    h1 { font-size: 1.8rem !important; }
    [data-testid="stMetric"] { padding: 11px; }
    [data-testid="stSidebar"] { min-width: 0; }
  }
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


# ---------------------------------------------------------------------- sidebar
def render_sidebar():
    """Show configuration widgets and return the effective settings."""
    base = load_settings()
    st.sidebar.markdown("## 📚 KnowledgePilot")
    st.sidebar.caption("Your private PDF research workspace")
    st.sidebar.divider()
    st.sidebar.header("⚙️ Settings")

    st.sidebar.subheader("Embeddings")
    backends = list(SUPPORTED_EMBEDDING_BACKENDS)
    backend = st.sidebar.selectbox(
        "Embedding backend",
        backends,
        index=backends.index(base.embedding_backend) if base.embedding_backend in backends else 0,
        help="'sentence-transformers' = real semantic search (downloads a model once). "
        "'hashing' = offline keyword-style demo with no downloads.",
    )

    st.sidebar.subheader("Chunking (applies when indexing)")
    chunk_size = st.sidebar.slider("Chunk size (characters)", 200, 2000, base.chunk_size, 50)
    max_overlap = max(0, chunk_size - 50)
    chunk_overlap = st.sidebar.slider(
        "Chunk overlap (characters)", 0, min(500, max_overlap), min(base.chunk_overlap, max_overlap), 10
    )

    st.sidebar.subheader("Retrieval")
    top_k = st.sidebar.slider("Top-k passages", 1, 10, base.top_k)
    default_threshold = 0.15 if backend == "hashing" else base.min_relevance
    min_relevance = st.sidebar.slider(
        "Minimum relevance score",
        0.0,
        1.0,
        float(default_threshold),
        0.01,
        help= "Passages below this cosine similarity are not treated as evidence. "
        "Sentence-transformer scores can be low; start around 0.10 and raise it if irrelevant "
        "passages are accepted. If nothing reaches it, the app shows the insufficient-evidence message.",
    )

    st.sidebar.subheader("Answer generation")
    provider = st.sidebar.selectbox(
        "LLM provider",
        ["none", "openai_compatible"],
        index=0 if base.llm_provider == "none" else 1,
        help="'none' shows retrieved passages only (free, offline).",
    )
    model = st.sidebar.text_input("LLM model", value=base.llm_model)
    base_url = st.sidebar.text_input("LLM base URL", value=base.llm_base_url)
    if provider == "openai_compatible":
        if base.has_api_key:
            st.sidebar.success("API key found in environment (value hidden).")
        else:
            st.sidebar.warning("No LLM_API_KEY found. Add it to your .env file and restart.")

    settings = base.with_overrides(
        embedding_backend=backend,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        min_relevance=min_relevance,
        llm_provider=provider,
        llm_model=model.strip() or base.llm_model,
        llm_base_url=base_url.strip() or base.llm_base_url,
    )
    return settings


# -------------------------------------------------------------------- documents
def render_documents_tab(settings, store: VectorStore, embedder) -> None:
    st.subheader("1. Upload and index PDFs")
    st.caption(
        "Text is extracted page by page. Page numbers shown in citations are the physical page "
        "positions in the PDF file (page 1 = first page, including any cover page). Printed page "
        "numbers in the document may differ."
    )
    uploads = st.file_uploader("Choose one or more PDF files", type=["pdf"], accept_multiple_files=True)

    if st.button("Index uploaded PDFs", disabled=not uploads, type="primary"):
        reports = []
        progress = st.progress(0.0, text="Starting...")
        for number, upload in enumerate(uploads, start=1):
            progress.progress((number - 1) / len(uploads), text=f"Indexing {upload.name}...")
            reports.append(
                ingest_pdf(
                    upload.getvalue(),
                    upload.name,
                    store,
                    embedder,
                    settings.chunk_size,
                    settings.chunk_overlap,
                )
            )
        progress.progress(1.0, text="Done")
        for rep in reports:
            if rep.status == "indexed":
                st.success(f"{rep.filename}: {rep.message}")
            elif rep.status == "duplicate":
                st.info(f"{rep.filename}: {rep.message}")
            else:
                st.error(f"{rep.filename}: {rep.message}")
            for warning in rep.warnings:
                st.warning(f"{rep.filename}: {warning}")

    st.subheader("2. Indexed documents")
    docs = store.list_documents()
    if not docs:
        st.info("No documents indexed yet. Upload a PDF above to get started.")
        return
    for doc in docs:
        cols = st.columns([5, 1, 1, 1])
        cols[0].write(f"**{doc['filename']}**  \n`id: {doc['doc_id']}`")
        cols[1].metric("Pages", doc["pages"])
        cols[2].metric("Chunks", doc["chunks"])
        if cols[3].button("Remove", key=f"del_{doc['doc_id']}"):
            store.delete_document(doc["doc_id"])
            st.rerun()


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
                export_lines.extend([f"## You\\n\\n{item['content']}", ""])
            else:
                export_lines.extend([f"## KnowledgePilot AI\\n\\n{item['result'].answer or item['result'].error or ''}", ""])
                for citation in item["result"].citations:
                    export_lines.append(f"- Source: {format_citation_label(citation)} (page {citation.page})")
                export_lines.append("")
        st.download_button(
            "⬇️ Export chat",
            "\\n".join(export_lines),
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
        st.bar_chart(retrieval_chart, x="Metric", y="Score", y_min=0, y_max=1)

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
        st.bar_chart(judged_chart, x="Metric", y="Score", y_min=0, y_max=1)
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
    st.title("KnowledgePilot AI")
    st.caption("Ask questions about your PDFs and get answers with verifiable page citations.")

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
