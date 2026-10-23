"""Command-line evaluation runner.

Examples (from the project root, with the virtual environment active)::

    # Offline, no downloads, no API key (lexical embedder, retrieval metrics only):
    python -m evaluation.run_evaluation --embedder hashing

    # Real semantic embeddings (downloads the model once):
    python -m evaluation.run_evaluation --embedder sentence-transformers

    # Also generate answers and LLM-judged metrics (needs LLM_API_KEY in .env):
    python -m evaluation.run_evaluation --embedder sentence-transformers --use-llm --judge
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import configure_logging, load_settings  # noqa: E402
from src.embeddings import Embedder, EmbeddingError, get_embedder  # noqa: E402
from src.evaluation import (  # noqa: E402
    EvalReport,
    LLMJudge,
    load_dataset,
    run_evaluation,
    save_report,
)
from src.generation import LLMConfigError, get_llm_client  # noqa: E402
from src.pipeline import ingest_pdf  # noqa: E402
from src.retrieval import Retriever  # noqa: E402
from src.vector_store import VectorStore  # noqa: E402

EVAL_DIR = PROJECT_ROOT / "evaluation"
DEFAULT_DATASET = EVAL_DIR / "sample_dataset.jsonl"
DEFAULT_DOCS = EVAL_DIR / "sample_docs"
DEFAULT_OUTPUT = EVAL_DIR / "results"


def build_eval_retriever(
    docs_dir: Path,
    persist_dir: Path,
    embedder: Embedder,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    min_relevance: float,
) -> Retriever:
    """Index every PDF in ``docs_dir`` into a dedicated evaluation store."""
    pdfs = sorted(docs_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(
            f"No PDFs found in {docs_dir}. Run 'python evaluation/make_sample_pdfs.py' first."
        )
    store = VectorStore(persist_dir, embedder.name, prefix="eval")
    for pdf in pdfs:
        report = ingest_pdf(pdf.read_bytes(), pdf.name, store, embedder, chunk_size, chunk_overlap)
        if report.status == "failed":
            raise RuntimeError(f"Could not index {pdf.name}: {report.message}")
    return Retriever(store, embedder, top_k=top_k, min_relevance=min_relevance)


def run(
    dataset_path: Path = DEFAULT_DATASET,
    docs_dir: Path = DEFAULT_DOCS,
    output_dir: Path = DEFAULT_OUTPUT,
    embedder_backend: str = "hashing",
    embedding_model: str = "",
    k: int = 4,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    min_relevance: float | None = None,
    use_llm: bool = False,
    judge: bool = False,
) -> EvalReport:
    """Run a complete evaluation and save the report. Used by the CLI and the dashboard."""
    settings = load_settings()
    embedder = get_embedder(embedder_backend, embedding_model or settings.embedding_model)

    # The hashing embedder's scores live on a different scale than neural embeddings.
    if min_relevance is None:
        min_relevance = 0.15 if embedder_backend == "hashing" else settings.min_relevance

    llm = get_llm_client(settings) if (use_llm or judge) else None
    judge_obj = LLMJudge(llm) if (judge and llm is not None) else None

    items = load_dataset(dataset_path)
    work_dir = Path(tempfile.mkdtemp(prefix="kp_eval_"))
    try:
        retriever = build_eval_retriever(
            docs_dir, work_dir, embedder, chunk_size, chunk_overlap, k, min_relevance
        )
        report = run_evaluation(
            items,
            retriever,
            k=k,
            llm=llm if use_llm else None,
            judge=judge_obj,
            price_input_per_m=settings.price_input_per_m,
            price_output_per_m=settings.price_output_per_m,
            config={
                "embedder": embedder.name,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "min_relevance": min_relevance,
                "llm_used": bool(use_llm),
                "llm_model": settings.llm_model if llm else None,
                "judge_used": judge_obj is not None,
                "dataset": str(dataset_path.name),
            },
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    save_report(report, output_dir)
    return report


def _fmt(value: object) -> str:
    if value is None:
        return "n/a (not computed)"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_summary(report: EvalReport) -> None:
    """Print a readable summary of measured results."""
    print("\n=== KnowledgePilot AI - evaluation summary ===")
    print("Config:", report.config)
    for section, values in report.summary.items():
        if isinstance(values, dict):
            print(f"\n[{section}]")
            for key, val in values.items():
                if key == "description":
                    print(f"  ({val})")
                else:
                    print(f"  {key}: {_fmt(val)}")
        elif isinstance(values, list):
            print(f"\n[{section}]")
            for line in values:
                print(f"  - {line}")
        else:
            print(f"{section}: {values}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the KnowledgePilot RAG pipeline.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--embedder", choices=["hashing", "sentence-transformers"], default="hashing")
    parser.add_argument("--embedding-model", default="")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--min-relevance", type=float, default=None)
    parser.add_argument("--use-llm", action="store_true", help="Generate answers with the LLM.")
    parser.add_argument("--judge", action="store_true", help="Add LLM-judged metrics.")
    args = parser.parse_args(argv)

    configure_logging()
    try:
        report = run(
            dataset_path=args.dataset,
            docs_dir=args.docs_dir,
            output_dir=args.output_dir,
            embedder_backend=args.embedder,
            embedding_model=args.embedding_model,
            k=args.k,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            min_relevance=args.min_relevance,
            use_llm=args.use_llm,
            judge=args.judge,
        )
    except (LLMConfigError, EmbeddingError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Evaluation could not run: {exc}", file=sys.stderr)
        return 1

    print_summary(report)
    print(f"\nSaved: {args.output_dir / 'latest.json'} and {args.output_dir / 'human_review.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
