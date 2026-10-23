import json
from pathlib import Path

import pymupdf
import pytest

from evaluation.make_sample_pdfs import write_samples
from evaluation.run_evaluation import build_eval_retriever, run
from src.evaluation import (
    EvalItem,
    LLMJudge,
    estimate_cost,
    hit_at_k,
    keyword_coverage,
    load_dataset,
    parse_judge_score,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    run_evaluation,
    save_report,
)
from src.generation import LLMResponse

DATASET = Path(__file__).resolve().parent.parent / "evaluation" / "sample_dataset.jsonl"
A, B, C, D = ("a.pdf", 1), ("a.pdf", 2), ("b.pdf", 1), ("b.pdf", 2)


# ---------------------------------------------- metrics with hand-computed answers
def test_recall_precision_hit_mrr_hand_computed():
    retrieved = [C, A, D, B]  # relevant = {A, B}
    relevant = {A, B}
    assert recall_at_k(retrieved, relevant, 1) == 0.0
    assert recall_at_k(retrieved, relevant, 2) == 0.5
    assert recall_at_k(retrieved, relevant, 4) == 1.0
    assert precision_at_k(retrieved, relevant, 2) == 0.5
    assert precision_at_k(retrieved, relevant, 4) == 0.5
    assert hit_at_k(retrieved, relevant, 1) == 0.0
    assert hit_at_k(retrieved, relevant, 2) == 1.0
    assert reciprocal_rank(retrieved, relevant) == 0.5
    assert reciprocal_rank([C, D], relevant) == 0.0


def test_precision_counts_missing_results_as_misses():
    assert precision_at_k([A], {A}, 4) == 0.25


def test_recall_requires_labels():
    with pytest.raises(ValueError):
        recall_at_k([A], set(), 1)


def test_keyword_coverage_ignores_case_and_whitespace():
    assert keyword_coverage(["The limit is  INR\n1,200 per day"], ["inr 1,200", "missing"]) == 0.5
    assert keyword_coverage(["x"], []) is None


def test_cost_estimate_only_when_known():
    assert estimate_cost(1_000_000, 500_000, 1.0, 2.0) == pytest.approx(2.0)
    assert estimate_cost(None, 10, 1.0, 2.0) is None
    assert estimate_cost(10, 10, 0.0, 0.0) is None


def test_judge_score_parsing_is_robust():
    assert parse_judge_score('{"score": 0.8, "reason": "ok"}') == 0.8
    assert parse_judge_score('Sure! {"score": 7}') == 1.0  # clamped
    assert parse_judge_score("no json") is None
    assert parse_judge_score('{"score": "high"}') is None


class ScriptedLLM:
    def __init__(self, text):
        self.text = text

    def generate(self, messages):
        return LLMResponse(text=self.text, prompt_tokens=5, completion_tokens=2)


def test_judge_accumulates_tokens_and_returns_scores():
    judge = LLMJudge(ScriptedLLM('{"score": 0.5, "reason": "x"}'))
    assert judge.faithfulness("ans", ["ctx"]) == 0.5
    assert judge.answer_relevance("q", "ans") == 0.5
    assert judge.context_relevance("q", ["ctx"]) == 0.5
    assert judge.prompt_tokens == 15 and judge.completion_tokens == 6


# ------------------------------------------------------------------- dataset loading
def test_loader_rejects_bad_datasets(tmp_path):
    def write(lines):
        p = tmp_path / "d.jsonl"
        p.write_text("\n".join(lines), encoding="utf-8")
        return p

    with pytest.raises(ValueError, match="invalid JSON"):
        load_dataset(write(["{not json"]))
    with pytest.raises(ValueError, match="missing field"):
        load_dataset(write(['{"id": "1", "question": "q"}']))
    with pytest.raises(ValueError, match="needs at least one relevant"):
        load_dataset(write(['{"id": "1", "question": "q", "answerable": true, "relevant": []}']))
    dup = '{"id": "1", "question": "q", "answerable": false}'
    with pytest.raises(ValueError, match="duplicate"):
        load_dataset(write([dup, dup]))
    with pytest.raises(ValueError, match="empty"):
        load_dataset(write([""]))


def test_sample_dataset_labels_match_the_sample_pdfs(tmp_path):
    """Every expected keyword must really appear on the labelled page (labels are verified)."""
    pdf_dir = tmp_path / "pdfs"
    write_samples(pdf_dir)
    page_text = {}
    for pdf in pdf_dir.glob("*.pdf"):
        with pymupdf.open(pdf) as doc:
            page_text[pdf.name.lower()] = [" ".join(p.get_text().split()).lower() for p in doc]

    items = load_dataset(DATASET)
    assert len(items) >= 10
    assert any(not i.answerable for i in items)
    for item in items:
        for filename, page in item.relevant:
            for keyword in item.expected_keywords:
                assert keyword.lower() in page_text[filename][page - 1], (item.id, keyword)


def test_committed_sample_pdfs_exist():
    docs = Path(__file__).resolve().parent.parent / "evaluation" / "sample_docs"
    assert len(list(docs.glob("*.pdf"))) == 2


# ------------------------------------------------------------- end-to-end evaluation
def test_end_to_end_evaluation_offline(tmp_path):
    report = run(output_dir=tmp_path / "out", embedder_backend="hashing", k=4)
    gt = report.summary["ground_truth_retrieval_metrics"]
    assert 0.0 <= gt["recall@4"] <= 1.0 and 0.0 <= gt["mrr"] <= 1.0
    assert report.summary["n_questions"] == len(load_dataset(DATASET))
    assert report.summary["operational_metrics"]["error_rate"] == 0.0
    # LLM-based metrics must be reported as "not computed", never invented
    judged = report.summary["llm_judged_metrics"]
    assert judged["faithfulness"] is None and judged["answer_relevance"] is None
    assert report.summary["operational_metrics"]["estimated_cost_usd"] is None

    saved = json.loads((tmp_path / "out" / "latest.json").read_text(encoding="utf-8"))
    assert saved["summary"]["n_questions"] == report.summary["n_questions"]
    assert (tmp_path / "out" / "human_review.csv").exists()


def test_evaluation_with_fake_llm_and_judge(tmp_path, embedder):
    pdf_dir = tmp_path / "pdfs"
    write_samples(pdf_dir)
    retriever = build_eval_retriever(pdf_dir, tmp_path / "db", embedder, 800, 120, 4, 0.15)
    items = [
        EvalItem(
            id="x1",
            question="How many days of paid annual leave do full-time employees get?",
            expected_answer="24 days",
            relevant=frozenset({("acme_employee_handbook.pdf", 2)}),
            expected_keywords=("24 days",),
            answerable=True,
        )
    ]

    class JudgeAndAnswerLLM:
        def generate(self, messages):
            if "strict evaluator" in messages[0]["content"]:
                return LLMResponse('{"score": 1.0}', 10, 3)
            return LLMResponse("They receive 24 days [1].", 200, 10)

    llm = JudgeAndAnswerLLM()
    report = run_evaluation(
        items, retriever, k=4, llm=llm, judge=LLMJudge(llm),
        price_input_per_m=1.0, price_output_per_m=2.0,
    )
    row = report.items[0]
    assert row.mode == "llm" and row.recall == 1.0
    assert row.faithfulness == 1.0 and row.answer_relevance == 1.0 and row.context_relevance == 1.0
    assert row.cost_usd == pytest.approx((200 * 1.0 + 10 * 2.0) / 1_000_000)
    save_report(report, tmp_path / "r")
