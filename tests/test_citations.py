from config import FALLBACK_MESSAGE, Settings
from src.citations import (
    build_citations,
    extract_citation_indices,
    format_citation_label,
    make_excerpt,
    strip_invalid_markers,
)
from src.generation import (
    LLMConfigError,
    LLMError,
    LLMResponse,
    OpenAICompatibleClient,
    build_messages,
    get_llm_client,
)
from src.models import Chunk, RetrievedChunk
from src.pipeline import answer_question, ingest_pdf
from src.retrieval import Retriever

import pytest


def _retrieved(n: int = 2) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk=Chunk(
                chunk_id=f"doc{i}-p{i + 1}-c0",
                doc_id=f"doc{i}",
                filename=f"file{i}.pdf",
                page=i + 1,
                chunk_index=0,
                text=f"Original text of source number {i}. " * 20,
                page_label="",
            ),
            score=0.9 - i * 0.1,
            rank=i,
        )
        for i in range(1, n + 1)
    ]


class FakeLLM:
    def __init__(self, text: str = "", error: Exception | None = None):
        self.text, self.error, self.calls = text, error, 0

    def generate(self, messages):
        self.calls += 1
        if self.error:
            raise self.error
        return LLMResponse(text=self.text, prompt_tokens=100, completion_tokens=20)


# ----------------------------------------------------------------- pure functions
def test_extract_indices_handles_groups_and_duplicates():
    assert extract_citation_indices("A [1]. B [2, 3][1]. C [10]") == [1, 2, 3, 10]
    assert extract_citation_indices("no markers here") == []


def test_citations_come_only_from_retrieved_chunks():
    sources = _retrieved(2)
    citations, invalid = build_citations("Fact [1] and fact [2] and made-up [7]", sources)
    assert [c.filename for c in citations] == ["file1.pdf", "file2.pdf"]
    assert [c.page for c in citations] == [2, 3]
    assert invalid == [7]


def test_excerpt_is_a_true_substring_of_the_source_chunk():
    sources = _retrieved(1)
    citation = build_citations("Claim [1]", sources)[0][0]
    assert citation.excerpt.rstrip(".").strip() in sources[0].chunk.text
    assert len(make_excerpt("word " * 200, 100)) <= 104


def test_strip_invalid_markers_keeps_valid_numbers():
    assert strip_invalid_markers("Yes [1][9] and [2, 9]", [9]) == "Yes [1] and [2]"


def test_label_mentions_printed_label_only_when_different():
    sources = _retrieved(1)
    c = build_citations("x [1]", sources)[0][0]
    assert format_citation_label(c) == "file1.pdf, page 2"
    c2 = type(c)(**{**c.__dict__, "page_label": "iv"})
    assert "printed label: iv" in format_citation_label(c2)


def test_prompt_numbers_match_sources_and_warns_about_untrusted_text():
    messages = build_messages("Q?", _retrieved(2))
    assert "[1] (file: file1.pdf, page: 2)" in messages[1]["content"]
    assert "[2] (file: file2.pdf, page: 3)" in messages[1]["content"]
    assert "untrusted" in messages[0]["content"]


# -------------------------------------------------------------- pipeline behaviour
@pytest.fixture
def retriever(store, embedder, pdf_cats):
    ingest_pdf(pdf_cats, "cats.pdf", store, embedder, 400, 50)
    return Retriever(store, embedder, top_k=3, min_relevance=0.15)


def test_unsupported_question_triggers_fallback_and_skips_llm(retriever):
    llm = FakeLLM(text="Should never be used [1]")
    result = answer_question("quantum chromodynamics lagrangian", retriever, llm=llm)
    assert result.answer == FALLBACK_MESSAGE
    assert not result.sufficient_evidence and result.mode == "fallback"
    assert result.citations == [] and llm.calls == 0


def test_llm_answer_gets_real_filename_and_page_citations(retriever):
    llm = FakeLLM(text="Cats sleep about fourteen hours daily [1].")
    result = answer_question("How many hours do cats sleep each day?", retriever, llm=llm)
    assert result.mode == "llm" and result.sufficient_evidence
    assert [(c.filename, c.page) for c in result.citations] == [("cats.pdf", 2)]
    assert "fourteen hours" in result.citations[0].excerpt
    assert (result.prompt_tokens, result.completion_tokens) == (100, 20)


def test_fabricated_source_numbers_are_removed_and_flagged(retriever):
    llm = FakeLLM(text="Cats sleep a lot [1][42].")
    result = answer_question("How many hours do cats sleep each day?", retriever, llm=llm)
    assert "[42]" not in result.answer
    assert all(c.index != 42 for c in result.citations)
    assert any("do not exist" in w for w in result.warnings)


def test_answer_without_any_citation_is_flagged_unverified(retriever):
    llm = FakeLLM(text="Cats sleep a lot.")
    result = answer_question("How many hours do cats sleep each day?", retriever, llm=llm)
    assert result.citations == []
    assert any("no valid citation" in w for w in result.warnings)


def test_llm_saying_insufficient_evidence_yields_fallback_without_citations(retriever):
    llm = FakeLLM(text=FALLBACK_MESSAGE)
    result = answer_question("How many hours do cats sleep each day?", retriever, llm=llm)
    assert result.mode == "fallback" and not result.sufficient_evidence
    assert result.citations == []


@pytest.mark.parametrize("kind", ["auth", "rate_limit", "connection", "timeout"])
def test_llm_failures_are_graceful_and_keep_retrieved_passages(retriever, kind):
    llm = FakeLLM(error=LLMError(f"simulated {kind}", kind))
    result = answer_question("How many hours do cats sleep each day?", retriever, llm=llm)
    assert result.mode == "error" and "simulated" in result.error
    assert result.retrieved  # passages are still available for inspection


def test_no_llm_mode_returns_extractive_answer_with_real_citations(retriever):
    result = answer_question("How many hours do cats sleep each day?", retriever, llm=None)
    assert result.mode == "extractive"
    assert result.citations and result.citations[0].filename == "cats.pdf"


def test_blank_question_is_handled(retriever):
    assert answer_question("   ", retriever).mode == "fallback"


# ------------------------------------------------------------ configuration safety
def test_missing_api_key_raises_clear_error():
    settings = Settings(llm_provider="openai_compatible", llm_api_key="")
    with pytest.raises(LLMConfigError, match="LLM_API_KEY"):
        OpenAICompatibleClient(settings)
    with pytest.raises(LLMConfigError):
        get_llm_client(settings)


def test_provider_none_means_no_client():
    assert get_llm_client(Settings(llm_provider="none")) is None


def test_api_key_is_hidden_from_repr():
    secret = "sk-test-SECRET-123"
    assert secret not in repr(Settings(llm_api_key=secret))
