from src.pipeline import ingest_pdf
from src.retrieval import Retriever


def _index(store, embedder, pdf, name):
    return ingest_pdf(pdf, name, store, embedder, chunk_size=400, chunk_overlap=50)


def test_ingest_then_retrieve_correct_file_and_page(store, embedder, pdf_cats, pdf_rockets):
    assert _index(store, embedder, pdf_cats, "cats.pdf").status == "indexed"
    assert _index(store, embedder, pdf_rockets, "rockets.pdf").status == "indexed"

    retriever = Retriever(store, embedder, top_k=3, min_relevance=0.05)
    top = retriever.retrieve("How many hours do cats sleep each day?")[0]
    assert (top.chunk.filename, top.chunk.page) == ("cats.pdf", 2)

    top = retriever.retrieve("Which propellants burn in liquid rocket engines?")[0]
    assert (top.chunk.filename, top.chunk.page) == ("rockets.pdf", 2)


def test_metadata_is_not_mixed_between_documents(store, embedder, pdf_cats, pdf_rockets):
    _index(store, embedder, pdf_cats, "cats.pdf")
    _index(store, embedder, pdf_rockets, "rockets.pdf")
    retriever = Retriever(store, embedder, top_k=10, min_relevance=0.0)
    for item in retriever.retrieve("kerosene oxygen thrust"):
        if "kerosene" in item.chunk.text:
            assert item.chunk.filename == "rockets.pdf"
        if "Taurine" in item.chunk.text:
            assert item.chunk.filename == "cats.pdf"


def test_duplicate_upload_is_skipped_even_with_new_name(store, embedder, pdf_cats):
    first = _index(store, embedder, pdf_cats, "cats.pdf")
    count_after_first = store.count()
    second = _index(store, embedder, pdf_cats, "renamed_copy.pdf")
    assert first.status == "indexed"
    assert second.status == "duplicate"
    assert store.count() == count_after_first


def test_bad_files_fail_gracefully_without_exceptions(store, embedder):
    assert _index(store, embedder, b"", "empty.pdf").status == "failed"
    assert _index(store, embedder, b"not a pdf", "bad.pdf").status == "failed"
    assert store.count() == 0


def test_doc_filter_restricts_results(store, embedder, pdf_cats, pdf_rockets):
    rep_cats = _index(store, embedder, pdf_cats, "cats.pdf")
    _index(store, embedder, pdf_rockets, "rockets.pdf")
    retriever = Retriever(store, embedder, top_k=5, min_relevance=0.0)
    results = retriever.retrieve("rocket engines", doc_ids=[rep_cats.doc_id])
    assert results and all(r.chunk.filename == "cats.pdf" for r in results)


def test_threshold_filters_weak_matches(store, embedder, pdf_cats):
    _index(store, embedder, pdf_cats, "cats.pdf")
    retriever = Retriever(store, embedder, top_k=3, min_relevance=0.5)
    weak = retriever.retrieve("quantum chromodynamics lagrangian")
    assert retriever.select_evidence(weak) == []


def test_delete_document_and_listing(store, embedder, pdf_cats):
    report = _index(store, embedder, pdf_cats, "cats.pdf")
    listing = store.list_documents()
    assert listing[0]["filename"] == "cats.pdf" and listing[0]["pages"] == 3
    store.delete_document(report.doc_id)
    assert store.count() == 0
    assert not store.has_document(report.doc_id)


def test_empty_question_returns_nothing(store, embedder, pdf_cats):
    _index(store, embedder, pdf_cats, "cats.pdf")
    assert Retriever(store, embedder).retrieve("   ") == []


def test_technical_qualifier_guard_rejects_5ghz_for_5g_question():
    from src.models import Chunk, RetrievedChunk

    wifi = RetrievedChunk(
        chunk=Chunk(
            chunk_id="wifi",
            doc_id="doc",
            filename="router.pdf",
            page=1,
            chunk_index=0,
            text="The Orion X200 uses dual-band 2.4 GHz and 5 GHz Wi-Fi.",
        ),
        score=0.82,
        rank=1,
    )
    retriever = Retriever(store=None, embedder=None, min_relevance=0.15)  # type: ignore[arg-type]
    assert retriever.select_evidence([wifi], question="Does it support 5G cellular backup?") == []


def test_technical_qualifier_guard_keeps_exact_5g_evidence():
    from src.models import Chunk, RetrievedChunk

    cellular = RetrievedChunk(
        chunk=Chunk(
            chunk_id="cellular",
            doc_id="doc",
            filename="router.pdf",
            page=2,
            chunk_index=0,
            text="The router supports 5G cellular backup through an external modem.",
        ),
        score=0.82,
        rank=1,
    )
    retriever = Retriever(store=None, embedder=None, min_relevance=0.15)  # type: ignore[arg-type]
    assert retriever.select_evidence([cellular], question="Does it support 5G cellular backup?") == [cellular]


def test_hybrid_reranker_promotes_exact_lexical_match():
    from src.models import Chunk, RetrievedChunk

    class FakeEmbedder:
        def embed_query(self, question):
            return [1.0]

    class FakeStore:
        def query(self, query_embedding, top_k, doc_ids=None):
            semantic_only = RetrievedChunk(
                Chunk("a", "d", "semantic.pdf", 1, 0, "A device connects wirelessly to a mobile network."),
                score=0.80,
                rank=1,
            )
            exact = RetrievedChunk(
                Chunk("b", "d", "exact.pdf", 2, 0, "The Orion X200 has 4 Gigabit LAN ports."),
                score=0.78,
                rank=2,
            )
            return [semantic_only, exact][:top_k]

    retriever = Retriever(FakeStore(), FakeEmbedder(), top_k=2, lexical_weight=0.5)
    results = retriever.retrieve("How many LAN ports does the Orion X200 have?")
    assert results[0].chunk.filename == "exact.pdf"
    assert results[0].score == 0.78  # original semantic score remains available
    assert results[0].rank == 1


def test_answer_pipeline_abstains_when_only_5ghz_wifi_matches_5g_question():
    from config import FALLBACK_MESSAGE
    from src.models import Chunk, RetrievedChunk
    from src.pipeline import answer_question

    wifi = RetrievedChunk(
        chunk=Chunk(
            chunk_id="wifi",
            doc_id="doc",
            filename="router.pdf",
            page=1,
            chunk_index=0,
            text="The Orion X200 uses dual-band 2.4 GHz and 5 GHz Wi-Fi.",
        ),
        score=0.82,
        rank=1,
    )

    class FakeEmbedder:
        def embed_query(self, question):
            return [1.0]

    class FakeStore:
        def query(self, query_embedding, top_k, doc_ids=None):
            return [wifi]

    retriever = Retriever(FakeStore(), FakeEmbedder(), top_k=4, min_relevance=0.15)
    result = answer_question("Does the Orion X200 support 5G cellular backup?", retriever)
    assert result.mode == "fallback"
    assert result.answer == FALLBACK_MESSAGE
    assert result.sufficient_evidence is False
