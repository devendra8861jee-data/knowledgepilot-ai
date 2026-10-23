import pytest

from src.chunking import chunk_document, validate_chunk_params
from src.models import PageText, ParsedDocument


def _doc(pages: dict[int, str], doc_id: str = "abc123", filename: str = "d.pdf") -> ParsedDocument:
    return ParsedDocument(
        doc_id=doc_id,
        filename=filename,
        num_pages=max(pages),
        pages=[PageText(page_number=n, text=t) for n, t in pages.items()],
    )


LONG_TEXT = " ".join(f"Sentence number {i} talks about topic {i % 7}." for i in range(80))


def test_chunks_keep_page_and_document_metadata():
    doc = _doc({2: LONG_TEXT, 5: "Short page with enough words to be kept."})
    chunks = chunk_document(doc, chunk_size=300, chunk_overlap=50)
    assert {c.page for c in chunks} == {2, 5}
    assert all(c.doc_id == "abc123" and c.filename == "d.pdf" for c in chunks)
    assert len([c for c in chunks if c.page == 5]) == 1


def test_chunk_size_is_respected_and_ids_unique():
    chunks = chunk_document(_doc({1: LONG_TEXT}), chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(len(c.text) <= 300 for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_overlap_creates_shared_text_between_neighbours():
    chunks = chunk_document(_doc({1: LONG_TEXT}), chunk_size=300, chunk_overlap=100)
    first_tail = chunks[0].text[-40:]
    assert any(word in chunks[1].text for word in first_tail.split()[-3:])


def test_chunks_never_span_pages():
    doc = _doc({1: "alpha " * 60, 2: "omega " * 60})
    for chunk in chunk_document(doc, chunk_size=200, chunk_overlap=20):
        assert not ("alpha" in chunk.text and "omega" in chunk.text)


@pytest.mark.parametrize(
    "size, overlap",
    [(50, 10), (300, -1), (300, 300), (300, 400)],
)
def test_invalid_parameters_rejected(size, overlap):
    with pytest.raises(ValueError):
        validate_chunk_params(size, overlap)
