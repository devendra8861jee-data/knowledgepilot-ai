import pytest

from src.ingestion import IngestionError, compute_doc_id, parse_pdf
from tests.conftest import make_pdf


def test_pages_text_and_numbers_are_preserved(pdf_cats):
    doc = parse_pdf(pdf_cats, "cats.pdf")
    assert doc.filename == "cats.pdf"
    assert doc.num_pages == 3
    assert [p.page_number for p in doc.pages] == [1, 2, 3]
    assert "fourteen hours" in doc.pages[1].text
    assert "Taurine" in doc.pages[2].text


def test_doc_id_depends_on_content_not_filename(pdf_cats, pdf_rockets):
    assert compute_doc_id(pdf_cats) == compute_doc_id(pdf_cats)
    assert compute_doc_id(pdf_cats) != compute_doc_id(pdf_rockets)
    assert parse_pdf(pdf_cats, "a.pdf").doc_id == parse_pdf(pdf_cats, "b.pdf").doc_id


def test_filename_path_is_stripped(pdf_cats):
    assert parse_pdf(pdf_cats, "C:/secret/folder/cats.pdf").filename == "cats.pdf"


def test_empty_bytes_rejected():
    with pytest.raises(IngestionError, match="empty"):
        parse_pdf(b"", "x.pdf")


def test_malformed_file_rejected():
    with pytest.raises(IngestionError, match="could not be opened|no pages|corrupted"):
        parse_pdf(b"this is definitely not a pdf", "fake.pdf")


def test_scanned_or_blank_pdf_gives_helpful_error():
    blank = make_pdf(["", "", ""])
    with pytest.raises(IngestionError, match="scanned"):
        parse_pdf(blank, "scan.pdf")


def test_partially_blank_pdf_warns_and_skips_pages():
    data = make_pdf(["A page with plenty of readable words in it.", "", "Another readable page here."])
    doc = parse_pdf(data, "mixed.pdf")
    assert [p.page_number for p in doc.pages] == [1, 3]
    assert doc.num_pages == 3
    assert any("page(s) had no extractable text" in w for w in doc.warnings)
