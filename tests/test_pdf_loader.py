"""Tests for the PDF ingestion loader."""
from __future__ import annotations

import io

import pytest
from fastapi import HTTPException
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from app.ingestion.pdf_loader import PDFLoader


@pytest.fixture
def loader() -> PDFLoader:
    return PDFLoader()


def _make_pdf(pages: list[str]) -> bytes:
    """Create a minimal PDF with the given list of text lines per page."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    for text in pages:
        if text:
            c.drawString(72, 750, text)
        c.showPage()
    c.save()
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_load_two_page_pdf(sample_pdf_bytes: bytes, loader: PDFLoader):
    docs = loader.load(sample_pdf_bytes, "test.pdf")
    assert len(docs) == 2


def test_page_numbers_are_1_based(sample_pdf_bytes: bytes, loader: PDFLoader):
    docs = loader.load(sample_pdf_bytes, "test.pdf")
    pages = [d.metadata["page"] for d in docs]
    assert pages[0] == 1
    assert pages[1] == 2


def test_source_is_filename(sample_pdf_bytes: bytes, loader: PDFLoader):
    docs = loader.load(sample_pdf_bytes, "report.pdf")
    assert all(d.source == "report.pdf" for d in docs)


def test_content_is_not_empty(sample_pdf_bytes: bytes, loader: PDFLoader):
    docs = loader.load(sample_pdf_bytes, "test.pdf")
    assert all(d.content.strip() for d in docs)


def test_page_text_is_extracted(sample_pdf_bytes: bytes, loader: PDFLoader):
    docs = loader.load(sample_pdf_bytes, "test.pdf")
    combined = " ".join(d.content for d in docs)
    assert "Acme Corp" in combined
    assert "Jane Smith" in combined


def test_blank_pages_are_skipped(loader: PDFLoader):
    # Page 1 has text, page 2 is blank
    pdf = _make_pdf(["First page content", ""])
    docs = loader.load(pdf, "mixed.pdf")
    assert len(docs) == 1
    assert docs[0].metadata["page"] == 1


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_malformed_pdf_raises_400(loader: PDFLoader):
    with pytest.raises(HTTPException) as exc_info:
        loader.load(b"this is not a pdf", "bad.pdf")
    assert exc_info.value.status_code == 400


def test_all_blank_pages_raises_400(loader: PDFLoader):
    pdf = _make_pdf(["", ""])  # Two blank pages
    with pytest.raises(HTTPException) as exc_info:
        loader.load(pdf, "blank.pdf")
    assert exc_info.value.status_code == 400
    assert "no extractable text" in exc_info.value.detail.lower()
