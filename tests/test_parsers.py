"""Tests for graphnotebook.ingestion.parsers."""

import hashlib
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from graphnotebook.ingestion.parsers import (
    ParsedDocument,
    parse_document,
    parse_text,
)


def test_parse_text_basic(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("Hello world.\n\nSecond paragraph.", encoding="utf-8")
    doc = parse_text(str(f))
    assert doc.file_type == "txt"
    assert "Hello world" in doc.raw_text
    assert doc.raw_text_length == len(doc.raw_text)
    assert len(doc.pages) == 1


def test_parse_text_hash_determinism(tmp_path):
    f = tmp_path / "same.txt"
    content = b"consistent content"
    f.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()
    doc1 = parse_text(str(f))
    doc2 = parse_text(str(f))
    assert doc1.file_hash == expected
    assert doc2.file_hash == expected  # also validates determinism


def test_parse_text_md_extension(tmp_path):
    f = tmp_path / "readme.md"
    f.write_text("# Title\n\nBody text.", encoding="utf-8")
    doc = parse_document(str(f))
    assert doc.file_type == "md"


def test_parse_document_unsupported_extension(tmp_path):
    f = tmp_path / "file.xyz"
    f.write_text("data")
    with pytest.raises(ValueError, match="Unsupported file type"):
        parse_document(str(f))


@patch("graphnotebook.ingestion.parsers.fitz.open")
def test_parse_pdf_extracts_pages(mock_fitz_open, tmp_path):
    """Mock PyMuPDF to avoid needing a real PDF binary."""
    fake_page = MagicMock()
    fake_page.get_text.return_value = "Page content here."
    mock_doc = MagicMock()
    mock_doc.__iter__ = MagicMock(return_value=iter([fake_page]))
    mock_doc.__len__ = MagicMock(return_value=1)
    mock_fitz_open.return_value = mock_doc

    f = tmp_path / "test.pdf"
    f.write_bytes(b"%PDF-1.4 fake bytes")  # needs to exist for Path.read_bytes()

    from graphnotebook.ingestion.parsers import parse_pdf

    doc = parse_pdf(str(f))
    assert doc.file_type == "pdf"
    assert "Page content here." in doc.raw_text
    assert doc.metadata["page_count"] == 1


@patch("graphnotebook.ingestion.parsers.DocxDocument")
def test_parse_docx_extracts_paragraphs(mock_docx_cls, tmp_path):
    p1, p2 = MagicMock(), MagicMock()
    p1.text = "First paragraph."
    p2.text = "Second paragraph."
    mock_doc_instance = MagicMock()
    mock_doc_instance.paragraphs = [p1, p2]
    mock_docx_cls.return_value = mock_doc_instance

    f = tmp_path / "doc.docx"
    f.write_bytes(b"PK fake docx bytes")

    from graphnotebook.ingestion.parsers import parse_docx

    doc = parse_docx(str(f))
    assert "First paragraph." in doc.raw_text
    assert "Second paragraph." in doc.raw_text
    assert doc.file_type == "docx"


@patch("graphnotebook.ingestion.parsers.DocxDocument")
def test_parse_docx_skips_empty_paragraphs(mock_docx_cls, tmp_path):
    p1, p_empty, p2 = MagicMock(), MagicMock(), MagicMock()
    p1.text = "Real content."
    p_empty.text = "   "  # whitespace only — should be excluded
    p2.text = "More content."
    mock_doc_instance = MagicMock()
    mock_doc_instance.paragraphs = [p1, p_empty, p2]
    mock_docx_cls.return_value = mock_doc_instance

    f = tmp_path / "doc.docx"
    f.write_bytes(b"PK fake")

    from graphnotebook.ingestion.parsers import parse_docx

    doc = parse_docx(str(f))
    assert "   " not in doc.raw_text
