import pytest
from pathlib import Path
from graphnotebook.ingestion.parsers import parse_text

def test_parse_text(tmp_path):
    # Create a temporary markdown file
    md_file = tmp_path / "sample.md"
    md_file.write_text("# Hello World\n\nThis is a test document.")
    
    parsed = parse_text(str(md_file))
    
    assert parsed.filename == "sample.md"
    assert parsed.file_type == "md"
    assert len(parsed.pages) == 1
    assert "Hello World" in parsed.raw_text
    assert parsed.raw_text_length > 0
    assert parsed.file_hash is not None
