import json
from unittest.mock import MagicMock
from graphnotebook.notebooks.manager import NotebookManager


def test_json_export_structure():
    mock_neo4j = MagicMock()
    # Mocking entities, rels, and communities as flat lists of dicts
    mock_neo4j.query.side_effect = [
        [{"id": "E1", "type": "Person", "description": "desc"}],  # Entities
        [{"source": "E1", "type": "WORKS_FOR", "target": "E2"}],  # Relationships
        [{"id": "C1", "title": "Community 1"}],  # Communities
        [{"n": {"id": "N1", "name": "Notebook 1"}}],  # Get notebook fetch
    ]

    manager = NotebookManager(mock_neo4j)
    json_data = manager.export_json("N1")

    parsed = json.loads(json_data)
    assert "metadata" in parsed
    assert len(parsed["entities"]) == 1
    assert len(parsed["relationships"]) == 1
    assert parsed["metadata"]["notebook_name"] == "Notebook 1"


def test_markdown_export():
    mock_neo4j = MagicMock()
    mock_neo4j.query.side_effect = [
        [
            {
                "title": "C1",
                "level": 0,
                "entity_count": 5,
                "summary": "Full summary here",
            }
        ],  # Communities
        [{"n": {"id": "N1", "name": "Test Notebook"}}],  # Get notebook
    ]

    manager = NotebookManager(mock_neo4j)
    md = manager.export_markdown("N1")

    assert "# Knowledge Graph Report: Test Notebook" in md
    assert "### C1" in md
    assert "Full summary here" in md
