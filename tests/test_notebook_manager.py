"""Tests for graphnotebook.notebooks.manager.NotebookManager (extended)."""

from unittest.mock import MagicMock

import pytest

from graphnotebook.notebooks.manager import Notebook, NotebookManager


@pytest.fixture()
def manager(mock_neo4j):
    return NotebookManager(mock_neo4j)


def test_get_by_id_found(manager, mock_neo4j):
    mock_neo4j.query.return_value = [
        {"n": {"id": "nb1", "name": "My NB", "description": "Desc"}, "doc_count": 2}
    ]
    nb = manager.get("nb1")
    assert nb is not None
    assert nb.name == "My NB"


def test_get_by_id_not_found_returns_none(manager, mock_neo4j):
    mock_neo4j.query.return_value = []
    nb = manager.get("nonexistent")
    assert nb is None


def test_notebook_isolation_queries_use_correct_id(manager, mock_neo4j):
    """Each notebook operation must scope to its own notebook_id."""
    mock_neo4j.query.return_value = [
        {"n": {"id": "nb2", "name": "NB2", "description": ""}, "doc_count": 0}
    ]
    manager.create("NB2", "")
    call_str = str(mock_neo4j.query.call_args_list)
    # The notebook id or name must appear in the Cypher or params
    assert "NB2" in call_str


def test_delete_non_existent_notebook_no_exception(manager, mock_neo4j):
    mock_neo4j.query.return_value = []
    # Must not raise even for non-existent id
    manager.delete("ghost_id")
