import pytest
from unittest.mock import MagicMock
from graphnotebook.notebooks.manager import NotebookManager, Notebook

def test_notebook_crud():
    mock_neo4j = MagicMock()
    
    # Mock result for create
    mock_node = {"name": "Test NB", "description": "Desc", "id": "123"}
    mock_neo4j.query.return_value = [{"n": mock_node}]
    
    manager = NotebookManager(mock_neo4j)
    nb = manager.create("Test NB", "Desc")
    
    assert nb.name == "Test NB"
    assert mock_neo4j.query.call_count == 1
    
    # Mock result for list
    mock_neo4j.query.return_value = [{"n": mock_node, "doc_count": 5}]
    nbs = manager.list_all()
    assert len(nbs) == 1
    assert nbs[0].doc_count == 5

def test_notebook_delete_cascade():
    mock_neo4j = MagicMock()
    manager = NotebookManager(mock_neo4j)
    
    manager.delete("123")
    
    # Should call delete cascade and then orphan cleanup
    assert mock_neo4j.query.call_count == 2
    args_list = mock_neo4j.query.call_args_list
    assert "DELETE_NOTEBOOK_CASCADE" in str(args_list[0])
    assert "CLEANUP_ORPHANED_ENTITIES" in str(args_list[1])
