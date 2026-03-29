"""Tests for graphnotebook.notebooks.manager.NotebookManager.

Mutation targets addressed:
  M10b — StringMutator/BroadOracle: `assert "NB2" in call_str` was a substring
         match against the entire serialised call list, not a parameter check.
         Killed by test_notebook_isolation_queries_use_correct_id, which now
         inspects the bound parameter dictionaries directly.
"""

from unittest.mock import MagicMock

import pytest

from graphnotebook.notebooks.manager import Notebook, NotebookManager


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def manager(mock_neo4j):
    return NotebookManager(mock_neo4j)


# ---------------------------------------------------------------------------
# get()
# ---------------------------------------------------------------------------


def test_get_by_id_found(manager, mock_neo4j):
    mock_neo4j.query.return_value = [
        {"n": {"id": "nb1", "name": "My NB", "description": "Desc"}, "doc_count": 2}
    ]
    nb = manager.get("nb1")
    assert nb is not None
    assert nb.name == "My NB"
    assert nb.id == "nb1"


def test_get_by_id_not_found_returns_none(manager, mock_neo4j):
    mock_neo4j.query.return_value = []
    nb = manager.get("nonexistent")
    assert nb is None


# ---------------------------------------------------------------------------
# create()
# ---------------------------------------------------------------------------


def test_create_returns_notebook_with_correct_name(manager, mock_neo4j):
    mock_neo4j.query.return_value = []   # CREATE returns nothing meaningful
    nb = manager.create("Research", "My research notebook")
    assert nb.name == "Research"


# ---------------------------------------------------------------------------
# M10b — StringMutator kill: parameter binding verification
# ---------------------------------------------------------------------------


def test_notebook_isolation_queries_use_correct_id(manager, mock_neo4j):
    """Notebook operations must pass the notebook name as a bound Cypher
    parameter, not just embed it in the query string.

    Kills M10b: previous assertion was `assert "NB2" in call_str`, which is a
    substring match on the serialised call list. A mutation that passes the
    name under the wrong parameter key still satisfied that weak assertion.

    This test extracts the actual parameter dictionaries from every .query()
    call and asserts that "NB2" appears as a *value* in at least one of them.
    """
    mock_neo4j.query.return_value = []
    manager.create("NB2", "isolation test")

    all_calls = mock_neo4j.query.call_args_list
    assert all_calls, "NotebookManager.create must call neo4j.query at least once"

    # Collect all parameter dicts passed to neo4j.query(cypher, params)
    param_values: list[str] = []
    for call in all_calls:
        args = call.args      # positional args: (cypher,) or (cypher, params)
        kwargs = call.kwargs  # keyword args: may include parameters={}
        if len(args) > 1 and isinstance(args[1], dict):
            param_values.extend(str(v) for v in args[1].values())
        if "parameters" in kwargs and isinstance(kwargs["parameters"], dict):
            param_values.extend(str(v) for v in kwargs["parameters"].values())
        if "params" in kwargs and isinstance(kwargs["params"], dict):
            param_values.extend(str(v) for v in kwargs["params"].values())

    assert any("NB2" in v for v in param_values), (
        "Notebook name 'NB2' was not found in any bound parameter value. "
        "It must be passed as a Cypher parameter, not embedded in the query string. "
        f"Observed param values: {param_values}"
    )


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------


def test_delete_non_existent_notebook_no_exception(manager, mock_neo4j):
    """Deleting a non-existent notebook must not raise any exception."""
    mock_neo4j.query.return_value = []
    manager.delete("ghost_id")   # must complete without raising


def test_delete_calls_neo4j_query(manager, mock_neo4j):
    """delete() must issue at least one Cypher query to Neo4j."""
    mock_neo4j.query.return_value = []
    manager.delete("nb_to_delete")
    mock_neo4j.query.assert_called()


# ---------------------------------------------------------------------------
# rename()
# ---------------------------------------------------------------------------


def test_rename_passes_correct_id_and_name(manager, mock_neo4j):
    """rename() must pass both the notebook id and new name as parameters."""
    mock_neo4j.query.return_value = []
    manager.rename("nb_id_123", "New Name")

    all_calls = mock_neo4j.query.call_args_list
    all_param_values = []
    for call in all_calls:
        args = call.args
        kwargs = call.kwargs
        if len(args) > 1 and isinstance(args[1], dict):
            all_param_values.extend(str(v) for v in args[1].values())
        for kw in ("parameters", "params"):
            if kw in kwargs and isinstance(kwargs[kw], dict):
                all_param_values.extend(str(v) for v in kwargs[kw].values())

    assert any("nb_id_123" in v for v in all_param_values), (
        "Notebook id 'nb_id_123' not found in rename() query parameters"
    )
    assert any("New Name" in v for v in all_param_values), (
        "New name 'New Name' not found in rename() query parameters"
    )
