import pytest
from unittest.mock import MagicMock
from graphnotebook.graph.communities import CommunityManager
from graphnotebook.retrieval.local_search import LocalSearcher
from graphnotebook.retrieval.global_search import GlobalSearcher

def test_local_search_isolation():
    """Verify LocalSearcher scopes queries by notebook_id."""
    mock_neo4j = MagicMock()
    # Mock return list
    mock_neo4j.query.return_value = []
    
    searcher = LocalSearcher(mock_neo4j, notebook_id="nb_blue")
    searcher.search(query_embedding=[0.1]*1536)
    
    # Check that the query was called with the correct notebook_id in parameters
    assert mock_neo4j.query.called
    args, kwargs = mock_neo4j.query.call_args
    # call_args is (args, kwargs). parameters is the second positional argument (args[1]) or keyword 'parameters'
    params = args[1] if len(args) > 1 else kwargs.get("parameters", {})
    assert params["notebook_id"] == "nb_blue"

def test_global_search_isolation():
    """Verify GlobalSearcher scopes queries by notebook_id."""
    mock_neo4j = MagicMock()
    mock_neo4j.query.return_value = []
    
    community_manager = CommunityManager(mock_neo4j, notebook_id="nb_red")
    searcher = GlobalSearcher(mock_neo4j, community_manager=community_manager)
    searcher.search(query="test", query_embedding=[0.1]*1536, notebook_id="nb_red")
    
    # Check community manager query call (get_relevant_summaries is called inside)
    # The last call to query should be get_relevant_summaries
    assert mock_neo4j.query.called
    args, kwargs = mock_neo4j.query.call_args
    params = args[1] if len(args) > 1 else kwargs.get("parameters", {})
    assert params["notebook_id"] == "nb_red"

def test_community_detection_isolation():
    """Verify community detection uses scoped projection."""
    mock_neo4j = MagicMock()
    manager = CommunityManager(mock_neo4j, notebook_id="nb_green")
    
    # Mock success of GDS calls
    mock_neo4j.query.return_value = [{"graphName": "g"}]
    
    manager.detect_communities()
    
    # Verify that one of the calls contains nb_green (the projection query)
    calls = [str(call) for call in mock_neo4j.query.call_args_list]
    assert any("nb_green" in c for c in calls)

if __name__ == "__main__":
    pytest.main([__file__])
