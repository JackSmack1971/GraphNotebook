import pytest
from unittest.mock import MagicMock
from graphnotebook.graph.communities import CommunityManager

def test_community_manager_init():
    mock_neo4j = MagicMock()
    mock_llm = MagicMock()
    
    manager = CommunityManager(neo4j_client=mock_neo4j, llm_gateway=mock_llm)
    assert manager.neo4j == mock_neo4j
    assert manager.llm == mock_llm

def test_detect_communities():
    mock_neo4j = MagicMock()
    manager = CommunityManager(neo4j_client=mock_neo4j, llm_gateway=MagicMock())
    
    manager.detect_communities()
    
    # Check that drop, project, and leiden algorithms were called
    calls = [call.args[0] for call in mock_neo4j.query.call_args_list]
    assert any("gds.graph.drop" in c for c in calls)
    assert any("gds.graph.project" in c for c in calls)
    assert any("gds.leiden.write" in c for c in calls)
    # Check that it drops projection in finally (we get two drops basically via exists check and finally)
    drop_calls = sum(1 for c in calls if "gds.graph.drop('entity_graph'" in c)
    assert drop_calls >= 1

def test_get_summary_cached():
    mock_neo4j = MagicMock()
    mock_neo4j.query.return_value = [{"title": "Test Title", "summary": "Test Summary", "key_findings": [], "rank": 0}]
    
    manager = CommunityManager(neo4j_client=mock_neo4j, llm_gateway=MagicMock())
    
    summary = manager.get_summary("123")
    assert summary["title"] == "Test Title"
    
def test_get_summary_lazy_generate():
    mock_neo4j = MagicMock()
    # First query for cache returns empty
    # Second query for context returns mocked context
    mock_neo4j.query.side_effect = [
        [],  # cache miss
        [{"entities": [{"name": "E1", "type": "Concept", "desc": "Example"}], "relationships": []}], # context mock
        None  # cache set
    ]
    
    mock_llm = MagicMock()
    mock_llm.invoke_json.return_value = {"title": "Generated Title", "summary": "Gen Summary", "key_findings": [], "rank": 1}
    
    manager = CommunityManager(neo4j_client=mock_neo4j, llm_gateway=mock_llm)
    
    summary = manager.get_summary("123")
    assert summary["title"] == "Generated Title"
    assert mock_llm.invoke_json.called
