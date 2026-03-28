import pytest
from graphnotebook.extraction.resolver import EntityResolver
from unittest.mock import MagicMock

def test_entity_resolver_fuzzy_logic():
    mock_neo4j = MagicMock()
    # Mock labels call
    mock_neo4j.query.side_effect = [
        [{"label": "Person"}], # db.labels
        [
            {"id": "john_smith", "name": "John Smith", "mc": 10},
            {"id": "j_smith", "name": "J. Smith", "mc": 5},
            {"id": "jon_smith", "name": "Jon Smith", "mc": 2},
            {"id": "completely_different", "name": "Alice Bob", "mc": 1}
        ], # Match person
        None,
        None,
    ]
    
    resolver = EntityResolver(mock_neo4j, threshold=50.0)
    resolver.resolve_all()
    
    # We should have merged j_smith and jon_smith into john_smith
    # The merged query is called twice
    assert mock_neo4j.query.call_count == 4
    
    # Let's inspect the exact calls made for merges
    merge_calls = mock_neo4j.query.call_args_list[2:] 
    
    assert merge_calls[0][1]["parameters"]["keep_id"] == "john_smith"
    assert merge_calls[0][1]["parameters"]["merge_id"] in ["j_smith", "jon_smith"]
    
    assert merge_calls[1][1]["parameters"]["keep_id"] == "john_smith"
    assert merge_calls[1][1]["parameters"]["merge_id"] in ["j_smith", "jon_smith"]
