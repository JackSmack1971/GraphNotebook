import pytest
from unittest.mock import MagicMock
from graphnotebook.retrieval.text2cypher import Text2CypherRetriever

def test_native_availability_check_true():
    mock_neo4j = MagicMock()
    # Mock finding the procedure
    mock_neo4j.query.return_value = [{"name": "ai.text2cypher"}]
    
    retriever = Text2CypherRetriever(neo4j_client=mock_neo4j, llm_gateway=None)
    assert retriever._native_available is True

def test_native_availability_check_false():
    mock_neo4j = MagicMock()
    # Mock empty or failing
    mock_neo4j.query.return_value = []
    
    retriever = Text2CypherRetriever(neo4j_client=mock_neo4j, llm_gateway=None)
    assert retriever._native_available is False

def test_llm_query_cleans_fences():
    mock_neo4j = MagicMock()
    mock_neo4j.query.side_effect = [[], [{"result": "success"}]] # First check fails, second executes
    
    mock_llm = MagicMock()
    # LLM returns markdown fences
    mock_llm.invoke.return_value = "```cypher\nMATCH (n) RETURN n\n```"
    
    retriever = Text2CypherRetriever(neo4j_client=mock_neo4j, llm_gateway=mock_llm)
    
    res = retriever.query("Get everything")
    
    # Assert query was correctly stripped and executed
    mock_neo4j.query.assert_any_call("MATCH (n) RETURN n")
    assert res == [{"result": "success"}]
