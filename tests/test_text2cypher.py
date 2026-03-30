import pytest
from unittest.mock import MagicMock, patch
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
    mock_neo4j.query.side_effect = [
        [],
        [{"result": "success"}],
    ]  # First check fails, second executes

    mock_llm = MagicMock()
    # LLM returns markdown fences
    mock_llm.invoke.return_value = "```cypher\nMATCH (n) RETURN n\n```"

    retriever = Text2CypherRetriever(neo4j_client=mock_neo4j, llm_gateway=mock_llm)

    res = retriever.query("Get everything")

    # Assert query was correctly stripped and executed
    mock_neo4j.query.assert_any_call("MATCH (n) RETURN n")
    assert res == [{"result": "success"}]
 
 
def test_query_returns_empty_on_no_results():
    mock_neo4j = MagicMock()
    mock_neo4j.query.side_effect = [[], []]  # Native check false, query returns []
 
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "MATCH (n) RETURN n"
 
    retriever = Text2CypherRetriever(neo4j_client=mock_neo4j, llm_gateway=mock_llm)
    res = retriever.query("Nothing here")
 
    assert res == []
 
 
def test_native_query_bypasses_llm():
    """If native text2cypher is available, it should be used via Neo4j directly."""
    mock_neo4j = MagicMock()
    # Mock result showing native procedure exists
    mock_neo4j.query.side_effect = [
        [{"name": "ai.text2cypher"}],    # 1. Availability check in __init__
        [{"cypher": "MATCH (n) RETURN n"}], # 2. CALL ai.text2cypher in _native_query
        [{"text": "native result"}],    # 3. Execution of returned cypher
    ]
 
    mock_llm = MagicMock()
    retriever = Text2CypherRetriever(neo4j_client=mock_neo4j, llm_gateway=mock_llm)
 
    res = retriever.query("native query")
    
    assert res == [{"text": "native result"}]
    mock_llm.invoke.assert_not_called()
    # SECOND call to query happens in _native_query to get the Cypher string
    mock_neo4j.query.assert_any_call(
        "CALL ai.text2cypher($query, {schema: $schema}) YIELD cypher RETURN cypher",
        params={"query": "native query", "schema": retriever._get_schema_description()}
    )
    # THIRD call is the execution of that Cypher
    mock_neo4j.query.assert_called_with("MATCH (n) RETURN n")
