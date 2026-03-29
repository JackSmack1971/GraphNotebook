import pytest
from graphnotebook.extraction.schema import build_default_schema, build_schema_from_json


def test_build_default_schema():
    schema = build_default_schema()
    assert schema is not None
    # Depending on neo4j-graphrag Version it could be a pydantic model or string representation
    schema_str = str(schema)
    assert "Person" in schema_str
    assert "Organization" in schema_str
    assert "WORKS_FOR" in schema_str


def test_build_schema_from_json():
    json_schema = {
        "entities": [
            {
                "label": "CustomEntity",
                "description": "A test entity",
                "properties": [{"name": "test_prop", "type": "STRING"}],
            }
        ],
        "relationships": [{"label": "CUSTOM_RELATION", "description": "A test rel"}],
    }
    schema = build_schema_from_json(json_schema)
    schema_str = str(schema)
    assert "CustomEntity" in schema_str
    assert "CUSTOM_RELATION" in schema_str
