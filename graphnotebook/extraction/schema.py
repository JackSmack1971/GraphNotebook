"""
Domain ontology definition.
Defines allowed entity types and relationship types for KG extraction.
Can be overridden per-notebook via schema_json on the Notebook node.
"""

import hashlib
import json

from neo4j_graphrag.experimental.components.schema import (
    NodeType,
    PropertyType,
    RelationshipType,
    SchemaBuilder,
)


def get_schema_hash(schema_json: dict) -> str:
    """
    Compute a stable SHA256 hash of the schema JSON.
    Used for incremental ingestion to detect if re-extraction is needed.
    """
    if not schema_json:
        return "default"
    # Sort keys for stability
    schema_str = json.dumps(schema_json, sort_keys=True)
    return hashlib.sha256(schema_str.encode()).hexdigest()

# ── Default Entity Types ────────────────────────────

DEFAULT_ENTITIES = [
    NodeType(
        label="Person",
        description="A named individual person. Examples: 'John Smith', 'Albert Einstein'. Do not include generic titles like 'CEO' or 'President' unless specific names are mentioned.",  # noqa: E501
        properties=[
            PropertyType(name="role", type="STRING"),
            PropertyType(name="affiliation", type="STRING"),
        ],
    ),
    NodeType(
        label="Organization",
        description="Company, institution, government body, NGO, or team. Examples: 'Google', 'United Nations', 'FBI'. Do not include general groups like 'the public'.",  # noqa: E501
        properties=[
            PropertyType(name="industry", type="STRING"),
        ],
    ),
    NodeType(
        label="Concept",
        description="Abstract idea, theory, methodology, framework, or principle. Examples: 'Quantum Mechanics', 'Agile Methodology', 'Natural Selection'.",  # noqa: E501
        properties=[
            PropertyType(name="domain", type="STRING"),
        ],
    ),
    NodeType(
        label="Technology",
        description="Software, protocol, tool, framework, language, platform, or hardware. Examples: 'Python', 'TCP/IP', 'Linux', 'React'.",  # noqa: E501
        properties=[
            PropertyType(name="category", type="STRING"),
        ],
    ),
    NodeType(
        label="Location",
        description="Geographic place, region, country, city, or address. Examples: 'New York City', 'Europe', 'Mars'.",  # noqa: E501
    ),
    NodeType(
        label="Event",
        description="Named event, conference, incident, historical milestone. Examples: 'World War II', 'CES 2024', 'The Moon Landing'.",  # noqa: E501
        properties=[
            PropertyType(name="date", type="STRING"),
        ],
    ),
    NodeType(
        label="Metric",
        description="Quantitative measurement, KPI, statistic, or numerical goal. Examples: 'Revenue', 'Churn Rate', '100 million users'.",  # noqa: E501
        properties=[
            PropertyType(name="value", type="STRING"),
            PropertyType(name="unit", type="STRING"),
        ],
    ),
]

# ── Default Relationship Types ──────────────────────

DEFAULT_RELATIONS = [
    RelationshipType(
        label="WORKS_FOR",
        description="Person is employed by or clearly affiliated with Organization.",
    ),  # noqa: E501
    RelationshipType(
        label="FOUNDED", description="Person founded or co-founded Organization."
    ),  # noqa: E501
    RelationshipType(
        label="USES",
        description="An Entity uses, implements, or depends on a Technology or Concept.",
    ),  # noqa: E501
    RelationshipType(
        label="RELATED_TO",
        description="A general semantic connection between two Entities. Use this when no other relation fits.",
    ),  # noqa: E501
    RelationshipType(
        label="PART_OF",
        description="Component or subset relationship, such as a city in a country or a module in software.",
    ),  # noqa: E501
    RelationshipType(
        label="LOCATED_IN",
        description="Geographic containment. For example, a Person or Organization based in a Location.",
    ),  # noqa: E501
    RelationshipType(
        label="CAUSED_BY",
        description="Causal relationship where one Entity causes another Event or Concept.",
    ),  # noqa: E501
    RelationshipType(
        label="PRECEDED_BY",
        description="Temporal ordering where one Event explicitly occurs before another.",
    ),  # noqa: E501
    RelationshipType(
        label="MEASURED_BY",
        description="An Entity is measured by or defined quantitatively by a Metric.",
    ),  # noqa: E501
    RelationshipType(
        label="COMPETES_WITH",
        description="Competitive relationship between Organizations, Technologies, or People.",
    ),  # noqa: E501
    RelationshipType(
        label="COLLABORATES_WITH",
        description="Partnership, alliance, or collaboration between People or Organizations.",
    ),  # noqa: E501
    RelationshipType(
        label="INFLUENCES",
        description="Directional influence where one Entity significantly impacts another.",
    ),  # noqa: E501
]


def build_default_schema():
    """Build the default extraction schema."""
    builder = SchemaBuilder()
    # Ensure we use node_types instead of entities for SchemaBuilder Model validation
    return builder.create_schema_model(
        node_types=DEFAULT_ENTITIES,
        relationship_types=DEFAULT_RELATIONS,
    )


def build_schema_from_json(schema_json: dict):
    """
    Build a schema from per-notebook JSON override.
    Stored on (:Notebook {schema_json: '...'}).
    Falls back to defaults for missing fields.
    """
    builder = SchemaBuilder()
    entities = [
        NodeType(
            label=e["label"],
            description=e.get("description") or f"Entity of type {e['label']}",
            properties=[
                PropertyType(name=p["name"], type=p.get("type", "STRING"))
                for p in e.get("properties", [])
            ],
        )
        for e in schema_json.get("entities", [])
    ] or DEFAULT_ENTITIES

    relations = [
        RelationshipType(
            label=r["label"],
            description=r.get("description", ""),
        )
        for r in schema_json.get("relationships", [])
    ] or DEFAULT_RELATIONS

    return builder.create_schema_model(
        node_types=entities, relationship_types=relations
    )  # noqa: E501
