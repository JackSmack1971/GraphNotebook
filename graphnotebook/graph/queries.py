"""
Cypher query constants for application use.
Crucial constraints: No inline Cypher string manipulation in business logic.
"""

# Store document metadata
UPSERT_DOCUMENT = """
// UPSERT_DOCUMENT
MERGE (d:Document {id: $id})
ON CREATE SET
    d.notebook_id = $notebook_id,
    d.filename = $filename,
    d.file_type = $file_type,
    d.file_hash = $file_hash,
    d.raw_text_length = $raw_text_length,
    d.chunk_count = $chunk_count,
    d.ingested_at = datetime(),
    d.status = 'processed'
"""

# Store a semantic text chunk with embedding
UPSERT_CHUNK = """
// UPSERT_CHUNK
MATCH (d:Document {id: $doc_id})
MERGE (c:Chunk {id: $id})
ON CREATE SET
    c.text = $text,
    c.chunk_index = $chunk_index,
    c.start_char = $start_char,
    c.end_char = $end_char,
    c.token_count = $token_count,
    c.page_number = $page_number,
    c.section_header = $section_header
// Overwrite embeddings anytime we upsert
SET c.embedding = $embedding
MERGE (d)-[:HAS_CHUNK]->(c)
"""

# Basic Vector Search against chunks
VECTOR_SEARCH = """
// VECTOR_SEARCH
CALL db.index.vector.queryNodes('chunk_embeddings', $top_k, $query_embedding)
YIELD node AS chunk, score AS vec_score
MATCH (chunk)<-[:HAS_CHUNK]-(doc:Document)
RETURN chunk.text AS chunk_text,
       chunk.id AS chunk_id,
       vec_score,
       doc.filename AS source_file,
       chunk.chunk_index AS chunk_index,
       chunk.page_number AS page_number
ORDER BY vec_score DESC
LIMIT $top_k
"""

# Graph stats overview
GRAPH_STATS = """
// GRAPH_STATS
MATCH (d:Document)
WITH count(d) as document_count
OPTIONAL MATCH (c:Chunk)
WITH document_count, count(c) as chunk_count
OPTIONAL MATCH (e) WHERE labels(e)[0] IN [
    'Person', 'Organization', 'Concept', 'Technology', 'Location', 'Event', 'Metric'
]
WITH document_count, chunk_count, count(e) as entity_count
OPTIONAL MATCH ()-[r]-() WHERE type(r) IN [
    'WORKS_FOR', 'FOUNDED', 'USES', 'RELATED_TO', 'PART_OF', 'LOCATED_IN', 
    'CAUSED_BY', 'PRECEDED_BY', 'MEASURED_BY', 'COMPETES_WITH', 
    'COLLABORATES_WITH', 'INFLUENCES'
]
WITH document_count, chunk_count, entity_count, 
     count(r)/2 as relationship_count
RETURN document_count, chunk_count, entity_count, relationship_count
"""

# Check for Duplicate documents (now bounded to notebook)
CHECK_DOC_HASH = """
// CHECK_DOC_HASH
MATCH (d:Document {file_hash: $file_hash, notebook_id: $notebook_id})
RETURN COUNT(d) > 0 AS exists, collect(d.id)[0] AS doc_id
"""

# Phase 2: Entity Queries
ENTITY_SEARCH_BY_NAME = """
// ENTITY_SEARCH_BY_NAME
MATCH (e)
WHERE labels(e)[0] IN [
    'Person', 'Organization', 'Concept', 'Technology', 
    'Location', 'Event', 'Metric'
]
  AND toLower(e.id) CONTAINS toLower($name)
RETURN e.id AS name, labels(e)[0] AS type, 
       e.description AS description, e.mention_count AS mention_count
ORDER BY e.mention_count DESC
LIMIT $top_k
"""

GET_ENTITY_NEIGHBORHOOD = """
// GET_ENTITY_NEIGHBORHOOD
MATCH (e {id: $entity_id})-[r]-(neighbor)
WHERE labels(e)[0] IN [
    'Person', 'Organization', 'Concept', 'Technology', 
    'Location', 'Event', 'Metric'
]
  AND labels(neighbor)[0] IN [
    'Person', 'Organization', 'Concept', 'Technology', 
    'Location', 'Event', 'Metric'
]
RETURN e.id AS source, labels(e)[0] AS source_type,
       type(r) AS relationship, 
       neighbor.id AS target, labels(neighbor)[0] AS target_type
LIMIT $limit
"""

# Phase 4: Notebook & Polish Queries
CREATE_NOTEBOOK = """
// CREATE_NOTEBOOK
MERGE (n:Notebook {id: $id})
ON CREATE SET
    n.name = $name,
    n.description = $description,
    n.schema_json = $schema_json,
    n.created_at = datetime(),
    n.updated_at = datetime()
RETURN n
"""

UPDATE_NOTEBOOK = """
// UPDATE_NOTEBOOK
MATCH (n:Notebook {id: $id})
SET n.name = COALESCE($name, n.name),
    n.description = COALESCE($description, n.description),
    n.schema_json = COALESCE($schema_json, n.schema_json),
    n.updated_at = datetime()
RETURN n
"""

GET_NOTEBOOK = """
// GET_NOTEBOOK
MATCH (n:Notebook {id: $id})
RETURN n
"""

LIST_NOTEBOOKS = """
// LIST_NOTEBOOKS
MATCH (n:Notebook)
OPTIONAL MATCH (d:Document {notebook_id: n.id})
WITH n, count(d) AS doc_count
RETURN n, doc_count
ORDER BY n.updated_at DESC
"""

GET_NOTEBOOK_DOCUMENTS = """
// GET_NOTEBOOK_DOCUMENTS
MATCH (d:Document {notebook_id: $notebook_id})
RETURN d.id AS id, d.filename AS filename, d.file_type AS file_type,
       d.chunk_count AS chunk_count, d.ingested_at AS ingested_at,
       d.status AS status
ORDER BY d.ingested_at DESC
"""

DELETE_NOTEBOOK_CASCADE = """
// DELETE_NOTEBOOK_CASCADE
MATCH (n:Notebook {id: $id})
OPTIONAL MATCH (d:Document {notebook_id: n.id})
OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
DETACH DELETE c, d, n
"""

CLEANUP_ORPHANED_ENTITIES = """
// CLEANUP_ORPHANED_ENTITIES
MATCH (e)
WHERE size(labels(e)) > 0
  AND NOT e:Notebook AND NOT e:Document AND NOT e:Chunk AND NOT e:Community
  AND NOT EXISTS((e)<-[:MENTIONS]-(:Chunk))
DETACH DELETE e
"""

DELETE_DOC_CASCADE = """
// DELETE_DOC_CASCADE
MATCH (d:Document {id: $doc_id})
OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
DETACH DELETE c, d
"""

EXPORT_ENTITIES_JSON = """
// EXPORT_ENTITIES_JSON
MATCH (e)
WHERE size(labels(e)) > 0 
  AND NOT e:Notebook AND NOT e:Document AND NOT e:Chunk AND NOT e:Community
RETURN e.id AS id, labels(e)[0] AS type, 
       e.description AS description, e.mention_count AS mention_count
"""

EXPORT_RELATIONSHIPS_JSON = """
// EXPORT_RELATIONSHIPS_JSON
MATCH (source)-[r]->(target)
WHERE NOT source:Notebook AND NOT source:Document 
  AND NOT source:Chunk AND NOT source:Community
  AND NOT target:Notebook AND NOT target:Document 
  AND NOT target:Chunk AND NOT target:Community
  AND type(r) <> 'MENTIONS' AND type(r) <> 'HAS_CHUNK' AND type(r) <> 'BELONGS_TO'
RETURN source.id AS source, type(r) AS type, 
       target.id AS target, properties(r) AS properties
"""

EXPORT_COMMUNITIES_JSON = """
// EXPORT_COMMUNITIES_JSON
MATCH (c:Community)
RETURN c.id AS id, c.level AS level, c.title AS title, 
       c.summary AS summary, c.entity_count AS entity_count
"""
