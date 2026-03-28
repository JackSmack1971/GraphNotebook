"""
Reusable UI components for Gradio, including Graph visualizations and HITL tables.
"""
from pyvis.network import Network

from graphnotebook.graph.neo4j_client import Neo4jClient


def create_graph_visualization(
    neo4j_client: Neo4jClient, notebook_id: str = None, entity_filter: str = None
) -> str:
    """
    Generate an interactive PyVis network graph.
    Returns a self-contained HTML string.
    """
    # Fetch top 100 entities by mention count
    query = """
    MATCH (e)
    WHERE size(labels(e)) > 0 
      AND NOT e:Notebook AND NOT e:Document AND NOT e:Chunk AND NOT e:Community
    """
    if entity_filter:
        query += " AND toLower(e.id) CONTAINS toLower($filter)"
    
    query += """
    RETURN e.id AS id, labels(e)[0] AS type, 
           e.description AS description, e.mention_count AS mc
    ORDER BY e.mention_count DESC
    LIMIT 100
    """
    
    params = {"filter": entity_filter} if entity_filter else {}
    entities = neo4j_client.query(query, params)
    
    if not entities:
        return (
            "<div style='padding:20px; text-align:center;'>"
            "No entities found in this notebook yet.</div>"
        )

    # Fetch relationships between these entities
    entity_ids = [e["id"] for e in entities]
    rel_query = """
    MATCH (s)-[r]->(t)
    WHERE s.id IN $ids AND t.id IN $ids
      AND type(r) <> 'MENTIONS' AND type(r) <> 'HAS_CHUNK' AND type(r) <> 'BELONGS_TO'
    RETURN s.id AS source, type(r) AS type, t.id AS target
    """
    rels = neo4j_client.query(rel_query, {"ids": entity_ids})

    # Build PyVis network
    net = Network(
        height="600px",
        width="100%",
        notebook=True,
        cdn_resources='in_line',
        bgcolor="#ffffff",
        font_color="#333333"
    )
    
    # Color mapping
    colors = {
        "Person": "#3b82f6",       # Blue
        "Organization": "#10b981", # Green
        "Technology": "#a855f7",   # Purple
        "Concept": "#f59e0b",      # Orange
        "Location": "#ef4444",      # Red
        "Event": "#eab308",         # Yellow
        "Metric": "#6b7280"         # Gray
    }

    for e in entities:
        label = e["id"]
        title = f"Type: {e['type']}\nMentions: {e['mc']}\n\n{e['description'] or ''}"
        color = colors.get(e["type"], "#94a3b8")
        size = 10 + min(e["mc"] * 2, 40)
        net.add_node(e["id"], label=label, title=title, color=color, size=size)

    for r in rels:
        net.add_edge(r["source"], r["target"], label=r["type"], title=r["type"])

    # Physics settings for nice layout but stable performance
    net.set_options("""
    {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -2000,
          "centralGravity": 0.3,
          "springLength": 95
        },
        "minVelocity": 0.75,
        "stabilization": {
          "enabled": true,
          "iterations": 100
        }
      }
    }
    """)
    
    return net.generate_html()


def create_entity_review_table(entities: list) -> list:
    """
    Format extracted entities for a Gradio Dataframe.
    Each row: [Selected, Name, Type, Description]
    """
    rows = []
    for e in entities:
        rows.append({
            "Approve": True,
            "Name": e.get("name", "Unknown"),
            "Type": e.get("type", "Concept"),
            "Description": e.get("description", "")
        })
    return rows
