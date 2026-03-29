"""
Manager for Notebooks, the highest level organizing hierarchy for documents and schemas.
"""

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from graphnotebook.graph import queries


@dataclass
class Notebook:
    id: str
    name: str
    description: str
    schema_json: Optional[str]
    schema_hash: str
    created_at: Any
    updated_at: Any
    doc_count: int = 0


class NotebookManager:
    """CRUD for notebooks and full graph exports."""

    def __init__(self, neo4j_client):
        self.neo4j = neo4j_client
        from graphnotebook.extraction.schema import get_schema_hash
        self.get_schema_hash = get_schema_hash

    def create(
        self, name: str, description: str = "", schema_json: str = None
    ) -> Notebook:
        """Create a new notebook namespace."""
        nb_id = str(uuid.uuid4())
        s_hash = self.get_schema_hash(json.loads(schema_json) if schema_json else None)
        result = self.neo4j.query(
            queries.CREATE_NOTEBOOK,
            params={
                "id": nb_id,
                "name": name,
                "description": description,
                "schema_json": schema_json,
                "schema_hash": s_hash,
            },
        )
        n = result[0]["n"]
        return Notebook(
            id=nb_id,
            name=n.get("name"),
            description=n.get("description"),
            schema_json=n.get("schema_json"),
            schema_hash=n.get("schema_hash", "default"),
            created_at=n.get("created_at"),
            updated_at=n.get("updated_at"),
        )

    def get(self, notebook_id: str) -> Optional[Notebook]:
        """Fetch notebook metadata."""
        result = self.neo4j.query(queries.GET_NOTEBOOK, params={"id": notebook_id})
        if not result:
            return None
        n = result[0]["n"]
        return Notebook(
            id=n.get("id"),
            name=n.get("name"),
            description=n.get("description"),
            schema_json=n.get("schema_json"),
            schema_hash=n.get("schema_hash", "default"),
            created_at=n.get("created_at"),
            updated_at=n.get("updated_at"),
        )

    def list_all(self) -> List[Notebook]:
        """List all notebooks with document counts."""
        result = self.neo4j.query(queries.LIST_NOTEBOOKS)
        notebooks = []
        for r in result:
            n = r["n"]
            notebooks.append(
                Notebook(
                    id=n.get("id"),
                    name=n.get("name", "Untitled"),
                    description=n.get("description", ""),
                    schema_json=n.get("schema_json"),
                    schema_hash=n.get("schema_hash", "default"),
                    created_at=n.get("created_at"),
                    updated_at=n.get("updated_at"),
                    doc_count=r.get("doc_count", 0),
                )
            )
        return notebooks

    def update(
        self,
        notebook_id: str,
        name: str = None,
        description: str = None,
        schema_json: str = None,
    ) -> Optional[Notebook]:
        """Update notebook details."""
        params = {"id": notebook_id}
        if name is not None:
            params["name"] = name
        if description is not None:
            params["description"] = description
        if schema_json is not None:
            params["schema_json"] = schema_json
            params["schema_hash"] = self.get_schema_hash(json.loads(schema_json) if schema_json else None)

        result = self.neo4j.query(queries.UPDATE_NOTEBOOK, params=params)
        if not result:
            return None
        n = result[0]["n"]
        return Notebook(
            id=notebook_id,
            name=n.get("name"),
            description=n.get("description"),
            schema_json=n.get("schema_json"),
            schema_hash=n.get("schema_hash", "default"),
            created_at=n.get("created_at"),
            updated_at=n.get("updated_at"),
        )

    def delete(self, notebook_id: str):
        """Cascade delete notebook and clean orphans."""
        self.neo4j.query(queries.DELETE_NOTEBOOK_CASCADE, params={"id": notebook_id})
        self.neo4j.query(queries.CLEANUP_ORPHANED_ENTITIES)

    def get_schema(self, notebook_id: str) -> Optional[Dict[str, Any]]:
        """Return parsed schema dict or None."""
        nb = self.get(notebook_id)
        if nb and nb.schema_json:
            try:
                return json.loads(nb.schema_json)
            except Exception:
                pass
        return None

    def set_schema(self, notebook_id: str, schema_json: str):
        """Stringified schema to Notebook state."""
        self.update(notebook_id, schema_json=schema_json)

    def get_documents(self, notebook_id: str) -> List[Dict[str, Any]]:
        """Fetch documents bound to Notebook."""
        return self.neo4j.query(
            queries.GET_NOTEBOOK_DOCUMENTS, params={"notebook_id": notebook_id}
        )

    def export_json(self, notebook_id: str) -> str:
        """Dump the full graph to a dictionary."""
        entities = self.neo4j.query(queries.EXPORT_ENTITIES_JSON, params={"notebook_id": notebook_id})
        rels = self.neo4j.query(queries.EXPORT_RELATIONSHIPS_JSON, params={"notebook_id": notebook_id})
        communities = self.neo4j.query(queries.EXPORT_COMMUNITIES_JSON, params={"notebook_id": notebook_id})

        nb = self.get(notebook_id)

        export_data = {
            "metadata": {
                "notebook_id": notebook_id,
                "notebook_name": nb.name if nb else "Unknown",
                "export_date": datetime.now().isoformat(),
            },
            "entities": entities,
            "relationships": rels,
            "communities": communities,
        }
        return json.dumps(export_data, indent=2)

    def export_markdown(self, notebook_id: str) -> str:
        """Export a structured markdown report of the graph's communities and schema."""
        communities = self.neo4j.query(queries.EXPORT_COMMUNITIES_JSON, params={"notebook_id": notebook_id})

        # Sort by level (highest logical grouping first) and entity count
        communities.sort(
            key=lambda x: (x.get("level", 0), x.get("entity_count", 0)), reverse=True
        )

        nb = self.get(notebook_id)
        md = f"# Knowledge Graph Report: {nb.name if nb else 'GraphNotebook'}\n\n"

        md += "## Communities\n\n"
        for c in communities:
            md += (
                f"### {c.get('title', 'Unknown Community')} "
                f"(Level {c.get('level', 0)})\n"
            )
            md += f"**Entities:** {c.get('entity_count', 0)}\n\n"

            summary = c.get("summary")
            if summary:
                md += f"{summary}\n\n"
            else:
                md += "*No summary available.*\n\n"

        return md
