"""
Context assembly with source attribution.
Builds the final context string for LLM synthesis.
"""

from typing import List, Optional

from .reranker import RetrievedChunk


class ContextBuilder:
    """Assemble retrieval results into structured context for LLM."""

    def build(
        self, chunks: Optional[List[RetrievedChunk]] = None, community_summaries: list = None
    ) -> str:  # noqa: E501
        """Build context string from chunks and community summaries."""
        parts = []

        if chunks:
            parts.append("## Relevant Document Passages\n")
            for i, chunk in enumerate(chunks):
                source = getattr(chunk, "source_file", "unknown")
                idx = getattr(chunk, "chunk_index", i)
                page = getattr(chunk, "page_number", "N/A")
                parts.append(
                    f"### [Source: {source}, Page {page}, Chunk {idx}]\n{chunk.text}\n"
                )

                # Add entity context if available
                entities = getattr(chunk, "entities", [])
                if entities:
                    entity_strs = [
                        f"  - {e.get('name', 'Unknown')} ({e.get('type', 'Unknown')})"
                        for e in entities
                        if isinstance(e, dict)
                    ]
                    if entity_strs:
                        parts.append("Entities explicitly mentioned in this passage:\n")
                        parts.extend(entity_strs)
                        parts.append("\n")

                # Add relationships if available
                rels = getattr(chunk, "relationships", [])
                if rels:
                    rel_strs = [
                        f"  - {r['source']} -[{r['rel']}]-> {r['target']}"
                        for r in rels
                        if isinstance(r, dict)
                    ]
                    if rel_strs:
                        parts.append("Known relationships:\n")
                        parts.extend(rel_strs)
                        parts.append("\n")

                parts.append("---\n")

        if community_summaries:
            parts.append("## Relevant Graph Communities\n")
            for s in community_summaries:
                parts.append(f"### {s['title']}\n{s['summary']}\n---\n")

        return "\n".join(parts)

    def extract_sources(self, chunks: List[RetrievedChunk]) -> list:
        """Extract unique sources from retrieved chunks."""
        sources = set()
        for chunk in chunks:
            source = getattr(chunk, "source_file", "unknown")
            page_number = getattr(chunk, "page_number", "N/A")
            sources.add(f"{source} (Page {page_number})")
        return list(sources)
