"""
Event handlers for Gradio UI.
Connects UI inputs to orchestration layers.
Includes streaming chat, HITL review, and Notebook management.
"""

import json
from typing import Any, Generator, List

from graphnotebook.ingestion.pipeline import ingestion_pipeline
from graphnotebook.notebooks.manager import NotebookManager
from graphnotebook.ui.components import (
    create_graph_visualization,
)


async def handle_upload(
    file_paths: List[str],
    notebook_id: str,
    neo4j_client,
    embedding_engine,
    config,
    llm_gateway,
) -> str:
    """Process uploaded files through the ingestion pipeline."""
    if not file_paths:
        return "No files uploaded."
    if not notebook_id:
        return "Please select or create a notebook first."

    status_lines = []

    for path in file_paths:
        try:
            filename = path.replace("\\", "/").split("/")[-1]
            result = await ingestion_pipeline.ainvoke(
                {
                    "file_path": path,
                    "notebook_id": notebook_id,
                    "neo4j_client": neo4j_client,
                    "embedding_engine": embedding_engine,
                    "config": config,
                    "llm_gateway": llm_gateway,
                    "status": "started",
                    "entity_count": 0,
                    "kg_built": False,
                }
            )

            if result.get("error"):
                status_lines.append(f"❌ {filename}: {result['error']}")
            else:
                ec = result.get("entity_count", 0)
                status_lines.append(
                    f"✅ {filename}: Ingested successfully ({ec} entities)."
                )
        except Exception as e:
            status_lines.append(f"❌ {filename}: Exception: {str(e)}")

    return "\n".join(status_lines)


def handle_query_stream(
    message: str,
    history: List[dict],
    notebook_id: str,
    search_mode: str,
    query_agent,
    embedding_engine,
    llm_gateway,
) -> Generator[Any, None, None]:
    """
    Streaming chat generator for Gradio.
    Yields intermediate status and then the final synthesis response.
    """
    if not message.strip():
        yield history + [{"role": "assistant", "content": ""}]
        return

    # 1. Embed Query (Status update)
    new_history = history + [{"role": "user", "content": message}]
    yield new_history + [
        {"role": "assistant", "content": "🔍 *Classifying query and embedding...*"}
    ]

    query_emb = embedding_engine.embed_single(message)

    # 2. Agentic Retrieval & Routing (Status updates)
    if search_mode == "vector-only":
        search_mode = "local"

    yield new_history + [
        {"role": "assistant", "content": "⛓️ *Navigating knowledge graph...*"}
    ]

    # State needs conversation_history for the agent to use
    state = query_agent.invoke(
        {
            "query": message,
            "query_embedding": query_emb,
            "search_mode": search_mode,
            "conversation_history": history,
            "iterations": 0,
            "stream": True,  # Flag we set in router.py
        }
    )

    # 3. Yield Streaming Synthesis
    full_prompt = state.get("full_synthesis_prompt")
    system = state.get("system_prompt")

    if not full_prompt:
        yield new_history + [
            {
                "role": "assistant",
                "content": "Error: Failed to generate synthesis prompt.",
            }
        ]
        return

    # Start the streaming yield
    bot_message = ""
    for token in llm_gateway.invoke_stream(prompt=full_prompt, system=system):
        bot_message += token
        yield new_history + [{"role": "assistant", "content": bot_message}]

    # Append source citations at the end
    sources_data = state.get("sources", [])
    if sources_data:
        bot_message += "\n\n---\n**Sources:**\n"
        for src in sources_data:
            t = src.get("type", "source")
            n = src.get("source", src.get("title", "Unknown"))
            bot_message += f"- {t}: {n}\n"
        yield new_history + [{"role": "assistant", "content": bot_message}]


def create_notebook_callback(name: str, desc: str, nb_manager: NotebookManager):
    """Create a new notebook and return updated dropdown choices."""
    if not name:
        return None, "Notebook name required."
    nb_manager.create(name, desc)
    all_nbs = nb_manager.list_all()
    # Gradio dropdown choices: list of (label, value)
    choices = [(f"{n.name} ({n.doc_count} docs)", n.id) for n in all_nbs]
    return choices, f"Created notebook '{name}'"


def delete_notebook_callback(notebook_id: str, nb_manager: NotebookManager):
    """Delete notebook and return updated choices."""
    if not notebook_id:
        return None, "No notebook selected."
    nb_manager.delete(notebook_id)
    all_nbs = nb_manager.list_all()
    choices = [(f"{n.name} ({n.doc_count} docs)", n.id) for n in all_nbs]
    return choices, "Deleted notebook. Graph cleaned."


def refresh_notebooks(nb_manager: NotebookManager):
    all_nbs = nb_manager.list_all()
    return [(f"{n.name} ({n.doc_count} docs)", n.id) for n in all_nbs]


def save_schema_callback(
    notebook_id: str, schema_json: str, nb_manager: NotebookManager
):
    if not notebook_id:
        return "Select a notebook first."
    try:
        # Validate JSON
        json.loads(schema_json)
        nb_manager.set_schema(notebook_id, schema_json)
        return "✅ Schema saved successfully."
    except Exception as e:
        return f"❌ Invalid JSON: {str(e)}"


def export_graph_callback(notebook_id: str, format: str, nb_manager: NotebookManager):
    if not notebook_id:
        return None

    if format == "JSON":
        data = nb_manager.export_json(notebook_id)
        filename = f"graph_export_{notebook_id[:8]}.json"
    else:
        data = nb_manager.export_markdown(notebook_id)
        filename = f"graph_report_{notebook_id[:8]}.md"

    path = f"./data/{filename}"
    with open(path, "w", encoding="utf-8") as f:
        f.write(data)
    return path


def update_viz_callback(notebook_id: str, neo4j_client, filter_text: str = None):
    return create_graph_visualization(neo4j_client, notebook_id, filter_text)
