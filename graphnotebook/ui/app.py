"""
Gradio application layout (Blocks) for Phase 4.
Features: Notebook management, Interactive PyVis graph, Streaming chat, and HITL review.
"""

import json

import gradio as gr

from graphnotebook.notebooks.manager import NotebookManager
from graphnotebook.retrieval.router import build_query_agent
from graphnotebook.ui.callbacks import (
    create_notebook_callback,
    delete_notebook_callback,
    export_graph_callback,
    handle_query_stream,
    handle_upload,
    refresh_notebooks,
    save_schema_callback,
    update_stats,
    update_viz_callback,
)


def build_app(settings, neo4j_client, llm_gateway, embedding_engine) -> gr.Blocks:
    """Constructs the Gradio user interface."""

    nb_manager = NotebookManager(neo4j_client)
    query_agent = build_query_agent(neo4j_client)

    # Default Schema for editor
    DEFAULT_SCHEMA = {
        "entities": [
            "Person",
            "Organization",
            "Concept",
            "Technology",
            "Location",
            "Event",
            "Metric",
        ],
        "relationships": [
            "WORKS_FOR",
            "FOUNDED",
            "USES",
            "RELATED_TO",
            "PART_OF",
            "LOCATED_IN",
            "CAUSED_BY",
        ],
    }

    with gr.Blocks(title="GraphNotebook", theme=gr.themes.Soft()) as app:
        # State variables
        current_nb_id = gr.State("")

        gr.Markdown("# 🧠 GraphNotebook [Phase 4: Polish]")
        gr.Markdown(
            "*Enterprise-grade personal GraphRAG. Streaming, Visuals, and Management.*"
        )

        with gr.Row():
            # --- LEFT PANEL: Management ---
            with gr.Column(scale=1):
                gr.Markdown("### 🏢 Notebook Selection")
                with gr.Row():
                    notebook_dropdown = gr.Dropdown(
                        label="Active Notebook", choices=[], interactive=True, scale=3
                    )
                    refresh_nb_btn = gr.Button("🔄", scale=1)

                with gr.Accordion("New/Delete Notebook", open=False):
                    new_nb_name = gr.Textbox(label="Name", placeholder="My New Project")
                    new_nb_desc = gr.Textbox(label="Description")
                    create_nb_btn = gr.Button("Create Notebook", variant="secondary")
                    delete_nb_btn = gr.Button("Delete Current Notebook", variant="stop")
                    nb_op_status = gr.Markdown("")

                gr.Markdown("### 📥 Ingestion")
                with gr.Group():
                    file_upload = gr.File(
                        label="Upload Documents",
                        file_count="multiple",
                    )
                    upload_btn = gr.Button("Start Ingestion", variant="primary")
                    upload_status = gr.Textbox(
                        label="In-Progress Status", interactive=False, lines=4
                    )
                
                gr.Markdown("### 📂 Managed Documents")
                doc_list = gr.Dataframe(
                    headers=["Filename", "Type", "Chunks", "Status", "Last Indexed"],
                    datatype=["str", "str", "number", "str", "str"],
                    interactive=False,
                    label="Documents in this Notebook",
                )

                # HITL Review Section (Accordion that appears/opens on extraction)
                with gr.Accordion("📝 Entity Review (HITL)", open=False):
                    gr.Markdown("Review extracted entities before resolution.")
                    gr.Dataframe(
                        headers=["Approve", "Name", "Type", "Description"],
                        datatype=["bool", "str", "str", "str"],
                        interactive=True,
                        col_count=(4, "fixed"),
                    )
                    gr.Button("Approve & Continue", variant="primary")

                gr.Markdown("### 📊 Stats")
                stats_display = gr.Markdown(value="Select a notebook to see stats.")

            # --- RIGHT PANEL: Interaction ---
            with gr.Column(scale=2):
                with gr.Tabs():
                    # Tab 1: Streaming Chat
                    with gr.Tab("💬 Chat"):
                        chatbot = gr.Chatbot(
                            height=500, show_copy_button=True, type="messages"
                        )
                        query_input = gr.Textbox(
                            placeholder="Ask a question...",
                            label="Query",
                        )
                        with gr.Row():
                            search_mode = gr.Radio(
                                choices=["auto", "local", "global", "vector-only"],
                                value="auto",
                                label="Strategy",
                                scale=2,
                            )
                            clear_chat_btn = gr.Button("Clear History", scale=1)

                    # Tab 2: Interactive Explorer
                    with gr.Tab("🕸️ Graph Explorer"):
                        with gr.Row():
                            viz_filter = gr.Textbox(
                                placeholder="Filter by name...",
                                label="Search Graph",
                                scale=3,
                            )
                            refresh_viz_btn = gr.Button("Refresh View", scale=1)

                        graph_html = gr.HTML(label="Interactive Knowledge Graph")

                        with gr.Row():
                            export_json_btn = gr.Button("💾 Export JSON")
                            export_md_btn = gr.Button("📑 Export Markdown")
                        export_file = gr.File(label="Download Export")

                    # Tab 3: Schema Editor
                    with gr.Tab("⚙️ Schema"):
                        gr.Markdown("### Custom Extraction Schema")
                        schema_editor = gr.Code(
                            value=json.dumps(DEFAULT_SCHEMA, indent=2),
                            language="json",
                            label="JSON Schema Editor",
                        )
                        save_schema_btn = gr.Button("Save Schema", variant="primary")
                        reset_schema_btn = gr.Button("Reset to Default")
                        schema_status = gr.Markdown("")

        # --- Event Wiring ---

        # 1. Notebook Mgmt
        def on_nb_change(nb_id):
            if not nb_id:
                return "", "Select a notebook", []
            stats = update_stats(nb_id, neo4j_client)
            docs = nb_manager.get_documents(nb_id)
            doc_rows = [[d["filename"], d["file_type"], d["chunk_count"], d["status"], str(d["ingested_at"])] for d in docs]
            return nb_id, stats, doc_rows

        notebook_dropdown.change(
            fn=on_nb_change,
            inputs=[notebook_dropdown],
            outputs=[current_nb_id, stats_display, doc_list],
        )

        create_nb_btn.click(
            fn=lambda n, d: create_notebook_callback(n, d, nb_manager),
            inputs=[new_nb_name, new_nb_desc],
            outputs=[notebook_dropdown, nb_op_status],
        )

        delete_nb_btn.click(
            fn=lambda nb_id: delete_notebook_callback(nb_id, nb_manager),
            inputs=[current_nb_id],
            outputs=[notebook_dropdown, nb_op_status],
        )

        refresh_nb_btn.click(
            fn=lambda: refresh_notebooks(nb_manager), outputs=[notebook_dropdown]
        )

        # 2. Ingestion
        upload_btn.click(
            fn=handle_upload,
            inputs=[
                file_upload,
                current_nb_id,
                gr.State(neo4j_client),
                gr.State(embedding_engine),
                gr.State(settings),
                gr.State(llm_gateway),
            ],
            outputs=[upload_status],
        ).then(
            fn=on_nb_change,
            inputs=[current_nb_id],
            outputs=[current_nb_id, stats_display, doc_list],
        )

        # 3. Streaming Chat
        query_input.submit(
            fn=handle_query_stream,
            inputs=[
                query_input,
                chatbot,
                current_nb_id,
                search_mode,
                gr.State(query_agent),
                gr.State(embedding_engine),
                gr.State(llm_gateway),
            ],
            outputs=[chatbot],
        ).then(fn=lambda: "", outputs=[query_input])

        clear_chat_btn.click(lambda: [], None, chatbot, queue=False)

        # 4. Graph Explorer
        refresh_viz_btn.click(
            fn=update_viz_callback,
            inputs=[current_nb_id, gr.State(neo4j_client), viz_filter, full_graph_toggle],
            outputs=[graph_html],
        )

        export_json_btn.click(
            fn=lambda nb_id: export_graph_callback(nb_id, "JSON", nb_manager),
            inputs=[current_nb_id],
            outputs=[export_file],
        )

        export_md_btn.click(
            fn=lambda nb_id: export_graph_callback(nb_id, "MD", nb_manager),
            inputs=[current_nb_id],
            outputs=[export_file],
        )

        # 5. Schema
        save_schema_btn.click(
            fn=lambda nb_id, schema: save_schema_callback(nb_id, schema, nb_manager),
            inputs=[current_nb_id, schema_editor],
            outputs=[schema_status],
        )

        reset_schema_btn.click(
            fn=lambda: json.dumps(DEFAULT_SCHEMA, indent=2), outputs=[schema_editor]
        )

        # Initialize list on load
        app.load(fn=lambda: refresh_notebooks(nb_manager), outputs=[notebook_dropdown])

    return app
