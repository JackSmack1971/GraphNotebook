"""
Semantic chunking with token-level control.
Preserves paragraph boundaries, attaches metadata.
"""

from dataclasses import dataclass
from typing import List

import tiktoken


@dataclass
class Chunk:
    id: str
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    token_count: int
    page_number: int = 0
    section_header: str = ""


class SemanticChunker:
    """
    Token-aware chunker that respects paragraph boundaries.
    Uses tiktoken for accurate token counting.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        encoding_name: str = "cl100k_base",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enc = tiktoken.get_encoding(encoding_name)

    def chunk_text(self, text: str, doc_id: str = "") -> List[Chunk]:
        """Split text into overlapping chunks at paragraph boundaries."""
        paragraphs = text.split("\n\n")
        chunks = []
        current_tokens = []
        current_text_parts = []
        current_start = 0
        chunk_index = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = self.enc.encode(para)

            # If adding this paragraph exceeds chunk_size, flush
            if (
                current_tokens
                and len(current_tokens) + len(para_tokens) > self.chunk_size
            ):
                chunk_text = "\n\n".join(current_text_parts)
                chunks.append(
                    Chunk(
                        id=f"{doc_id}_chunk_{chunk_index:04d}",
                        text=chunk_text,
                        chunk_index=chunk_index,
                        start_char=current_start,
                        end_char=current_start + len(chunk_text),
                        token_count=len(current_tokens),
                    )
                )
                chunk_index += 1

                # Overlap: keep last N tokens worth of text
                # We do this by decoding the last chunk_overlap tokens, but it's simpler
                # to just keep overlapping text at paragraph bound if possible.
                # Here we just decode the last N tokens:
                overlap_text = self.enc.decode(current_tokens[-self.chunk_overlap :])
                current_tokens = self.enc.encode(overlap_text)
                current_text_parts = [overlap_text]
                current_start = current_start + len(chunk_text) - len(overlap_text)

            current_tokens.extend(para_tokens)
            current_text_parts.append(para)

        # Flush remaining
        if current_text_parts:
            chunk_text = "\n\n".join(current_text_parts)
            chunks.append(
                Chunk(
                    id=f"{doc_id}_chunk_{chunk_index:04d}",
                    text=chunk_text,
                    chunk_index=chunk_index,
                    start_char=current_start,
                    end_char=current_start + len(chunk_text),
                    token_count=len(current_tokens),
                )
            )

        return chunks
