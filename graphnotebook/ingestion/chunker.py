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
        if not text:
            return []

        # Find paragraphs and their absolute start positions in original text
        paras_with_offsets = []
        last_pos = 0
        for p in text.split("\n\n"):
            # find where this paragraph actually starts (skip potential leading \n)
            start_in_text = text.find(p, last_pos)
            paras_with_offsets.append({
                "text": p,
                "start": start_in_text,
                "end": start_in_text + len(p)
            })
            last_pos = start_in_text + len(p)

        chunks = []
        current_paras = []
        current_tokens_count = 0
        chunk_index = 0

        for p_info in paras_with_offsets:
            p_text = p_info["text"]
            p_tokens = self.enc.encode(p_text)
            
            # If adding this exceeds chunk_size, flush
            if current_paras and (current_tokens_count + len(p_tokens) > self.chunk_size):
                # Build current chunk
                c_start = current_paras[0]["start"]
                c_end = current_paras[-1]["end"]
                c_text = text[c_start:c_end]
                
                chunks.append(Chunk(
                    id=f"{doc_id}_chunk_{chunk_index:04d}",
                    text=c_text,
                    chunk_index=chunk_index,
                    start_char=c_start,
                    end_char=c_end,
                    token_count=current_tokens_count,
                ))
                chunk_index += 1

                # Overlap: find how many paragraphs we need to keep to satisfy chunk_overlap
                # We work backwards from the end of current_paras
                overlap_paras = []
                overlap_tokens_count = 0
                for op in reversed(current_paras):
                    op_tokens_len = len(self.enc.encode(op["text"]))
                    if overlap_tokens_count + op_tokens_len <= self.chunk_overlap:
                        overlap_paras.insert(0, op)
                        overlap_tokens_count += op_tokens_len
                    else:
                        break
                
                # If no full paragraph fits in overlap, just take the last one anyway to avoid gaps
                if not overlap_paras and current_paras:
                    overlap_paras = [current_paras[-1]]
                    overlap_tokens_count = len(self.enc.encode(current_paras[-1]["text"]))

                current_paras = overlap_paras
                current_tokens_count = overlap_tokens_count

            current_paras.append(p_info)
            current_tokens_count += len(p_tokens)

        # Flush remaining
        if current_paras:
            c_start = current_paras[0]["start"]
            c_end = current_paras[-1]["end"]
            c_text = text[c_start:c_end]
            chunks.append(Chunk(
                id=f"{doc_id}_chunk_{chunk_index:04d}",
                text=c_text,
                chunk_index=chunk_index,
                start_char=c_start,
                end_char=c_end,
                token_count=current_tokens_count,
            ))

        return chunks
