"""
Chunking module supporting Fixed, Recursive, and Semantic strategies.
Allows systematic comparison of chunking approaches for agricultural advisory RAG.
"""

from abc import ABC, abstractmethod
import re
from typing import Any, Dict, List
from pydantic import BaseModel, Field

from backend.rag.document_loader import LoadedDocument, DocumentSection


class DocumentChunk(BaseModel):
    """Represents an embedded unit of knowledge with rich source citations."""
    chunk_id: str
    document_id: str
    document_name: str
    crop: str
    state: str
    page_number: str
    section_title: str
    trust_level: str
    strategy: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseChunker(ABC):
    """Abstract base class for document chunkers."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 80):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def chunk_document(self, document: LoadedDocument) -> List[DocumentChunk]:
        """Splits a loaded document into chunks."""
        pass


class FixedChunker(BaseChunker):
    """Fixed-size character chunking with uniform overlap."""

    def chunk_document(self, document: LoadedDocument) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        full_text = document.full_text
        step = max(1, self.chunk_size - self.chunk_overlap)

        idx = 0
        start = 0
        while start < len(full_text):
            end = min(start + self.chunk_size, len(full_text))
            chunk_text = full_text[start:end].strip()

            if chunk_text:
                # Find best matching section or page
                matched_sec = "Main Document"
                matched_page = "Page 1"
                for sec in document.sections:
                    if chunk_text[:50] in sec.content or sec.content[:50] in chunk_text:
                        matched_sec = sec.title
                        matched_page = sec.page_number
                        break

                chunk_id = f"{document.document_id}_fixed_{idx}"
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        document_id=document.document_id,
                        document_name=document.document_name,
                        crop=document.crop,
                        state=document.state,
                        page_number=matched_page,
                        section_title=matched_sec,
                        trust_level=document.trust_level,
                        strategy="fixed",
                        text=chunk_text,
                        metadata={
                            "document_id": document.document_id,
                            "crop": document.crop,
                            "page": matched_page,
                            "trust_level": document.trust_level,
                            "start_char": start,
                            "end_char": end,
                        }
                    )
                )
                idx += 1

            if end >= len(full_text):
                break
            start += step

        return chunks


class RecursiveChunker(BaseChunker):
    """Recursive character/paragraph chunking respecting semantic boundaries."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 80):
        super().__init__(chunk_size, chunk_overlap)
        self.separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively splits text using hierarchical separators."""
        final_chunks: List[str] = []
        if not separators:
            # Fallback to direct slicing
            return [text[i:i + self.chunk_size] for i in range(0, len(text), max(1, self.chunk_size - self.chunk_overlap))]

        separator = separators[0]
        remaining_separators = separators[1:]

        splits = text.split(separator) if separator in text else [text]
        current_chunk = ""

        for piece in splits:
            piece = piece.strip()
            if not piece:
                continue

            test_chunk = f"{current_chunk}{separator}{piece}".strip() if current_chunk else piece

            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    final_chunks.append(current_chunk)
                    # Handle overlap from previous chunk
                    if self.chunk_overlap > 0 and len(current_chunk) > self.chunk_overlap:
                        overlap_text = current_chunk[-self.chunk_overlap:]
                        current_chunk = f"{overlap_text} {piece}".strip()
                    else:
                        current_chunk = piece
                else:
                    # Single piece is larger than chunk_size, recurse on remaining separators
                    sub_pieces = self._split_text(piece, remaining_separators)
                    final_chunks.extend(sub_pieces)
                    current_chunk = ""

        if current_chunk:
            final_chunks.append(current_chunk)

        return final_chunks

    def chunk_document(self, document: LoadedDocument) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        chunk_idx = 0

        for section in document.sections:
            section_text = f"[{document.crop.upper()} ADVISORY - {section.title}]\n{section.content}".strip()
            raw_splits = self._split_text(section_text, self.separators)

            for split in raw_splits:
                split = split.strip()
                if not split or len(split) < 30:
                    continue

                chunk_id = f"{document.document_id}_recur_{chunk_idx}"
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        document_id=document.document_id,
                        document_name=document.document_name,
                        crop=document.crop,
                        state=document.state,
                        page_number=section.page_number,
                        section_title=section.title,
                        trust_level=document.trust_level,
                        strategy="recursive",
                        text=split,
                        metadata={
                            "document_id": document.document_id,
                            "crop": document.crop,
                            "page": section.page_number,
                            "section": section.title,
                            "trust_level": document.trust_level,
                        }
                    )
                )
                chunk_idx += 1

        return chunks


class SemanticChunker(BaseChunker):
    """Semantic chunking preserving topic units, bullet lists, and heading contexts."""

    def chunk_document(self, document: LoadedDocument) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        chunk_idx = 0

        for section in document.sections:
            # Split section by numbered items (e.g. '1. ', '2. ') or paragraphs
            sub_units = re.split(r"\n(?=\d+\.\s+|-\s+)", section.content)

            for unit in sub_units:
                unit = unit.strip()
                if not unit or len(unit) < 25:
                    continue

                # Add contextual prefix so each chunk is self-contained
                context_prefix = f"[{document.document_name} | Crop: {document.crop.title()} | {section.title} ({section.page_number})]\n"
                chunk_text = f"{context_prefix}{unit}"

                chunk_id = f"{document.document_id}_sem_{chunk_idx}"
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        document_id=document.document_id,
                        document_name=document.document_name,
                        crop=document.crop,
                        state=document.state,
                        page_number=section.page_number,
                        section_title=section.title,
                        trust_level=document.trust_level,
                        strategy="semantic",
                        text=chunk_text,
                        metadata={
                            "document_id": document.document_id,
                            "crop": document.crop,
                            "page": section.page_number,
                            "section": section.title,
                            "trust_level": document.trust_level,
                        }
                    )
                )
                chunk_idx += 1

        return chunks


class ChunkerFactory:
    """Factory for instantiating document chunkers by strategy name."""

    @staticmethod
    def get_chunker(strategy: str = "recursive", chunk_size: int = 500, chunk_overlap: int = 80) -> BaseChunker:
        strat = strategy.lower().strip()
        if strat == "fixed":
            return FixedChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif strat == "semantic":
            return SemanticChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif strat == "recursive":
            return RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        else:
            # Default to recursive
            return RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
