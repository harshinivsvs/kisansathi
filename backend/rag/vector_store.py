"""
Vector store implementation for KisanSathi RAG.
Provides high-performance in-memory vector storage with cosine similarity,
metadata filtering, and pluggable vector database support.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from backend.rag.chunking import DocumentChunk


class BaseVectorStore(ABC):
    """Abstract vector store interface."""

    @abstractmethod
    def add_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        """Add chunks and embeddings into store."""
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 4,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """Search top_k nearest chunks by cosine similarity."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return total chunks in store."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all stored chunks."""
        pass


class InMemoryVectorStore(BaseVectorStore):
    """
    In-memory vector store powered by NumPy matrix operations.
    Supports real-time filtering by crop, trust level, and state.
    """

    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self.embeddings_matrix: Optional[np.ndarray] = None

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        if not chunks:
            return

        new_embeddings = np.array(embeddings, dtype=np.float32)

        # Normalize each row to unit length
        norms = np.linalg.norm(new_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized_embeddings = new_embeddings / norms

        if self.embeddings_matrix is None or len(self.chunks) == 0:
            self.chunks = list(chunks)
            self.embeddings_matrix = normalized_embeddings
        else:
            self.chunks.extend(chunks)
            self.embeddings_matrix = np.vstack([self.embeddings_matrix, normalized_embeddings])

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 4,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        if self.embeddings_matrix is None or len(self.chunks) == 0:
            return []

        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        # Cosine similarity is dot product of normalized vectors
        scores = np.dot(self.embeddings_matrix, q_vec)

        # Apply metadata filters if provided
        valid_indices = []
        for idx, chunk in enumerate(self.chunks):
            match = True
            if filter_metadata:
                for k, v in filter_metadata.items():
                    val = str(v).lower()
                    if k == "crop" and chunk.crop.lower() != val and val != "all":
                        match = False
                        break
                    elif k == "state" and chunk.state.lower() != val:
                        match = False
                        break
                    elif k == "trust_level" and chunk.trust_level.lower() != val:
                        match = False
                        break
            if match:
                valid_indices.append(idx)

        if not valid_indices:
            return []

        filtered_scores = scores[valid_indices]
        # Rank by score descending
        ranked_local_indices = np.argsort(-filtered_scores)[:top_k]

        results: List[Tuple[DocumentChunk, float]] = []
        for loc_idx in ranked_local_indices:
            orig_idx = valid_indices[loc_idx]
            sim_score = float(filtered_scores[loc_idx])
            results.append((self.chunks[orig_idx], sim_score))

        return results

    def count(self) -> int:
        return len(self.chunks)

    def clear(self) -> None:
        self.chunks.clear()
        self.embeddings_matrix = None


# Global vector store singleton
_vector_store_instance: Optional[BaseVectorStore] = None


def get_vector_store(backend: str = "inmemory") -> BaseVectorStore:
    """Acquires vector store singleton."""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = InMemoryVectorStore()
    return _vector_store_instance
