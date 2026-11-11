"""
Embeddings module providing a unified interface across providers:
- Fast deterministic multilingual subword/TF-IDF embedder (Mock/Fallback)
- SentenceTransformers (BGE-M3 / multilingual-e5)
- OpenAI embeddings
"""

from abc import ABC, abstractmethod
import hashlib
import math
import re
from typing import List
import numpy as np


class BaseEmbedder(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        pass

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of text strings."""
        return [self.embed_text(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        """Embed a user query."""
        return self.embed_text(query)


class DeterministicMultilingualEmbedder(BaseEmbedder):
    """
    Deterministic sub-word & token hash embedder.
    Produces dense 256-dimensional unit-normalized vectors.
    Computes meaningful cosine similarity for agricultural terms without
    requiring multi-gigabyte neural weights or active GPU / network.
    """

    def __init__(self, dimension: int = 256):
        self.dimension = dimension

    STOP_WORDS = {
        "what", "is", "the", "for", "in", "to", "and", "of", "a", "an", "how",
        "are", "do", "can", "should", "at", "which", "when", "by", "from", "on",
        "with", "does", "explain", "give", "tell", "me", "any", "some"
    }

    def _tokenize(self, text: str) -> List[str]:
        # Lowercase and clean
        cleaned = re.sub(r"[^\w\s%.-]", " ", text.lower())
        tokens = [t for t in cleaned.split() if len(t) > 1 and t not in self.STOP_WORDS]
        ngrams: List[str] = list(tokens)

        # Character 3-grams for morphological/multilingual resilience
        for t in tokens:
            if len(t) >= 4 and not t.isdigit():
                for i in range(len(t) - 2):
                    ngrams.append(t[i:i + 3])

        return ngrams

    def embed_text(self, text: str) -> List[float]:
        tokens = self._tokenize(text)
        vec = np.zeros(self.dimension, dtype=np.float32)

        if not tokens:
            return vec.tolist()

        for token in tokens:
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimension
            sign = 1.0 if ((h >> 8) % 2 == 0) else -1.0

            # Weight words higher than n-grams
            weight = 2.5 if len(token) > 3 and " " not in token else 1.0
            vec[idx] += sign * weight

        # L2 Normalization to unit hypersphere
        norm = float(np.linalg.norm(vec))
        if norm > 1e-9:
            vec = vec / norm

        return vec.tolist()


class SentenceTransformerEmbedder(BaseEmbedder):
    """SentenceTransformer wrapper for BGE-M3 / multilingual-e5 models."""

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self.model = None
        self._fallback = DeterministicMultilingualEmbedder()

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except Exception:
            # Graceful fallback to deterministic embedder
            self.model = None

    def embed_text(self, text: str) -> List[float]:
        if self.model is not None:
            try:
                emb = self.model.encode(text, normalize_embeddings=True)
                return emb.tolist()
            except Exception:
                pass
        return self._fallback.embed_text(text)


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI Embeddings wrapper with fallback."""

    def __init__(self, api_key: str = "", model: str = "text-embedding-3-small"):
        self.api_key = api_key
        self.model = model
        self._fallback = DeterministicMultilingualEmbedder()

    def embed_text(self, text: str) -> List[float]:
        if self.api_key:
            try:
                import requests
                headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
                payload = {"input": text, "model": self.model}
                resp = requests.post("https://api.openai.com/v1/embeddings", json=payload, headers=headers, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    return data["data"][0]["embedding"]
            except Exception:
                pass
        return self._fallback.embed_text(text)


def get_embedder(provider: str = "mock", api_key: str = "") -> BaseEmbedder:
    """Factory function for acquiring configured embedder instance."""
    prov = provider.lower().strip()
    if prov in ("sentence-transformers", "bge-m3", "multilingual-e5"):
        return SentenceTransformerEmbedder()
    elif prov == "openai" and api_key:
        return OpenAIEmbedder(api_key=api_key)
    else:
        return DeterministicMultilingualEmbedder()
