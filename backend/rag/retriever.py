"""
Retriever module for KisanSathi RAG.
Orchestrates query embedding, crop filtering, similarity search, and confidence evaluation.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from backend.rag.chunking import DocumentChunk
from backend.rag.embeddings import BaseEmbedder, get_embedder
from backend.rag.vector_store import BaseVectorStore, get_vector_store
from backend.config import settings


class RetrievedChunk(BaseModel):
    """Chunk enriched with similarity score and human-readable citation."""
    chunk: DocumentChunk
    similarity_score: float
    citation: str

    @property
    def document_name(self) -> str:
        return self.chunk.document_name

    @property
    def page_number(self) -> str:
        return self.chunk.page_number

    @property
    def text(self) -> str:
        return self.chunk.text


class RetrievalResult(BaseModel):
    """Complete output of retrieval stage."""
    query: str
    detected_crop: Optional[str]
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)
    top_score: float = 0.0
    is_confident: bool = False
    confidence_threshold: float = 0.58


class RAGRetriever:
    """Retrieves relevant agronomic advisory chunks for a farmer query."""

    CROP_KEYWORDS = {
        "tomato": ["tomato", "tamatar", "टोमॅटो", "टमाटर"],
        "onion": ["onion", "kanda", "कांदा", "pyaz", "प्याज़"],
        "soybean": ["soybean", "soya", "सोयाबीन"],
        "cotton": ["cotton", "kapas", "कापूस", "कपास"]
    }

    def __init__(
        self,
        embedder: Optional[BaseEmbedder] = None,
        vector_store: Optional[BaseVectorStore] = None,
        confidence_threshold: Optional[float] = None
    ):
        self.embedder = embedder or get_embedder(settings.EMBEDDING_PROVIDER)
        self.vector_store = vector_store or get_vector_store(settings.VECTOR_STORE_BACKEND)
        self.confidence_threshold = confidence_threshold or settings.RETRIEVAL_CONFIDENCE_THRESHOLD

    def detect_crop(self, query: str) -> Optional[str]:
        """Detects if query references one of our supported crops across English, Hindi, and Telugu."""
        from backend.rag.multilingual import MultilingualAgriThesaurus
        return MultilingualAgriThesaurus.detect_crop_multilingual(query)

    def calibrate_confidence(self, raw_score: float) -> float:
        """
        Calibrate raw embedding cosine similarity to standardized [0.0, 1.0] confidence.
        For deterministic token/hash embeddings:
          raw_score <= 0.10 is out-of-domain noise -> 0.0
          raw_score 0.12 - 0.22 is weak overlap -> 0.10 - 0.40
          raw_score >= 0.28 is strong agricultural match -> >= 0.60
        """
        if raw_score <= 0.10:
            return 0.0
        calibrated = (raw_score - 0.10) / (0.38 - 0.10)
        return round(min(1.0, max(0.0, calibrated)), 4)

    def retrieve(
        self,
        query: str,
        crop: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> RetrievalResult:
        """Retrieves top matching advisory chunks for query."""
        from backend.rag.multilingual import MultilingualAgriThesaurus, LanguageDetector

        k = top_k or settings.RETRIEVAL_TOP_K
        detected_crop = crop or self.detect_crop(query)
        detected_lang = LanguageDetector.detect_language(query)

        filter_metadata: Dict[str, Any] = {}
        if detected_crop:
            filter_metadata["crop"] = detected_crop

        # Expand Indic queries with cross-lingual agronomic keywords
        expanded_query = MultilingualAgriThesaurus.expand_query_for_retrieval(query, detected_lang)
        query_emb = self.embedder.embed_query(expanded_query)

        raw_results = self.vector_store.search(
            query_embedding=query_emb,
            top_k=k,
            filter_metadata=filter_metadata if filter_metadata else None
        )

        retrieved: List[RetrievedChunk] = []
        raw_top_score = 0.0

        for chunk, score in raw_results:
            if score > raw_top_score:
                raw_top_score = score

            citation = f"{chunk.document_name} ({chunk.page_number})"
            retrieved.append(
                RetrievedChunk(
                    chunk=chunk,
                    similarity_score=round(score, 4),
                    citation=citation
                )
            )

        calibrated_conf = self.calibrate_confidence(raw_top_score)
        is_confident = (calibrated_conf >= self.confidence_threshold) and (len(retrieved) > 0)

        return RetrievalResult(
            query=query,
            detected_crop=detected_crop,
            retrieved_chunks=retrieved,
            top_score=calibrated_conf,
            is_confident=is_confident,
            confidence_threshold=self.confidence_threshold
        )
