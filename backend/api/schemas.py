"""
Pydantic schemas for KisanSathi API endpoints.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Farmer inquiry payload."""
    query: str = Field(..., description="Farmer's question in plain text")
    crop: Optional[str] = Field(None, description="Optional crop hint (e.g. tomato, onion, soybean, cotton)")
    language: str = Field("en", description="Language code (en, hi, mr)")
    conversation_id: Optional[str] = Field(None, description="Session ID for multi-turn tracking")


class ChunkMetadataItem(BaseModel):
    """Metadata summary of a retrieved document chunk."""
    chunk_id: str
    document_name: str
    page_number: str
    score: float
    text: str


class QueryResponse(BaseModel):
    """Advisory response with citations, confidence, and latency."""
    query: str
    answer: str
    citations: List[str]
    confidence: float
    is_fallback: bool
    latency_ms: float
    token_count: int
    estimated_cost_usd: float
    provider_used: str
    retrieved_chunks: List[ChunkMetadataItem] = Field(default_factory=list)


class IngestRequest(BaseModel):
    """Trigger ingestion parameters."""
    directory_path: str = Field("data/raw", description="Relative or absolute path to raw docs")
    chunking_strategy: Optional[str] = Field(None, description="fixed, recursive, or semantic")
    chunk_size: Optional[int] = Field(None, description="Character chunk size")
    chunk_overlap: Optional[int] = Field(None, description="Character chunk overlap")


class IngestResponse(BaseModel):
    """Ingestion execution summary."""
    status: str
    documents_loaded: int
    chunks_created: int
    strategy: str
    time_taken_seconds: float
    documents: List[str]


class HealthResponse(BaseModel):
    """System health and operational parameters."""
    status: str
    app_name: str
    environment: str
    default_state: str
    supported_crops: List[str]
    vector_store_chunks: int
    embedding_provider: str
    llm_provider: str
    chunking_strategy: str
