"""
Pydantic schemas for KisanSathi API endpoints (Advisory, Multilingual Voice, and Farmer Profile).
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Farmer inquiry payload supporting text and voice."""
    query: str = Field(..., description="Farmer's question in plain text")
    crop: Optional[str] = Field(None, description="Optional crop hint (tomato, onion, soybean, cotton)")
    language: Optional[str] = Field(None, description="Language code (en, hi, te). Auto-detected if omitted.")
    phone_number: str = Field("+919876543210", description="Farmer's WhatsApp or mobile number for profile persistence")
    session_id: Optional[str] = Field("web-session", description="Session ID for multi-turn tracking")
    return_audio: bool = Field(False, description="Whether to synthesize and return an MP3 voice note")


class ChunkMetadataItem(BaseModel):
    """Metadata summary of a retrieved document chunk."""
    chunk_id: str
    document_name: str
    page_number: str
    score: float
    text: str


class FarmerProfileDTO(BaseModel):
    """Farmer profile representation."""
    phone_number: str
    name: Optional[str] = None
    village: Optional[str] = None
    district: Optional[str] = None
    state: str = "Maharashtra"
    primary_crop: Optional[str] = None
    sowing_date: Optional[str] = None
    preferred_language: str = "te"


class QueryResponse(BaseModel):
    """Advisory response with citations, confidence, memory prompts, and audio."""
    query: str
    answer: str
    citations: List[str]
    confidence: float
    detected_language: str
    is_fallback: bool
    latency_ms: float
    token_count: int
    estimated_cost_usd: float
    provider_used: str
    retrieved_chunks: List[ChunkMetadataItem] = Field(default_factory=list)
    audio_url: Optional[str] = None
    audio_duration_sec: Optional[float] = None
    followup_profile_prompt: Optional[str] = None
    farmer_profile: Optional[FarmerProfileDTO] = None


class VoiceQueryRequest(BaseModel):
    """Voice note inquiry payload."""
    audio_path: Optional[str] = Field(None, description="Local path to uploaded voice note")
    phone_number: str = Field("+919876543210", description="Farmer's phone number")
    language_hint: Optional[str] = Field(None, description="Optional hint: en, hi, te")
    return_audio: bool = Field(True, description="Return voice response")


class VoiceQueryResponse(BaseModel):
    """Voice inquiry result with speech transcription details and advisory answer."""
    raw_transcript: str
    corrected_transcript: str
    vocab_corrections_applied: bool
    advisory: QueryResponse


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
    supported_languages: List[str]
    vector_store_chunks: int
    embedding_provider: str
    llm_provider: str
    chunking_strategy: str
