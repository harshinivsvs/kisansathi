"""
API routing endpoints for KisanSathi RAG advisory system.
Supports multilingual queries (Telugu, Hindi, English), voice notes, and farmer profile memory.
"""

from pathlib import Path
import time
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse

from backend.config import settings
from backend.api.schemas import (
    QueryRequest,
    QueryResponse,
    VoiceQueryRequest,
    VoiceQueryResponse,
    IngestRequest,
    IngestResponse,
    HealthResponse,
    FarmerProfileDTO,
    ChunkMetadataItem
)
from backend.rag.document_loader import DocumentLoader
from backend.rag.chunking import ChunkerFactory
from backend.rag.embeddings import get_embedder
from backend.rag.vector_store import get_vector_store
from backend.rag.retriever import RAGRetriever
from backend.rag.generator import RAGGenerator
from backend.rag.multilingual import LanguageDetector
from backend.speech.stt import SpeechToTextService
from backend.speech.tts import TextToSpeechService, AUDIO_DIR
from backend.memory.memory_manager import MemoryManager

router = APIRouter(prefix="/api/v1", tags=["Advisory"])

memory_manager = MemoryManager()
stt_service = SpeechToTextService()
tts_service = TextToSpeechService()


def get_retriever_and_generator():
    embedder = get_embedder(settings.EMBEDDING_PROVIDER)
    vector_store = get_vector_store(settings.VECTOR_STORE_BACKEND)
    retriever = RAGRetriever(
        embedder=embedder,
        vector_store=vector_store,
        confidence_threshold=settings.RETRIEVAL_CONFIDENCE_THRESHOLD
    )
    generator = RAGGenerator(provider=settings.LLM_PROVIDER)
    return retriever, generator


@router.post("/query", response_model=QueryResponse)
async def query_advisory(payload: QueryRequest):
    """
    Primary farm advisory query endpoint.
    Retrieves grounded agronomic context, tracks farmer memory, and returns localized advice.
    """
    if not payload.query or not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")

    # 1. Manage Farmer Profile & Memory
    farmer = memory_manager.get_or_create_farmer(
        phone_number=payload.phone_number,
        default_lang=payload.language or settings.SECONDARY_LANGUAGE
    )
    memory_manager.extract_profile_updates(payload.query, farmer)

    # 2. Language Detection
    detected_lang = payload.language or LanguageDetector.detect_language(payload.query)
    if detected_lang not in ("en", "hi", "te"):
        detected_lang = farmer.preferred_language or settings.SECONDARY_LANGUAGE

    # 3. Contextual Crop resolution from memory if omitted
    target_crop = payload.crop or farmer.primary_crop

    # 4. RAG Retrieval & Generation
    retriever, generator = get_retriever_and_generator()
    retrieval_res = retriever.retrieve(
        query=payload.query.strip(),
        crop=target_crop,
        top_k=settings.RETRIEVAL_TOP_K
    )
    advisory_ans = generator.generate_answer(retrieval_res, target_language=detected_lang)

    # 5. Check sequential missing profile detail (ask one at a time)
    missing_prompt = memory_manager.get_next_missing_profile_prompt(farmer, language=detected_lang)

    # Combine response text for audio if missing prompt exists
    full_audio_text = advisory_ans.answer
    if missing_prompt and not advisory_ans.is_fallback:
        full_audio_text += f"\n\n{missing_prompt}"

    # 6. Audio Voice Note Synthesis (if requested)
    audio_url = None
    audio_duration = None
    if payload.return_audio:
        tts_res = tts_service.synthesize(text=full_audio_text, language=detected_lang)
        audio_url = tts_res.relative_url
        audio_duration = tts_res.duration_estimate_sec

    # 7. Record turn and telemetry in SQLite
    memory_manager.add_turn(
        phone_number=payload.phone_number,
        session_id=payload.session_id or "web",
        user_query=payload.query,
        assistant_answer=advisory_ans.answer,
        language=detected_lang,
        crop_context=target_crop
    )
    memory_manager.log_query(
        phone_number=payload.phone_number,
        query=payload.query,
        answer=advisory_ans.answer,
        retrieved_chunks=advisory_ans.retrieved_chunks,
        citations=advisory_ans.citations,
        confidence=advisory_ans.confidence,
        latency_ms=advisory_ans.latency_ms,
        token_cost=advisory_ans.estimated_cost_usd,
        audio_url=audio_url or "",
        language=detected_lang
    )

    chunks_data = [
        ChunkMetadataItem(
            chunk_id=c["chunk_id"],
            document_name=c["document_name"],
            page_number=c["page_number"],
            score=c["score"],
            text=c["text"]
        )
        for c in advisory_ans.retrieved_chunks
    ]

    farmer_dto = FarmerProfileDTO(
        phone_number=farmer.phone_number,
        name=farmer.name,
        village=farmer.village,
        district=farmer.district,
        state=farmer.state,
        primary_crop=farmer.primary_crop,
        sowing_date=farmer.sowing_date,
        preferred_language=farmer.preferred_language
    )

    return QueryResponse(
        query=advisory_ans.query,
        answer=advisory_ans.answer,
        citations=advisory_ans.citations,
        confidence=advisory_ans.confidence,
        detected_language=detected_lang,
        is_fallback=advisory_ans.is_fallback,
        latency_ms=advisory_ans.latency_ms,
        token_count=advisory_ans.token_count,
        estimated_cost_usd=advisory_ans.estimated_cost_usd,
        provider_used=advisory_ans.provider_used,
        retrieved_chunks=chunks_data,
        audio_url=audio_url,
        audio_duration_sec=audio_duration,
        followup_profile_prompt=missing_prompt,
        farmer_profile=farmer_dto
    )


@router.post("/voice/query", response_model=VoiceQueryResponse)
async def query_voice(payload: VoiceQueryRequest):
    """
    Transcribes audio voice note, corrects vocabulary, and delivers voice-enabled agronomic advice.
    """
    if not payload.audio_path:
        # Fallback simulation
        sample_transcript = "What is the spray for onion tripes and purple blotch?"
        trans_res = stt_service.transcribe_text_mock(sample_transcript, payload.language_hint)
    else:
        trans_res = stt_service.transcribe_audio_file(payload.audio_path, payload.language_hint)

    # Route to main query advisory
    query_req = QueryRequest(
        query=trans_res.corrected_transcript,
        phone_number=payload.phone_number,
        language=trans_res.detected_language,
        return_audio=payload.return_audio
    )
    advisory_resp = await query_advisory(query_req)

    return VoiceQueryResponse(
        raw_transcript=trans_res.raw_transcript,
        corrected_transcript=trans_res.corrected_transcript,
        vocab_corrections_applied=trans_res.vocab_corrections_applied,
        advisory=advisory_resp
    )


@router.get("/voice/audio/{filename}")
async def serve_audio(filename: str):
    """Serves synthesized MP3 voice notes."""
    audio_path = AUDIO_DIR / filename
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(path=str(audio_path), media_type="audio/mpeg", filename=filename)


@router.get("/farmer/{phone_number}", response_model=FarmerProfileDTO)
async def get_farmer_profile(phone_number: str):
    """Retrieves current farmer profile from SQLite memory."""
    farmer = memory_manager.get_or_create_farmer(phone_number)
    return FarmerProfileDTO(
        phone_number=farmer.phone_number,
        name=farmer.name,
        village=farmer.village,
        district=farmer.district,
        state=farmer.state,
        primary_crop=farmer.primary_crop,
        sowing_date=farmer.sowing_date,
        preferred_language=farmer.preferred_language
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents(payload: IngestRequest = IngestRequest()):
    """Ingests advisory documents from raw data directory into the vector store."""
    start_time = time.perf_counter()

    raw_dir = Path(payload.directory_path)
    if not raw_dir.is_absolute():
        raw_dir = Path(__file__).resolve().parent.parent.parent / payload.directory_path

    if not raw_dir.exists():
        raise HTTPException(status_code=404, detail=f"Directory '{payload.directory_path}' not found.")

    docs = DocumentLoader.load_directory(raw_dir)
    if not docs:
        raise HTTPException(status_code=404, detail="No advisory documents found in directory.")

    strategy = payload.chunking_strategy or settings.CHUNKING_STRATEGY
    c_size = payload.chunk_size or settings.CHUNK_SIZE
    c_overlap = payload.chunk_overlap or settings.CHUNK_OVERLAP

    chunker = ChunkerFactory.get_chunker(strategy=strategy, chunk_size=c_size, chunk_overlap=c_overlap)

    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunker.chunk_document(doc))

    embedder = get_embedder(settings.EMBEDDING_PROVIDER)
    embeddings = embedder.embed_documents([c.text for c in all_chunks])

    vector_store = get_vector_store(settings.VECTOR_STORE_BACKEND)
    vector_store.clear()
    vector_store.add_chunks(all_chunks, embeddings)

    elapsed = round(time.perf_counter() - start_time, 3)

    return IngestResponse(
        status="success",
        documents_loaded=len(docs),
        chunks_created=len(all_chunks),
        strategy=strategy,
        time_taken_seconds=elapsed,
        documents=[d.document_name for d in docs]
    )


@router.get("/health", response_model=HealthResponse)
async def get_health():
    """Returns system status, crop scopes, and indexed vector counts."""
    vector_store = get_vector_store(settings.VECTOR_STORE_BACKEND)
    return HealthResponse(
        status="healthy",
        app_name=settings.APP_NAME,
        environment=settings.APP_ENV,
        default_state=settings.DEFAULT_STATE,
        supported_crops=settings.crop_list,
        supported_languages=settings.language_list,
        vector_store_chunks=vector_store.count(),
        embedding_provider=settings.EMBEDDING_PROVIDER,
        llm_provider=settings.LLM_PROVIDER,
        chunking_strategy=settings.CHUNKING_STRATEGY
    )


@router.get("/crops")
async def list_crops():
    """Lists supported crops and languages."""
    return {
        "state": settings.DEFAULT_STATE,
        "supported_crops": settings.crop_list,
        "default_language": settings.DEFAULT_LANGUAGE,
        "secondary_language": settings.SECONDARY_LANGUAGE,
        "supported_languages": settings.language_list
    }
