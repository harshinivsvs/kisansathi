"""
API routing endpoints for KisanSathi RAG advisory system.
"""

from pathlib import Path
import time
from typing import List
from fastapi import APIRouter, HTTPException, Depends

from backend.config import settings
from backend.api.schemas import (
    QueryRequest,
    QueryResponse,
    IngestRequest,
    IngestResponse,
    HealthResponse,
    ChunkMetadataItem
)
from backend.rag.document_loader import DocumentLoader
from backend.rag.chunking import ChunkerFactory
from backend.rag.embeddings import get_embedder
from backend.rag.vector_store import get_vector_store
from backend.rag.retriever import RAGRetriever
from backend.rag.generator import RAGGenerator

router = APIRouter(prefix="/api/v1", tags=["Advisory"])


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
    Retrieves grounded agronomic context and synthesizes source-cited advice.
    """
    if not payload.query or not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")

    retriever, generator = get_retriever_and_generator()

    retrieval_res = retriever.retrieve(
        query=payload.query.strip(),
        crop=payload.crop,
        top_k=settings.RETRIEVAL_TOP_K
    )

    advisory_ans = generator.generate_answer(retrieval_res)

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

    return QueryResponse(
        query=advisory_ans.query,
        answer=advisory_ans.answer,
        citations=advisory_ans.citations,
        confidence=advisory_ans.confidence,
        is_fallback=advisory_ans.is_fallback,
        latency_ms=advisory_ans.latency_ms,
        token_count=advisory_ans.token_count,
        estimated_cost_usd=advisory_ans.estimated_cost_usd,
        provider_used=advisory_ans.provider_used,
        retrieved_chunks=chunks_data
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents(payload: IngestRequest = IngestRequest()):
    """
    Ingests advisory documents from raw data directory into the vector store.
    """
    start_time = time.perf_counter()

    raw_dir = Path(payload.directory_path)
    if not raw_dir.is_absolute():
        raw_dir = Path(__file__).resolve().parent.parent.parent / payload.directory_path

    if not raw_dir.exists():
        raise HTTPException(status_code=404, detail=f"Directory '{payload.directory_path}' not found.")

    docs = DocumentLoader.load_directory(raw_dir)
    if not docs:
        raise HTTPException(status_code=404, detail="No markdown/text advisory documents found in directory.")

    strategy = payload.chunking_strategy or settings.CHUNKING_STRATEGY
    c_size = payload.chunk_size or settings.CHUNK_SIZE
    c_overlap = payload.chunk_overlap or settings.CHUNK_OVERLAP

    chunker = ChunkerFactory.get_chunker(strategy=strategy, chunk_size=c_size, chunk_overlap=c_overlap)

    all_chunks = []
    for doc in docs:
        chunks = chunker.chunk_document(doc)
        all_chunks.extend(chunks)

    embedder = get_embedder(settings.EMBEDDING_PROVIDER)
    chunk_texts = [c.text for c in all_chunks]
    embeddings = embedder.embed_documents(chunk_texts)

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
        vector_store_chunks=vector_store.count(),
        embedding_provider=settings.EMBEDDING_PROVIDER,
        llm_provider=settings.LLM_PROVIDER,
        chunking_strategy=settings.CHUNKING_STRATEGY
    )


@router.get("/crops")
async def list_crops():
    """Lists supported crops for Maharashtra v1 scope."""
    return {
        "state": settings.DEFAULT_STATE,
        "supported_crops": settings.crop_list,
        "default_language": settings.DEFAULT_LANGUAGE,
        "secondary_language": settings.SECONDARY_LANGUAGE
    }
