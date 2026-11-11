"""
Main entry point for KisanSathi Farm Advisory API.
"""

from contextlib import asynccontextmanager
from pathlib import Path
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.api.routes import router
from backend.api.whatsapp import router as whatsapp_router
from backend.rag.document_loader import DocumentLoader
from backend.rag.chunking import ChunkerFactory
from backend.rag.embeddings import get_embedder
from backend.rag.vector_store import get_vector_store

logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("kisansathi")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: Automatically ingest seed advisory documents if vector store is uninitialized.
    """
    logger.info("Starting up %s (Environment: %s)...", settings.APP_NAME, settings.APP_ENV)
    store = get_vector_store(settings.VECTOR_STORE_BACKEND)

    if store.count() == 0:
        raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
        if raw_dir.exists():
            docs = DocumentLoader.load_directory(raw_dir)
            if docs:
                chunker = ChunkerFactory.get_chunker(
                    strategy=settings.CHUNKING_STRATEGY,
                    chunk_size=settings.CHUNK_SIZE,
                    chunk_overlap=settings.CHUNK_OVERLAP
                )
                all_chunks = []
                for doc in docs:
                    all_chunks.extend(chunker.chunk_document(doc))

                embedder = get_embedder(settings.EMBEDDING_PROVIDER)
                embeddings = embedder.embed_documents([c.text for c in all_chunks])
                store.add_chunks(all_chunks, embeddings)
                logger.info(
                    "Auto-seeded vector store: %d documents, %d chunks indexed using '%s' strategy.",
                    len(docs),
                    len(all_chunks),
                    settings.CHUNKING_STRATEGY
                )

    yield

    logger.info("Shutting down %s...", settings.APP_NAME)


app = FastAPI(
    title=settings.APP_NAME,
    description="Multilingual, Voice-First AI Farm Advisory System for Indian Farmers",
    version="0.1.0",
    lifespan=lifespan
)

# CORS Middleware for web demo and agronomist dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Advisory and WhatsApp API routes
app.include_router(router)
app.include_router(whatsapp_router)

# Mount Web Demo static files if frontend directory exists
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/demo", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


@app.get("/")
def root():
    return {
        "app": settings.APP_NAME,
        "version": "0.1.0",
        "description": "Multilingual Farm Advisory RAG API",
        "docs_url": "/docs",
        "web_demo": "/demo",
        "whatsapp_webhook": "/api/v1/whatsapp/webhook",
        "health_check": "/api/v1/health",
        "state_scope": settings.DEFAULT_STATE,
        "supported_crops": settings.crop_list
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
