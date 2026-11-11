"""
CLI Ingestion Script for KisanSathi Agronomic Knowledge Base.
Loads advisory files from data/raw, chunks them, computes embeddings,
and stores them into the vector database.
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import settings
from backend.rag.document_loader import DocumentLoader
from backend.rag.chunking import ChunkerFactory
from backend.rag.embeddings import get_embedder
from backend.rag.vector_store import get_vector_store


def main():
    parser = argparse.ArgumentParser(description="Ingest KisanSathi agronomic advisories into vector store.")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Path to raw advisory files")
    parser.add_argument("--strategy", type=str, default="recursive", choices=["fixed", "recursive", "semantic"], help="Chunking strategy")
    parser.add_argument("--chunk_size", type=int, default=500, help="Target chunk size in characters")
    parser.add_argument("--chunk_overlap", type=int, default=80, help="Overlap between chunks in characters")
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / args.data_dir

    print(f"==================================================")
    print(f"KisanSathi Knowledge Ingestion Engine")
    print(f"Target Directory: {data_path}")
    print(f"Chunking Strategy: {args.strategy}")
    print(f"Chunk Size: {args.chunk_size} | Overlap: {args.chunk_overlap}")
    print(f"==================================================")

    start_time = time.perf_counter()

    # 1. Load documents
    docs = DocumentLoader.load_directory(data_path)
    if not docs:
        print(f"[ERROR] No documents found at {data_path}. Please check directory.")
        sys.exit(1)

    print(f"[INFO] Loaded {len(docs)} documents:")
    for d in docs:
        print(f"  - [{d.crop.upper()}] {d.document_name} ({len(d.sections)} sections, trust: {d.trust_level})")

    # 2. Chunk documents
    chunker = ChunkerFactory.get_chunker(strategy=args.strategy, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    all_chunks = []
    for d in docs:
        chunks = chunker.chunk_document(d)
        all_chunks.extend(chunks)

    print(f"\n[INFO] Generated {len(all_chunks)} chunks across {len(docs)} documents.")

    # 3. Embed chunks
    embedder = get_embedder(settings.EMBEDDING_PROVIDER)
    print(f"[INFO] Generating embeddings using '{settings.EMBEDDING_PROVIDER}' embedder...")
    chunk_texts = [c.text for c in all_chunks]
    embeddings = embedder.embed_documents(chunk_texts)

    # 4. Store in vector database
    store = get_vector_store(settings.VECTOR_STORE_BACKEND)
    store.clear()
    store.add_chunks(all_chunks, embeddings)

    total_time = time.perf_counter() - start_time
    print(f"\n[SUCCESS] Successfully indexed {store.count()} chunks into vector store in {total_time:.2f} seconds.")
    print("==================================================")


if __name__ == "__main__":
    main()
