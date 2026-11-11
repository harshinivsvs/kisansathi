"""
End-to-end unit tests for RAG pipeline, retrieval, citations, and out-of-domain fallback.
"""

from pathlib import Path
import pytest
from backend.rag.document_loader import DocumentLoader
from backend.rag.chunking import ChunkerFactory
from backend.rag.embeddings import get_embedder
from backend.rag.vector_store import InMemoryVectorStore
from backend.rag.retriever import RAGRetriever
from backend.rag.generator import RAGGenerator


@pytest.fixture(scope="module")
def rag_system():
    raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    docs = DocumentLoader.load_directory(raw_dir)
    chunker = ChunkerFactory.get_chunker("recursive", chunk_size=500, chunk_overlap=80)

    chunks = []
    for d in docs:
        chunks.extend(chunker.chunk_document(d))

    embedder = get_embedder("mock")
    embeddings = embedder.embed_documents([c.text for c in chunks])

    store = InMemoryVectorStore()
    store.add_chunks(chunks, embeddings)

    retriever = RAGRetriever(embedder=embedder, vector_store=store, confidence_threshold=0.55)
    generator = RAGGenerator(provider="mock")

    return retriever, generator


def test_retrieval_and_answer_tomato(rag_system):
    retriever, generator = rag_system
    res = retriever.retrieve("What is the seed rate for hybrid tomato per acre in Maharashtra?")
    assert res.detected_crop == "tomato"
    assert res.is_confident is True
    assert len(res.retrieved_chunks) > 0

    ans = generator.generate_answer(res)
    assert ans.is_fallback is False
    assert len(ans.citations) > 0
    assert "Tomato" in ans.citations[0]
    assert "150-200" in ans.answer or "grams" in ans.answer


def test_retrieval_and_answer_onion(rag_system):
    retriever, generator = rag_system
    res = retriever.retrieve("What is the economic threshold level for onion thrips?")
    assert res.detected_crop == "onion"
    assert res.is_confident is True

    ans = generator.generate_answer(res)
    assert ans.is_fallback is False
    assert any("Onion" in c for c in ans.citations)
    assert "thrips" in ans.answer.lower() or "15-20" in ans.answer


def test_out_of_domain_referral(rag_system):
    retriever, generator = rag_system
    # Out of domain crop not in Maharashtra seed docs
    res = retriever.retrieve("How to prune cardamom plants in Kerala high ranges?")
    ans = generator.generate_answer(res)
    assert ans.is_fallback is True
    assert "1800-180-1551" in ans.answer or "KVK" in ans.answer
