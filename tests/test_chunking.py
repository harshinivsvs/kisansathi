"""
Unit tests for KisanSathi chunking strategies.
"""

import pytest
from backend.rag.document_loader import LoadedDocument, DocumentSection
from backend.rag.chunking import ChunkerFactory, FixedChunker, RecursiveChunker, SemanticChunker


@pytest.fixture
def sample_document():
    sections = [
        DocumentSection(
            title="Section 1: Nursery Preparation (Page 1)",
            page_number="Page 1",
            content="Tomato hybrid nursery beds should be raised 15 cm above ground level. Seed rate is 150-200g per acre."
        ),
        DocumentSection(
            title="Section 2: Pest Management (Page 2)",
            page_number="Page 2",
            content="Fruit borer (Helicoverpa armigera) attacks young fruits. Spray Chlorantraniliprole 18.5% SC at 0.3 ml/L."
        )
    ]
    return LoadedDocument(
        document_id="ICAR-TEST-01",
        document_name="ICAR Test Advisory",
        organization="ICAR",
        state="Maharashtra",
        crop="tomato",
        season="Kharif",
        trust_level="official_icar",
        year="2024",
        file_path="/dummy/path",
        full_text="Tomato hybrid nursery beds should be raised 15 cm above ground level. Fruit borer attacks young fruits.",
        sections=sections
    )


def test_fixed_chunker(sample_document):
    chunker = ChunkerFactory.get_chunker("fixed", chunk_size=80, chunk_overlap=20)
    assert isinstance(chunker, FixedChunker)
    chunks = chunker.chunk_document(sample_document)
    assert len(chunks) > 0
    for c in chunks:
        assert c.crop == "tomato"
        assert c.document_id == "ICAR-TEST-01"
        assert len(c.text) <= 80


def test_recursive_chunker(sample_document):
    chunker = ChunkerFactory.get_chunker("recursive", chunk_size=200, chunk_overlap=30)
    assert isinstance(chunker, RecursiveChunker)
    chunks = chunker.chunk_document(sample_document)
    assert len(chunks) >= 2
    # Check section title preservation
    pages = [c.page_number for c in chunks]
    assert "Page 1" in pages
    assert "Page 2" in pages


def test_semantic_chunker(sample_document):
    chunker = ChunkerFactory.get_chunker("semantic")
    assert isinstance(chunker, SemanticChunker)
    chunks = chunker.chunk_document(sample_document)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.strategy == "semantic"
        assert "Tomato" in c.text or "tomato" in c.crop
