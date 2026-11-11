"""
Unit tests for embeddings and vector normalization.
"""

import numpy as np
import pytest
from backend.rag.embeddings import DeterministicMultilingualEmbedder, get_embedder


def test_embedder_normalization():
    embedder = DeterministicMultilingualEmbedder(dimension=256)
    vec = embedder.embed_text("Tomato fruit borer Helicoverpa armigera spray chlorantraniliprole")
    norm = np.linalg.norm(vec)
    assert pytest.approx(norm, rel=1e-3) == 1.0
    assert len(vec) == 256


def test_similarity_separation():
    embedder = DeterministicMultilingualEmbedder(dimension=256)
    q = np.array(embedder.embed_query("What is the spray for onion thrips?"))

    matched_doc = np.array(embedder.embed_text("Onion thrips Thrips tabaci control with Fipronil or Profenofos spray."))
    unrelated_doc = np.array(embedder.embed_text("Apply nitrogen fertilizer to apple trees during winter dormancy in Kashmir."))

    sim_matched = float(np.dot(q, matched_doc))
    sim_unrelated = float(np.dot(q, unrelated_doc))

    assert sim_matched > sim_unrelated
    assert sim_matched > 0.25
    assert sim_unrelated < 0.15


def test_factory_fallback():
    embedder = get_embedder("mock")
    assert isinstance(embedder, DeterministicMultilingualEmbedder)
