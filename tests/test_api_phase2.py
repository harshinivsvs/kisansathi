"""
API Integration tests for Phase 2: Multilingual queries, speech voice notes, and profile memory.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_telugu_query_advisory(client):
    payload = {
        "query": "ఉల్లిలో తామర పురుగుల నివారణకు ఏమి పిచికారీ చేయాలి?",
        "phone_number": "+919876500001",
        "return_audio": True
    }
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["detected_language"] == "te"
    assert data["is_fallback"] is False
    assert "తామర" in data["answer"] or "ఫిప్రోనిల్" in data["answer"]
    assert data["audio_url"] is not None
    assert data["farmer_profile"] is not None
    assert data["farmer_profile"]["phone_number"] == "+919876500001"
    # First turn should ask for name
    assert data["followup_profile_prompt"] is not None
    assert "పేరు" in data["followup_profile_prompt"]


def test_serve_audio_endpoint(client):
    # First generate audio
    payload = {
        "query": "What is the spray for fruit borer in tomato?",
        "phone_number": "+919876500002",
        "return_audio": True
    }
    resp = client.post("/api/v1/query", json=payload)
    data = resp.json()
    assert data["audio_url"] is not None

    # Fetch the audio file
    audio_endpoint = data["audio_url"]
    audio_resp = client.get(audio_endpoint)
    assert audio_resp.status_code == 200
    assert audio_resp.headers["content-type"] == "audio/mpeg"
    assert len(audio_resp.content) > 0


def test_voice_query_simulation(client):
    payload = {
        "phone_number": "+919876500003",
        "language_hint": "en",
        "return_audio": True
    }
    resp = client.post("/api/v1/voice/query", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert "raw_transcript" in data
    assert "corrected_transcript" in data
    assert data["advisory"]["answer"] != ""


def test_get_farmer_profile_endpoint(client):
    phone = "+919876500001"
    resp = client.get(f"/api/v1/farmer/{phone}")
    assert resp.status_code == 200
    profile = resp.json()
    assert profile["phone_number"] == phone
    assert profile["preferred_language"] == "te"
