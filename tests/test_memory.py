"""
Unit tests for farmer profiles, SQLite memory, and sequential missing profile inquiries.
"""

from pathlib import Path
import pytest
from backend.memory.memory_manager import MemoryManager


@pytest.fixture
def memory_manager(tmp_path):
    test_db = tmp_path / "test_memory.db"
    return MemoryManager(db_path=test_db)


def test_farmer_creation_and_retrieval(memory_manager):
    phone = "+919876543210"
    farmer = memory_manager.get_or_create_farmer(phone, default_lang="te")

    assert farmer.phone_number == phone
    assert farmer.preferred_language == "te"
    assert farmer.name is None

    # Retrieve again
    retrieved = memory_manager.get_or_create_farmer(phone)
    assert retrieved.phone_number == phone


def test_sequential_missing_profile_prompts(memory_manager):
    phone = "+919999888877"
    farmer = memory_manager.get_or_create_farmer(phone, default_lang="te")

    # Step 1: Missing Name
    p1 = memory_manager.get_next_missing_profile_prompt(farmer, language="te")
    assert "పేరు" in p1  # Asks for Name in Telugu

    # Update name
    farmer.name = "రాములు"
    memory_manager.update_farmer(farmer)

    # Step 2: Missing Village
    p2 = memory_manager.get_next_missing_profile_prompt(farmer, language="te")
    assert "గ్రామం" in p2 or "జిల్లా" in p2

    # Update village
    farmer.village = "వరంగల్"
    memory_manager.update_farmer(farmer)

    # Step 3: Missing Crop
    p3 = memory_manager.get_next_missing_profile_prompt(farmer, language="te")
    assert "పంట" in p3

    # Update crop
    farmer.primary_crop = "cotton"
    memory_manager.update_farmer(farmer)

    # Step 4: Missing Sowing Date
    p4 = memory_manager.get_next_missing_profile_prompt(farmer, language="te")
    assert "విత్తారు" in p4

    # Update sowing date
    farmer.sowing_date = "June 20"
    memory_manager.update_farmer(farmer)

    # All profile details filled
    p5 = memory_manager.get_next_missing_profile_prompt(farmer, language="te")
    assert p5 is None


def test_extract_profile_updates(memory_manager):
    phone = "+919112233445"
    farmer = memory_manager.get_or_create_farmer(phone, default_lang="te")

    # In Telugu: "నా పేరు రమేష్, నేను పత్తి సాగు చేస్తున్నాను"
    updated = memory_manager.extract_profile_updates("నా పేరు రమేష్, నేను పత్తి సాగు చేస్తున్నాను", farmer)
    assert updated is True
    assert farmer.name == "రమేష్"
    assert farmer.primary_crop == "cotton"


def test_dialog_history_logging(memory_manager):
    phone = "+919123456789"
    memory_manager.add_turn(
        phone_number=phone,
        session_id="test-session",
        user_query="What is the spray for thrips?",
        assistant_answer="Use Fipronil 5% SC.",
        language="en",
        crop_context="onion"
    )

    history = memory_manager.get_recent_history(phone, limit=2)
    assert len(history) == 1
    assert history[0]["user_query"] == "What is the spray for thrips?"
    assert history[0]["crop_context"] == "onion"
