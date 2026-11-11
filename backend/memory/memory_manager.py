"""
Memory manager for KisanSathi conversational state and profile tracking.
Stores farmer profile attributes and asks for missing profile details one at a time.
"""

import json
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional

from backend.config import settings
from backend.memory.database import DB_PATH, FarmerProfile, init_db
from backend.rag.multilingual import MultilingualAgriThesaurus, LanguageDetector


class MemoryManager:
    """Manages farmer profiles, conversational context windows, and telemetry logging."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        init_db(self.db_path)

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_or_create_farmer(self, phone_number: str, default_lang: Optional[str] = None) -> FarmerProfile:
        """Retrieves existing farmer profile or registers a new profile."""
        clean_phone = phone_number.strip()
        lang = default_lang or settings.SECONDARY_LANGUAGE  # default: te (Telugu)

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM farmers WHERE phone_number = ?", (clean_phone,))
        row = cursor.fetchone()

        if row:
            profile = FarmerProfile(
                phone_number=row["phone_number"],
                name=row["name"],
                village=row["village"],
                district=row["district"],
                state=row["state"],
                primary_crop=row["primary_crop"],
                sowing_date=row["sowing_date"],
                preferred_language=row["preferred_language"],
                created_at=row["created_at"],
                updated_at=row["updated_at"]
            )
            conn.close()
            return profile

        now = time.time()
        cursor.execute("""
            INSERT INTO farmers (phone_number, preferred_language, created_at, updated_at)
            VALUES (?, ?, ?, ?)
        """, (clean_phone, lang, now, now))
        conn.commit()
        conn.close()

        return FarmerProfile(
            phone_number=clean_phone,
            preferred_language=lang,
            created_at=now,
            updated_at=now
        )

    def update_farmer(self, profile: FarmerProfile) -> None:
        """Saves updated profile attributes to SQLite."""
        profile.updated_at = time.time()
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE farmers
            SET name = ?, village = ?, district = ?, state = ?,
                primary_crop = ?, sowing_date = ?, preferred_language = ?, updated_at = ?
            WHERE phone_number = ?
        """, (
            profile.name,
            profile.village,
            profile.district,
            profile.state,
            profile.primary_crop,
            profile.sowing_date,
            profile.preferred_language,
            profile.updated_at,
            profile.phone_number
        ))
        conn.commit()
        conn.close()

    def extract_profile_updates(self, text: str, profile: FarmerProfile) -> bool:
        """
        Extracts farmer profile details from conversational input.
        Returns True if any attribute was updated.
        """
        updated = False
        t_lower = text.lower()

        # 1. Name extraction
        name_patterns = [
            r"(?:my name is|i am|call me)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            r"(?:मेरा नाम|नाम है)\s+([^\s,]+)",
            r"(?:నా పేరు|పేరు)\s+([^\s,]+)"
        ]
        if not profile.name:
            for pat in name_patterns:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    extracted = m.group(1).strip()
                    if len(extracted) > 2 and extracted.lower() not in ("kisan", "farmer", "hello", "namaste"):
                        profile.name = extracted.title()
                        updated = True
                        break

        # 2. Village / Location extraction
        village_patterns = [
            r"(?:from|village|district|in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            r"(?:गांव|गाँव|जिले से|रहता हूँ)\s+([^\s,]+)",
            r"([^\s,]+)\s+నుండి",
            r"(?:గ్రామం|జిల్లా|మా ఊరు)\s+([^\s,]+)"
        ]
        if not profile.village:
            for pat in village_patterns:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    extracted = m.group(1).strip()
                    if len(extracted) > 2 and extracted.lower() not in ("farm", "field", "here", "india"):
                        profile.village = extracted.title()
                        updated = True
                        break

        # 3. Crop extraction
        detected_crop = MultilingualAgriThesaurus.detect_crop_multilingual(text)
        if detected_crop and (not profile.primary_crop or profile.primary_crop != detected_crop):
            profile.primary_crop = detected_crop
            updated = True

        # 4. Sowing date / period extraction
        sowing_patterns = [
            r"(?:sowed|sown|planted)\s+(?:on|in)?\s*([a-zA-Z0-9\s,-]+(?:days ago|weeks ago|june|july|august|october|november))",
            r"(?:बुवाई|बोया)\s*(?:था)?\s*([^\n,]+)",
            r"(?:విత్తనం\s*వేసాను|నాటాను|విత్తిన తేదీ)\s*([^\n,]+)",
            r"([0-9]+\s*రోజుల\s*క్రితం[^\n,]*)"
        ]
        if not profile.sowing_date:
            for pat in sowing_patterns:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    profile.sowing_date = m.group(1).strip()
                    updated = True
                    break

        if updated:
            self.update_farmer(profile)

        return updated

    def get_next_missing_profile_prompt(self, profile: FarmerProfile, language: str = "te") -> Optional[str]:
        """
        Returns a polite inquiry for the next missing farmer detail (one at a time).
        Follows priority sequence: Name -> Village -> Crop -> Sowing Date.
        """
        lang = language.lower()
        if lang not in ("en", "hi", "te"):
            lang = profile.preferred_language or "te"

        # Sequential inquiry prompts
        prompts = {
            "name": {
                "en": "By the way, may I know your name so I can personalize your advisories?",
                "hi": "वैसे, क्या मैं आपका नाम जान सकता हूँ ताकि आपको बेहतर सलाह दे सकूँ?",
                "te": "అలాగే, మీకు వ్యక్తిగతీకరించిన సలహాలు అందించడానికి మీ పేరు తెలుసుకోవచ్చా?"
            },
            "village": {
                "en": "Which village or district is your farm located in?",
                "hi": "आपका खेत किस गांव या जिले में स्थित है?",
                "te": "మీ వ్యవసాయ పొలం ఏ గ్రామం లేదా జిల్లాలో ఉంది?"
            },
            "crop": {
                "en": "Which primary crop are you cultivating this season (Tomato, Onion, Soybean, Cotton)?",
                "hi": "आप इस मौसम में कौन सी मुख्य फसल उगा रहे हैं (टमाटर, प्याज, सोयाबीन, कपास)?",
                "te": "ఈ సీజన్‌లో మీరు ఏ ప్రధాన పంటను సాగు చేస్తున్నారు (టమాటా, ఉల్లిపాయ, సోయాబీన్, పత్తి)?"
            },
            "sowing_date": {
                "en": "Approximately when did you sow your crop?",
                "hi": "आपने अपनी फसल की बुवाई लगभग किस तारीख को की थी?",
                "te": "మీరు పంటను దాదాపు ఎప్పుడు విత్తారు లేదా నాటారు?"
            }
        }

        if not profile.name:
            return prompts["name"].get(lang, prompts["name"]["en"])
        if not profile.village:
            return prompts["village"].get(lang, prompts["village"]["en"])
        if not profile.primary_crop:
            return prompts["crop"].get(lang, prompts["crop"]["en"])
        if not profile.sowing_date:
            return prompts["sowing_date"].get(lang, prompts["sowing_date"]["en"])

        return None

    def add_turn(
        self,
        phone_number: str,
        session_id: str,
        user_query: str,
        assistant_answer: str,
        language: str,
        crop_context: Optional[str] = None
    ) -> None:
        """Stores a conversational exchange in history."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO conversations (phone_number, session_id, user_query, assistant_answer, language, crop_context, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (phone_number, session_id, user_query, assistant_answer, language, crop_context or "", time.time()))
        conn.commit()
        conn.close()

    def get_recent_history(self, phone_number: str, limit: int = 4) -> List[Dict[str, Any]]:
        """Retrieves the recent conversational turns for the farmer."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT user_query, assistant_answer, language, crop_context, timestamp
            FROM conversations
            WHERE phone_number = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (phone_number, limit))
        rows = cursor.fetchall()
        conn.close()

        history = []
        for r in reversed(rows):
            history.append({
                "user_query": r["user_query"],
                "assistant_answer": r["assistant_answer"],
                "language": r["language"],
                "crop_context": r["crop_context"],
                "timestamp": r["timestamp"]
            })
        return history

    def log_query(
        self,
        phone_number: str,
        query: str,
        answer: str,
        retrieved_chunks: List[Any],
        citations: List[str],
        confidence: float,
        latency_ms: float,
        token_cost: float,
        audio_url: str = "",
        language: str = "te"
    ) -> None:
        """Logs comprehensive telemetry for every advisory query."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO query_logs (
                phone_number, query, answer, retrieved_chunks_json, citations_json,
                confidence, latency_ms, token_cost, audio_url, language, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            phone_number,
            query,
            answer,
            json.dumps(retrieved_chunks),
            json.dumps(citations),
            confidence,
            latency_ms,
            token_cost,
            audio_url,
            language,
            time.time()
        ))
        conn.commit()
        conn.close()
