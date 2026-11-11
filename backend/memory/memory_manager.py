"""
Memory manager for KisanSathi conversational state, location-aware profiles,
price watchlists, and personalized agricultural digests.
"""

from datetime import datetime
import json
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple

from backend.config import settings
from backend.memory.database import DB_PATH, FarmerProfile, PriceWatchlist, init_db
from backend.rag.multilingual import MultilingualAgriThesaurus, LanguageDetector
from backend.tools.location import LocationManager, GeoLocation


class MemoryManager:
    """Manages location-aware farmer profiles, watchlists, and contextual prompts."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        init_db(self.db_path)

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_or_create_farmer(self, phone_number: str, default_lang: Optional[str] = None) -> FarmerProfile:
        """Retrieves existing profile or creates a new one with location defaults."""
        clean_phone = phone_number.strip()
        lang = default_lang or settings.SECONDARY_LANGUAGE  # default: te (Telugu)

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM farmers WHERE phone_number = ?", (clean_phone,))
        row = cursor.fetchone()

        if row:
            keys = row.keys()
            profile = FarmerProfile(
                phone_number=row["phone_number"],
                name=row["name"],
                village=row["village"],
                taluka=row["taluka"] if "taluka" in keys else None,
                district=row["district"] if "district" in keys and row["district"] else "Nashik",
                state=row["state"] if "state" in keys and row["state"] else "Maharashtra",
                latitude=float(row["latitude"]) if "latitude" in keys and row["latitude"] is not None else 19.9975,
                longitude=float(row["longitude"]) if "longitude" in keys and row["longitude"] is not None else 73.7898,
                location_consent_given=bool(row["location_consent_given"]) if "location_consent_given" in keys else False,
                primary_crop=row["primary_crop"],
                sowing_date=row["sowing_date"],
                land_size_acres=float(row["land_size_acres"]) if "land_size_acres" in keys and row["land_size_acres"] is not None else None,
                preferred_language=(row["preferred_language"] if "preferred_language" in keys else None) or lang,
                created_at=row["created_at"],
                updated_at=row["updated_at"]
            )
            conn.close()
            return profile

        now = time.time()
        cursor.execute("""
            INSERT INTO farmers (
                phone_number, district, latitude, longitude,
                location_consent_given, preferred_language, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (clean_phone, "Nashik", 19.9975, 73.7898, 0, lang, now, now))
        conn.commit()
        conn.close()

        return FarmerProfile(
            phone_number=clean_phone,
            district="Nashik",
            latitude=19.9975,
            longitude=73.7898,
            location_consent_given=False,
            preferred_language=lang,
            created_at=now,
            updated_at=now
        )

    def update_farmer(self, profile: FarmerProfile) -> None:
        """Persists profile state."""
        profile.updated_at = time.time()
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE farmers
            SET name = ?, village = ?, taluka = ?, district = ?, state = ?,
                latitude = ?, longitude = ?, location_consent_given = ?,
                primary_crop = ?, sowing_date = ?, land_size_acres = ?,
                preferred_language = ?, updated_at = ?
            WHERE phone_number = ?
        """, (
            profile.name,
            profile.village,
            profile.taluka,
            profile.district,
            profile.state,
            profile.latitude,
            profile.longitude,
            1 if profile.location_consent_given else 0,
            profile.primary_crop,
            profile.sowing_date,
            profile.land_size_acres,
            profile.preferred_language,
            profile.updated_at,
            profile.phone_number
        ))
        conn.commit()
        conn.close()

    def update_location(
        self,
        phone_number: str,
        latitude: float,
        longitude: float,
        village: Optional[str] = None,
        district: Optional[str] = None,
        taluka: Optional[str] = None,
        consent_given: bool = True
    ) -> FarmerProfile:
        """Directly updates farmer's location coordinates and consent status."""
        profile = self.get_or_create_farmer(phone_number)
        profile.latitude = latitude
        profile.longitude = longitude
        if village:
            profile.village = village
        if district:
            profile.district = district
        if taluka:
            profile.taluka = taluka
        profile.location_consent_given = consent_given
        self.update_farmer(profile)
        return profile

    def delete_location(self, phone_number: str) -> None:
        """Removes stored location upon farmer request."""
        profile = self.get_or_create_farmer(phone_number)
        profile.latitude = 19.9975
        profile.longitude = 73.7898
        profile.village = None
        profile.taluka = None
        profile.district = "Nashik"
        profile.location_consent_given = False
        self.update_farmer(profile)

    def extract_profile_updates(self, text: str, profile: FarmerProfile) -> bool:
        """
        Extracts farmer profile details, location pin/names, land size, and crop from input.
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

        # 2. Location / Village / District extraction
        village_patterns = [
            r"(?:from|village|district|in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            r"(?:गांव|गाँव|जिले से|रहता हूँ)\s+([^\s,]+)",
            r"([^\s,]+)\s+నుండి",
            r"(?:గ్రామం|జిల్లా|మా ఊరు)\s+([^\s,]+)"
        ]
        for pat in village_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                extracted = m.group(1).strip()
                if len(extracted) > 2 and extracted.lower() not in ("farm", "field", "here", "india"):
                    profile.village = extracted.title()
                    geo = LocationManager.resolve_location(extracted, default_district=profile.district or "Nashik")
                    profile.latitude = geo.latitude
                    profile.longitude = geo.longitude
                    if geo.district:
                        profile.district = geo.district
                    profile.location_consent_given = True
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

        # 5. Land size extraction (e.g. 5 acres, 2 hec)
        land_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:acres?|एकड़|ఎకరాలు|hec|hectares?)", text, re.IGNORECASE)
        if land_match and not profile.land_size_acres:
            try:
                profile.land_size_acres = float(land_match.group(1))
                updated = True
            except Exception:
                pass

        if updated:
            self.update_farmer(profile)

        return updated

    def get_next_missing_profile_prompt(self, profile: FarmerProfile, language: str = "te") -> Optional[str]:
        """
        Sequentially queries missing farmer details one by one:
        Name -> Location / Consent -> Crop -> Sowing Date.
        """
        lang = language.lower()
        if lang not in ("en", "hi", "te"):
            lang = profile.preferred_language or "te"

        prompts = {
            "name": {
                "en": "By the way, may I know your name so I can personalize your advisories?",
                "hi": "वैसे, क्या मैं आपका नाम जान सकता हूँ ताकि आपको बेहतर सलाह दे सकूँ?",
                "te": "అలాగే, మీకు వ్యక్తిగతీకరించిన సలహాలు అందించడానికి మీ పేరు తెలుసుకోవచ్చా?"
            },
            "location": {
                "en": "To provide accurate local weather and nearest mandi prices, may I know your village or district? (You can share your WhatsApp location pin)",
                "hi": "सटीक स्थानीय मौसम और नजदीकी मंडी भाव के लिए, क्या आप अपना गांव या जिला बताएंगे? (आप व्हाट्सएप लोकेशन पिन भी साझा कर सकते हैं)",
                "te": "ఖచ్చితమైన స్థానిక వాతావరణం మరియు సమీప మార్కెట్ ధరలను అందించడానికి, మీ గ్రామం లేదా జిల్లా పేరు తెలుపగలరా? (లేదా వాట్సాప్ లొకేషన్ పిన్ పంపవచ్చు)"
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
            return prompts["location"].get(lang, prompts["location"]["en"])
        if not profile.primary_crop:
            return prompts["crop"].get(lang, prompts["crop"]["en"])
        if not profile.sowing_date:
            return prompts["sowing_date"].get(lang, prompts["sowing_date"]["en"])

        return None

    # Watchlist Management
    def create_price_watchlist(
        self,
        phone_number: str,
        crop: str,
        target_price: float,
        condition: str = "above"
    ) -> PriceWatchlist:
        """Stores a price threshold alert request."""
        conn = self._get_connection()
        cursor = conn.cursor()
        now = time.time()
        cursor.execute("""
            INSERT INTO price_watchlists (phone_number, crop, target_price, condition, active, created_at)
            VALUES (?, ?, ?, ?, 1, ?)
        """, (phone_number, crop.lower().strip(), target_price, condition.lower(), now))
        conn.commit()
        wid = cursor.lastrowid
        conn.close()

        return PriceWatchlist(
            id=wid,
            phone_number=phone_number,
            crop=crop.lower().strip(),
            target_price=target_price,
            condition=condition,
            active=True,
            created_at=now
        )

    def check_watchlists(self, crop: str, current_modal_price: float, mandi_name: str) -> List[Dict[str, Any]]:
        """Identifies triggered price threshold alerts."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, phone_number, crop, target_price, condition
            FROM price_watchlists
            WHERE crop = ? AND active = 1
        """, (crop.lower().strip(),))
        rows = cursor.fetchall()
        conn.close()

        triggered = []
        for r in rows:
            target = r["target_price"]
            cond = r["condition"]
            is_triggered = (cond == "above" and current_modal_price >= target) or (cond == "below" and current_modal_price <= target)

            if is_triggered:
                triggered.append({
                    "watchlist_id": r["id"],
                    "phone_number": r["phone_number"],
                    "crop": crop,
                    "target_price": target,
                    "current_price": current_modal_price,
                    "mandi_name": mandi_name,
                    "message_en": f"Alert: {crop.title()} has reached ₹{current_modal_price:.0f}/q at {mandi_name} (target: ₹{target:.0f}/q).",
                    "message_te": f"ధర హెచ్చరిక: {mandi_name} మార్కెట్‌లో {crop.title()} ధర ₹{current_modal_price:.0f}/క్వింటాల్‌కు చేరింది (లక్ష్యం: ₹{target:.0f}).",
                    "message_hi": f"भाव अलर्ट: {mandi_name} में {crop.title()} का भाव ₹{current_modal_price:.0f}/क्विंटल पहुंच गया है।"
                })

        return triggered

    # Dialog and Logging
    def add_turn(
        self,
        phone_number: str,
        session_id: str,
        user_query: str,
        assistant_answer: str,
        language: str,
        crop_context: Optional[str] = None
    ) -> None:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO conversations (phone_number, session_id, user_query, assistant_answer, language, crop_context, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (phone_number, session_id, user_query, assistant_answer, language, crop_context or "", time.time()))
        conn.commit()
        conn.close()

    def get_recent_history(self, phone_number: str, limit: int = 4) -> List[Dict[str, Any]]:
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

    def generate_personalized_digest(
        self,
        phone_number: str,
        weather_tool: Any,
        mandi_tool: Any
    ) -> Dict[str, Any]:
        """
        Creates a personalized morning briefing combining:
        1. Local weather & rain probability for farmer's location
        2. Nearest mandi prices & net earnings for their crop
        3. Stage-wise advisory tip based on sowing date
        """
        farmer = self.get_or_create_farmer(phone_number)
        lang = farmer.preferred_language or "te"
        crop = farmer.primary_crop or "onion"
        lat = farmer.latitude or 19.9975
        lon = farmer.longitude or 73.7898
        loc_name = farmer.village or farmer.district or "Nashik"

        # 1. Weather
        weather = weather_tool.get_weather(lat, lon, location_name=loc_name)
        # 2. Mandi prices
        mandi = mandi_tool.query_prices(crop=crop, user_lat=lat, user_lon=lon, location_label=loc_name)

        # 3. Crop stage tip
        tip_en = f"Maintain regular field monitoring for {crop.title()} pests and check moisture before next irrigation."
        tip_hi = f"{crop.title()} की फसल में कीटों की नियमित निगरानी करें और अगली सिंचाई से पहले मिट्टी में नमी की जांच करें।"
        tip_te = f"{crop.title()} పంటలో పురుగుల ఉధృతిని గమనించండి మరియు తదుపరి నీటి తడి ఇచ్చే ముందు నేలలో తేమను పరిశీలించండి."

        if lang == "te":
            digest_text = (
                f"🌾 నమస్కారం {farmer.name or 'రైతు సోదరా'}! మీ రోజువారీ వ్యవసాయ సమాచారం ({loc_name}):\n"
                f"🌦️ వాతావరణం: {weather.agronomic_summary_te}\n"
                f"💰 మార్కెట్ ధరలు: {mandi.summary_te}\n"
                f"💡 పంట సలహా: {tip_te}"
            )
        elif lang == "hi":
            digest_text = (
                f"🌾 नमस्ते {farmer.name or 'किसान भाई'}! आपका दैनिक कृषि समाचार ({loc_name}):\n"
                f"🌦️ मौसम: {weather.agronomic_summary_hi}\n"
                f"💰 मंडी भाव: {mandi.summary_hi}\n"
                f"💡 सलाह: {tip_hi}"
            )
        else:
            digest_text = (
                f"🌾 Hello {farmer.name or 'Farmer'}! Your daily agricultural digest for {loc_name}:\n"
                f"🌦️ Weather: {weather.agronomic_summary}\n"
                f"💰 Mandi Prices: {mandi.summary_en}\n"
                f"💡 Crop Tip: {tip_en}"
            )

        return {
            "phone_number": phone_number,
            "crop": crop,
            "location": loc_name,
            "digest_text": digest_text,
            "weather": weather,
            "mandi": mandi
        }
