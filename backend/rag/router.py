"""
AI Intent & Tool Router for KisanSathi.
Classifies query intent (weather, mandi prices, price watchlist, RAG advisory),
extracts agronomic parameters, applies location fallbacks from profile, and delivers
personalized, unified responses in the farmer's language.
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from backend.config import settings
from backend.memory.database import FarmerProfile
from backend.memory.memory_manager import MemoryManager
from backend.rag.multilingual import LanguageDetector, MultilingualAgriThesaurus
from backend.tools.location import LocationManager, GeoLocation
from backend.tools.weather import WeatherTool, WeatherReport
from backend.tools.mandi import MandiTool, MandiQueryResult
from backend.rag.retriever import RAGRetriever
from backend.rag.generator import RAGGenerator, AdvisoryAnswer


class RoutedAction(BaseModel):
    """Router decision with detected intent and extracted parameters."""
    intent: str  # "weather", "mandi", "watchlist", "rag", "greeting"
    crop: Optional[str] = None
    target_place: Optional[str] = None
    target_price: Optional[float] = None
    watchlist_condition: Optional[str] = None
    resolved_location: GeoLocation
    personalized_prefix: str = ""
    is_profile_location_used: bool = True


class ToolRouter:
    """Intelligently routes farmer inquiries to tools or RAG knowledge base."""

    def __init__(
        self,
        weather_tool: Optional[WeatherTool] = None,
        mandi_tool: Optional[MandiTool] = None,
        retriever: Optional[RAGRetriever] = None,
        generator: Optional[RAGGenerator] = None,
        memory_manager: Optional[MemoryManager] = None
    ):
        self.weather_tool = weather_tool or WeatherTool()
        self.mandi_tool = mandi_tool or MandiTool()
        self.memory_manager = memory_manager or MemoryManager()
        self.retriever = retriever
        self.generator = generator or RAGGenerator()

    def classify_and_extract(self, query: str, farmer: FarmerProfile) -> RoutedAction:
        """Determines intent and extracts parameters using semantic regex and entity parsing."""
        q_lower = query.lower()

        # 1. Detect Crop (from query, fallback to profile crop)
        detected_crop = MultilingualAgriThesaurus.detect_crop_multilingual(query)
        crop = detected_crop or farmer.primary_crop or "onion"

        # 2. Detect Specific Place mentioned in query
        explicit_place = None
        for alias, canon in LocationManager.MULTILINGUAL_PLACE_ALIASES.items():
            if alias.lower() in query.lower():
                explicit_place = canon
                break
        if not explicit_place:
            for town in LocationManager.AGRI_TOWN_COORDINATES.keys():
                if town in q_lower:
                    explicit_place = town.title()
                    break
        if not explicit_place:
            for dist in LocationManager.DISTRICT_CENTROIDS.keys():
                if dist in q_lower:
                    explicit_place = dist.title()
                    break

        is_profile_used = False
        if explicit_place:
            resolved_geo = LocationManager.resolve_location(explicit_place, default_district=farmer.district or "Nashik")
        else:
            is_profile_used = True
            loc_label = farmer.village or farmer.district or "Nashik"
            resolved_geo = GeoLocation(
                latitude=farmer.latitude or 19.9975,
                longitude=farmer.longitude or 73.7898,
                village=farmer.village,
                district=farmer.district or "Nashik",
                state=farmer.state or "Maharashtra",
                source="profile",
                consent_given=farmer.location_consent_given
            )

        # 3. Detect Watchlist Setup (e.g. "crosses 2000", "2500 దాటినప్పుడు", "2000 पार होते ही", "alert me when")
        watchlist_match = re.search(
            r"(?:(?:crosses|cross|reaches|above|alert|अलर्ट|హెచ్చరిక)\D*?(\d{3,5})|(\d{3,5})\s*(?:దాటితే|దాటినప్పుడు|पार|above|cross))",
            query,
            re.IGNORECASE
        )
        if watchlist_match or any(w in q_lower for w in ["watchlist", "alert me when", "హెచ్చరిక"]):
            price = float(watchlist_match.group(1) or watchlist_match.group(2)) if watchlist_match else 2000.0
            return RoutedAction(
                intent="watchlist",
                crop=crop,
                target_place=explicit_place,
                target_price=price,
                watchlist_condition="above",
                resolved_location=resolved_geo,
                is_profile_location_used=is_profile_used
            )

        # 4. Detect Weather Intent
        weather_keywords = [
            "weather", "rain", "temperature", "forecast", "wind", "humidity", "spray weather",
            "shower", "showers", "cloud", "precipitation",
            "వర్ష", "వాతావరణం", "ఎండ", "గాలి", "వాన", "ఉష్ణోగ్రత", "వర్షపాతం",
            "मौसम", "बारिश", "तापमान", "हवा", "बौछार"
        ]
        if any(w in q_lower for w in weather_keywords):
            return RoutedAction(
                intent="weather",
                crop=crop,
                target_place=explicit_place,
                resolved_location=resolved_geo,
                is_profile_location_used=is_profile_used
            )

        # 5. Detect Mandi Price Intent
        # Filter out false-positive substrings like संभावना (probability) and agronomic rates (seed rate, spray rate)
        is_agronomic_rate = any(agr in q_lower for agr in ["seed rate", "spray rate", "application rate", "germination rate", "mortality rate"])
        clean_for_mandi = q_lower.replace("संभावना", "").replace("సంభావ్యత", "")
        if is_agronomic_rate:
            clean_for_mandi = clean_for_mandi.replace("rate", "").replace("rates", "")

        mandi_keywords = [
            "mandi", "bhav", "price", "rate", "rates", "market", "quintal",
            "ధర", "మార్కెట్", "భావం", "రేటు", "ధరలు", "క్వింటాల్",
            "भाव", "दाम", "मंडी भाव", "बाजार", "मंडी", "मंडियों", "रेट"
        ]
        if any(w in clean_for_mandi for w in mandi_keywords):
            return RoutedAction(
                intent="mandi",
                crop=crop,
                target_place=explicit_place,
                resolved_location=resolved_geo,
                is_profile_location_used=is_profile_used
            )

        # 6. Default: Agronomic RAG Advisory
        return RoutedAction(
            intent="rag",
            crop=crop,
            target_place=explicit_place,
            resolved_location=resolved_geo,
            is_profile_location_used=is_profile_used
        )

    def route_and_execute(
        self,
        query: str,
        phone_number: str,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes the routed tool, composes personal framing, and produces the unified response.
        """
        farmer = self.memory_manager.get_or_create_farmer(phone_number)
        action = self.classify_and_extract(query, farmer)
        lang = language or farmer.preferred_language or LanguageDetector.detect_language(query)
        if lang not in ("en", "hi", "te"):
            lang = settings.SECONDARY_LANGUAGE

        loc_label = action.target_place or farmer.village or farmer.district or "Nashik"

        # Personalization prefix incorporating crop and location
        crop_name = action.crop.title()
        personal_prefix = ""
        if action.is_profile_location_used and farmer.village:
            if lang == "te":
                personal_prefix = f"{farmer.name or 'రైతు సోదరా'}, {loc_label}లోని మీ {crop_name} పంటకు సంబంధించి:\n"
            elif lang == "hi":
                personal_prefix = f"{farmer.name or 'किसान भाई'}, {loc_label} में आपकी {crop_name} फसल के लिए:\n"
            else:
                personal_prefix = f"For your {crop_name} in {loc_label}:\n"

        # Intent 1: Weather
        if action.intent == "weather":
            w_report = self.weather_tool.get_weather(
                latitude=action.resolved_location.latitude,
                longitude=action.resolved_location.longitude,
                location_name=loc_label
            )

            if lang == "te":
                body = f"🌦️ వాతావరణ సమాచారం ({loc_label}): {w_report.agronomic_summary_te}"
            elif lang == "hi":
                body = f"🌦️ मौसम पूर्वानुमान ({loc_label}): {w_report.agronomic_summary_hi}"
            else:
                body = f"🌦️ Weather Advisory ({loc_label}): {w_report.agronomic_summary}"

            full_answer = f"{personal_prefix}{body}".strip()
            return {
                "intent": "weather",
                "answer": full_answer,
                "citations": ["IMD / OpenWeather Local Station Observation"],
                "confidence": 0.95,
                "tool_data": w_report.model_dump(),
                "action": action
            }

        # Intent 2: Mandi Price
        elif action.intent == "mandi":
            m_report = self.mandi_tool.query_prices(
                crop=action.crop,
                user_lat=action.resolved_location.latitude,
                user_lon=action.resolved_location.longitude,
                specific_mandi_name=action.target_place,
                location_label=loc_label
            )

            if lang == "te":
                body = f"💰 సమీప మార్కెట్ ధరలు ({loc_label}): {m_report.summary_te}"
            elif lang == "hi":
                body = f"💰 नजदीकी मंडी भाव ({loc_label}): {m_report.summary_hi}"
            else:
                body = f"💰 Nearest Mandi Prices ({loc_label}): {m_report.summary_en}"

            full_answer = f"{personal_prefix}{body}".strip()
            citations = [f"Agmarknet / {m.mandi_name} ({m.report_date})" for m in m_report.prices[:2]]

            return {
                "intent": "mandi",
                "answer": full_answer,
                "citations": citations if citations else ["Agmarknet Mandi Reports"],
                "confidence": 0.94,
                "tool_data": m_report.model_dump(),
                "action": action
            }

        # Intent 3: Watchlist Setup
        elif action.intent == "watchlist":
            target = action.target_price or 2000.0
            wl = self.memory_manager.create_price_watchlist(
                phone_number=phone_number,
                crop=action.crop,
                target_price=target,
                condition=action.watchlist_condition or "above"
            )

            if lang == "te":
                ans = f"✅ ధర హెచ్చరిక నమోదైంది! {loc_label} సమీప మార్కెట్లలో {action.crop.title()} ధర ₹{target:.0f}/క్వింటాల్ దాటినప్పుడు మీకు వాట్సాప్ సందేశం పంపబడుతుంది."
            elif lang == "hi":
                ans = f"✅ भाव अलर्ट सेट हो गया है! {loc_label} के नजदीकी बाजारों में {action.crop.title()} का भाव ₹{target:.0f}/क्विंटल पार होने पर आपको तुरंत सूचित किया जाएगा।"
            else:
                ans = f"✅ Price alert set! I will notify you as soon as {action.crop.title()} crosses ₹{target:.0f}/q in mandis near {loc_label}."

            return {
                "intent": "watchlist",
                "answer": ans,
                "citations": ["KisanSathi Price Watchlist Service"],
                "confidence": 1.0,
                "tool_data": wl.model_dump(),
                "action": action
            }

        # Intent 4: RAG Agronomic Advisory
        else:
            if not self.retriever:
                from backend.rag.embeddings import get_embedder
                from backend.rag.vector_store import get_vector_store
                self.retriever = RAGRetriever(
                    embedder=get_embedder(settings.EMBEDDING_PROVIDER),
                    vector_store=get_vector_store(settings.VECTOR_STORE_BACKEND)
                )

            ret_res = self.retriever.retrieve(query=query, crop=action.crop)
            adv_ans = self.generator.generate_answer(ret_res, target_language=lang)

            framed_answer = adv_ans.answer
            if personal_prefix and not adv_ans.is_fallback:
                framed_answer = f"{personal_prefix}{adv_ans.answer}"

            return {
                "intent": "rag",
                "answer": framed_answer,
                "citations": adv_ans.citations,
                "confidence": adv_ans.confidence,
                "tool_data": adv_ans.model_dump(),
                "action": action
            }
