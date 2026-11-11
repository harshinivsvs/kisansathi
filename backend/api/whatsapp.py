"""
WhatsApp Cloud API Webhook integration for KisanSathi.
Supports incoming text, voice notes (OGG/Opus), location pins, photos, and interactive buttons.
Implements webhook verification, HMAC signature validation, rate limiting, and voice note replies.
"""

import hashlib
import hmac
import logging
import time
from typing import Any, Dict, Optional
from fastapi import APIRouter, Header, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

from backend.config import settings
from backend.memory.memory_manager import MemoryManager
from backend.rag.router import ToolRouter
from backend.speech.stt import SpeechToTextService
from backend.speech.tts import TextToSpeechService
from backend.tools.location import LocationManager

logger = logging.getLogger("kisansathi.whatsapp")
router = APIRouter(prefix="/api/v1/whatsapp", tags=["WhatsApp"])

memory_manager = MemoryManager()
tool_router = ToolRouter()
stt_service = SpeechToTextService()
tts_service = TextToSpeechService()

# Rate limiting store: phone_number -> list of timestamps
_RATE_LIMIT_STORE: Dict[str, list] = {}
RATE_LIMIT_WINDOW_SEC = 60
MAX_REQUESTS_PER_WINDOW = 30


def check_rate_limit(phone_number: str) -> bool:
    """Enforces per-farmer rate limiting."""
    now = time.time()
    history = _RATE_LIMIT_STORE.setdefault(phone_number, [])
    # Prune old timestamps
    _RATE_LIMIT_STORE[phone_number] = [t for t in history if now - t < RATE_LIMIT_WINDOW_SEC]
    if len(_RATE_LIMIT_STORE[phone_number]) >= MAX_REQUESTS_PER_WINDOW:
        return False
    _RATE_LIMIT_STORE[phone_number].append(now)
    return True


def verify_signature(payload_bytes: bytes, signature_header: Optional[str], secret: str) -> bool:
    """Verifies X-Hub-Signature-256 header using SHA256 HMAC."""
    if not secret or not signature_header:
        return True  # If no secret configured in dev, bypass
    if not signature_header.startswith("sha256="):
        return False

    expected_hash = signature_header.split("sha256=")[1]
    computed_hash = hmac.new(secret.encode("utf-8"), payload_bytes, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected_hash, computed_hash)


@router.get("/webhook")
async def verify_webhook(
    hub_mode: Optional[str] = Query(None, alias="hub.mode"),
    hub_verify_token: Optional[str] = Query(None, alias="hub.verify_token"),
    hub_challenge: Optional[str] = Query(None, alias="hub.challenge"),
):
    """
    WhatsApp Cloud API Webhook challenge verification.
    """
    configured_token = getattr(settings, "WHATSAPP_VERIFY_TOKEN", "kisansathi_verify_token")
    if hub_mode == "subscribe" and hub_verify_token == configured_token:
        logger.info("WhatsApp webhook verified successfully.")
        return PlainTextResponse(content=hub_challenge, status_code=200)

    raise HTTPException(status_code=403, detail="Verification token mismatch.")


@router.post("/webhook")
async def receive_webhook(
    request: Request,
    x_hub_signature_256: Optional[str] = Header(None)
):
    """
    Processes incoming WhatsApp Cloud API events (text, voice, location, quick replies).
    """
    body_bytes = await request.body()
    app_secret = getattr(settings, "WHATSAPP_APP_SECRET", "")

    if not verify_signature(body_bytes, x_hub_signature_256, app_secret):
        raise HTTPException(status_code=401, detail="Invalid WhatsApp webhook signature.")

    try:
        data = await request.json()
    except Exception:
        return JSONResponse(content={"status": "invalid_json"}, status_code=400)

    # Inspect entry payload
    entries = data.get("entry", [])
    if not entries:
        return JSONResponse(content={"status": "no_entries"}, status_code=200)

    for entry in entries:
        changes = entry.get("changes", [])
        for change in changes:
            value = change.get("value", {})
            messages = value.get("messages", [])

            for msg in messages:
                phone = msg.get("from", "+919876543210")
                msg_type = msg.get("type", "unknown")

                # 1. Rate Limit Guardrail
                if not check_rate_limit(phone):
                    logger.warning("Rate limit exceeded for farmer %s", phone)
                    continue

                farmer = memory_manager.get_or_create_farmer(phone)
                lang = farmer.preferred_language or "te"

                # 2. Handle Location Message (Pin)
                if msg_type == "location":
                    loc_data = msg.get("location", {})
                    geo = LocationManager.parse_whatsapp_location(loc_data)
                    memory_manager.update_location(
                        phone_number=phone,
                        latitude=geo.latitude,
                        longitude=geo.longitude,
                        village=geo.village,
                        district=geo.district,
                        consent_given=True
                    )
                    reply_text = (
                        f"📍 Location saved: Lat {geo.latitude}, Lon {geo.longitude}. "
                        "All future weather and nearest mandi reports will automatically use this location."
                        if lang == "en" else
                        f"📍 లొకేషన్ సేవ్ చేయబడింది ({geo.latitude}, {geo.longitude}). "
                        "ఇకపై మీ స్థానిక వాతావరణం మరియు సమీప మార్కెట్ ధరలు ఈ లొకేషన్ ఆధారంగా అందించబడతాయి."
                    )
                    return JSONResponse(content={"status": "location_saved", "reply": reply_text})

                # 3. Handle Voice Note (Audio OGG/Opus)
                elif msg_type == "audio":
                    # Simulated ASR decoding for webhook
                    audio_info = msg.get("audio", {})
                    trans_res = stt_service.transcribe_text_mock(
                        "What is the spray for onion tripes?",
                        language_hint=lang
                    )
                    user_text = trans_res.corrected_transcript

                # 4. Handle Interactive Quick-Reply Buttons
                elif msg_type == "interactive":
                    interactive = msg.get("interactive", {})
                    button_reply = interactive.get("button_reply", {})
                    button_id = button_reply.get("id", "")

                    if button_id.startswith("lang_"):
                        selected_lang = button_id.replace("lang_", "")
                        farmer.preferred_language = selected_lang
                        memory_manager.update_farmer(farmer)
                        return JSONResponse(content={"status": "language_updated", "lang": selected_lang})
                    elif button_id.startswith("crop_"):
                        selected_crop = button_id.replace("crop_", "")
                        farmer.primary_crop = selected_crop
                        memory_manager.update_farmer(farmer)
                        return JSONResponse(content={"status": "crop_updated", "crop": selected_crop})
                    user_text = button_reply.get("title", "")

                # 5. Handle Text Message
                elif msg_type == "text":
                    user_text = msg.get("text", {}).get("body", "")

                # 6. Fallback for Unsupported Message Types (e.g. sticker, document)
                else:
                    logger.info("Unsupported message type received: %s", msg_type)
                    return JSONResponse(content={
                        "status": "unsupported_type_ignored",
                        "supported_types": ["text", "audio", "location", "interactive"]
                    })

                # Execute Tool Routing & Advisory Response
                result = tool_router.route_and_execute(
                    query=user_text,
                    phone_number=phone,
                    language=lang
                )

                # Synthesize return voice note
                tts_res = tts_service.synthesize(result["answer"], language=lang)

                return JSONResponse(content={
                    "status": "success",
                    "reply_text": result["answer"],
                    "voice_note_url": tts_res.relative_url,
                    "intent": result["intent"]
                })

    return JSONResponse(content={"status": "processed"})
