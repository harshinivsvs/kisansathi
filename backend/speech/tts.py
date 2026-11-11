"""
Text-to-Speech (TTS) service for KisanSathi.
Converts short agronomic advice into audio voice notes for WhatsApp & Web demo
in English (en), Hindi (hi), and Telugu (te).
"""

import hashlib
import os
from pathlib import Path
import time
from typing import Optional
from pydantic import BaseModel

from backend.config import settings

AUDIO_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


class TTSResult(BaseModel):
    """Audio generation metadata."""
    audio_id: str
    audio_path: str
    relative_url: str
    language: str
    duration_estimate_sec: float
    latency_ms: float
    file_size_bytes: int


class TextToSpeechService:
    """Synthesizes advisory text to MP3 audio files using gTTS or mock synthesizer."""

    def __init__(self, provider: Optional[str] = None):
        self.provider = (provider or settings.TTS_PROVIDER).lower()

    def synthesize(self, text: str, language: str = "en") -> TTSResult:
        """
        Converts text into an MP3 voice note.
        Language codes: 'en', 'hi', 'te'.
        """
        start_time = time.perf_counter()
        lang_code = language.lower().strip()
        if lang_code not in ("en", "hi", "te"):
            lang_code = "en"

        # Deterministic audio hash to prevent redundant audio synthesis
        clean_text = text.strip()
        text_hash = hashlib.md5(f"{clean_text}_{lang_code}".encode("utf-8")).hexdigest()[:12]
        filename = f"voice_note_{lang_code}_{text_hash}.mp3"
        target_path = AUDIO_DIR / filename
        relative_url = f"/api/v1/voice/audio/{filename}"

        if target_path.exists():
            file_size = target_path.stat().st_size
            words = len(clean_text.split())
            est_duration = max(1.5, round(words / 2.5, 1))
            latency = (time.perf_counter() - start_time) * 1000
            return TTSResult(
                audio_id=text_hash,
                audio_path=str(target_path),
                relative_url=relative_url,
                language=lang_code,
                duration_estimate_sec=est_duration,
                latency_ms=round(latency, 2),
                file_size_bytes=file_size
            )

        # Generate audio with gTTS (with 3-second timeout for offline safety)
        try:
            from gtts import gTTS
            tts = gTTS(text=clean_text, lang=lang_code, slow=False, timeout=3)
            tts.save(str(target_path))
            file_size = target_path.stat().st_size
        except Exception:
            # Fallback mock audio generation (lightweight valid MP3 header or dummy file)
            target_path.write_bytes(b"\xFF\xFB\x90\x44" + b"\x00" * 2048)
            file_size = target_path.stat().st_size

        words = len(clean_text.split())
        est_duration = max(1.5, round(words / 2.5, 1))
        latency = (time.perf_counter() - start_time) * 1000

        return TTSResult(
            audio_id=text_hash,
            audio_path=str(target_path),
            relative_url=relative_url,
            language=lang_code,
            duration_estimate_sec=est_duration,
            latency_ms=round(latency, 2),
            file_size_bytes=file_size
        )
