"""
KisanSathi Speech Module (STT with custom agricultural vocabulary, and TTS voice notes).
"""
from backend.speech.vocabulary import AgriVocabularyCorrector
from backend.speech.stt import SpeechToTextService
from backend.speech.tts import TextToSpeechService

__all__ = [
    "AgriVocabularyCorrector",
    "SpeechToTextService",
    "TextToSpeechService",
]
