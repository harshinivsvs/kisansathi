"""
Unit tests for Speech module: vocabulary correction, STT, and TTS voice notes.
"""

from pathlib import Path
from backend.speech.vocabulary import AgriVocabularyCorrector
from backend.speech.stt import SpeechToTextService
from backend.speech.tts import TextToSpeechService


def test_vocabulary_corrections():
    raw_text = "how to control onion tripes with feroni and pink ball worm in patti crop"
    corrected = AgriVocabularyCorrector.correct_transcript(raw_text)

    assert "thrips" in corrected
    assert "Fipronil" in corrected
    assert "pink bollworm" in corrected
    assert "patti" in corrected


def test_tts_synthesis_telugu(tmp_path):
    tts = TextToSpeechService()
    res = tts.synthesize(
        text="ఉల్లిలో తామర పురుగుల నివారణకు ఫిప్రోనిల్ పిచికారీ చేయండి.",
        language="te"
    )

    assert res.language == "te"
    assert res.audio_path.endswith(".mp3")
    assert Path(res.audio_path).exists()
    assert res.file_size_bytes > 0
    assert res.duration_estimate_sec > 1.0


def test_tts_synthesis_hindi(tmp_path):
    tts = TextToSpeechService()
    res = tts.synthesize(
        text="टमाटर में फल छेदक के लिए फेरोमोन ट्रैप लगाएं।",
        language="hi"
    )

    assert res.language == "hi"
    assert Path(res.audio_path).exists()
    assert res.file_size_bytes > 0


def test_stt_mock_transcribe():
    stt = SpeechToTextService()
    res = stt.transcribe_text_mock("What is the spray for onion tripes?", language_hint="en")

    assert res.vocab_corrections_applied is True
    assert "thrips" in res.corrected_transcript
    assert res.detected_language == "en"
