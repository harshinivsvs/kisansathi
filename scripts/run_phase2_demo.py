"""
Phase 2 Interactive Demonstration CLI for KisanSathi.
Demonstrates:
1. Multilingual advisory in Telugu (te) and Hindi (hi)
2. Speech transcription with custom agricultural vocabulary post-correction
3. Audio voice note synthesis (MP3)
4. Persistent conversational farmer memory & profile onboarding (asking one detail at a time)
5. Localized safety guardrails in Telugu
"""

from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from backend.config import settings
from backend.rag.document_loader import DocumentLoader
from backend.rag.chunking import ChunkerFactory
from backend.rag.embeddings import get_embedder
from backend.rag.vector_store import get_vector_store
from backend.rag.retriever import RAGRetriever
from backend.rag.generator import RAGGenerator
from backend.rag.multilingual import LanguageDetector
from backend.speech.vocabulary import AgriVocabularyCorrector
from backend.speech.stt import SpeechToTextService
from backend.speech.tts import TextToSpeechService
from backend.memory.memory_manager import MemoryManager


def run_phase2_demo():
    print("==================================================")
    print("   KisanSathi Phase 2 Multilingual & Voice Demo   ")
    print("        Configured Regional Language: TELUGU      ")
    print("==================================================\n")

    # 1. Initialize RAG Knowledge Base
    raw_dir = PROJECT_ROOT / "data" / "raw"
    docs = DocumentLoader.load_directory(raw_dir)
    chunker = ChunkerFactory.get_chunker("semantic")
    all_chunks = []
    for d in docs:
        all_chunks.extend(chunker.chunk_document(d))

    embedder = get_embedder("mock")
    embeddings = embedder.embed_documents([c.text for c in all_chunks])
    store = get_vector_store("inmemory")
    store.clear()
    store.add_chunks(all_chunks, embeddings)

    retriever = RAGRetriever(embedder=embedder, vector_store=store, confidence_threshold=0.55)
    generator = RAGGenerator(provider="mock")
    memory = MemoryManager()
    tts = TextToSpeechService()

    farmer_phone = "+919848012345"

    print("--- SCENARIO 1: Voice Input with ASR Vocabulary Correction ---")
    noisy_voice_asr = "What is the spray for onion tripes and purple bloch?"
    print(f"Raw ASR Voice Transcript : {noisy_voice_asr}")
    corrected_transcript = AgriVocabularyCorrector.correct_transcript(noisy_voice_asr)
    print(f"Corrected ASR Transcript  : {corrected_transcript}\n")

    print("--- SCENARIO 2: Multilingual Telugu Query & Conversational Memory ---")
    farmer = memory.get_or_create_farmer(farmer_phone, default_lang="te")

    # Turn 1: Farmer introduces name and location while asking about cotton pest
    turn1_query = "నా పేరు రమేష్, వరంగల్ నుండి. పత్తిలో గులాబీ రంగు పురుగు నివారణ ఏమిటి?"
    print(f"Farmer Query (Turn 1)    : {turn1_query}")

    # Extract profile attributes
    memory.extract_profile_updates(turn1_query, farmer)
    print(f"Detected Language        : {LanguageDetector.detect_language(turn1_query)}")
    print(f"Updated Profile          : Name='{farmer.name}', Village/Dist='{farmer.village}', Crop='{farmer.primary_crop}'")

    ret1 = retriever.retrieve(query=turn1_query, crop=farmer.primary_crop)
    ans1 = generator.generate_answer(ret1, target_language="te")
    missing_prompt1 = memory.get_next_missing_profile_prompt(farmer, language="te")

    print(f"\nAdvisory Answer (Telugu) :\n{ans1.answer}")
    print(f"\nFollow-up Inquiry (Memory): {missing_prompt1}")

    # Generate Telugu Audio Voice Note
    audio_text1 = f"{ans1.answer} {missing_prompt1}"
    tts_res1 = tts.synthesize(audio_text1, language="te")
    print(f"Synthesized Voice Note   : {tts_res1.relative_url} ({tts_res1.duration_estimate_sec} sec, {tts_res1.file_size_bytes} bytes)\n")

    # Turn 2: Farmer supplies sowing date
    print("--- SCENARIO 3: Multi-Turn Context Continuity ---")
    turn2_query = "నేను 20 రోజుల క్రితం జూన్ లో విత్తనం వేసాను"
    print(f"Farmer Query (Turn 2)    : {turn2_query}")
    memory.extract_profile_updates(turn2_query, farmer)
    print(f"Profile Status           : Name='{farmer.name}', Crop='{farmer.primary_crop}', Sowing='{farmer.sowing_date}'")
    missing_prompt2 = memory.get_next_missing_profile_prompt(farmer, language="te")
    print(f"Next Missing Detail      : {missing_prompt2 if missing_prompt2 else 'None (Profile Complete!)'}\n")

    print("--- SCENARIO 4: Localized Telugu Safety Guardrails ---")
    harmful_query = "పురుగుమందును మాస్క్ లేకుండా పిచికారీ చేయవచ్చా?"
    print(f"Harmful Query            : {harmful_query}")
    ret_hazard = retriever.retrieve(harmful_query)
    ans_hazard = generator.generate_answer(ret_hazard, target_language="te")
    print(f"Safety Refusal (Telugu)  : {ans_hazard.answer}\n")

    print("==================================================")
    print("           Phase 2 Demonstration Complete         ")
    print("==================================================")


if __name__ == "__main__":
    run_phase2_demo()
