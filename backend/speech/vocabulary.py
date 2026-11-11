"""
Agricultural vocabulary post-correction dictionary for Whisper STT.
Fixes common acoustic and phonetic transcription errors for crops, pests, diseases,
and active chemical ingredients across English, Hindi, and Telugu.
"""

import re
from typing import Dict, List, Tuple


class AgriVocabularyCorrector:
    """
    Applies regex-based and phonetic correction rules to speech transcripts.
    Remaps commonly confused acoustic tokens to canonical agronomic terms.
    """

    # Common English, Hinglish, and Teluglish phonetic ASR confusions
    CORRECTION_RULES: List[Tuple[re.Pattern, str]] = [
        # Pests & Diseases
        (re.compile(r"\b(?:tripes|trips|traps|thrips tabaci)\b", re.IGNORECASE), "thrips"),
        (re.compile(r"\b(?:pink\s*ball\s*worm|pink\s*bowl\s*worm|pink\s*bol\s*worm)\b", re.IGNORECASE), "pink bollworm"),
        (re.compile(r"\b(?:stem\s*ply|stem\s*fry|stemfly)\b", re.IGNORECASE), "stem fly"),
        (re.compile(r"\b(?:girdle\s*beetle|girdal\s*beetle)\b", re.IGNORECASE), "girdle beetle"),
        (re.compile(r"\b(?:tuta\s*absolute|tuta\s*solute|tuta\s*soluta)\b", re.IGNORECASE), "Tuta absoluta"),
        (re.compile(r"\b(?:fruit\s*bore|fruit\s*border)\b", re.IGNORECASE), "fruit borer"),
        (re.compile(r"\b(?:purple\s*blotch|purple\s*bloch|purple\s*block)\b", re.IGNORECASE), "purple blotch"),
        (re.compile(r"\b(?:early\s*light|early\s*blite)\b", re.IGNORECASE), "early blight"),
        (re.compile(r"\b(?:leaf\s*curl\s*virus|tol\s*cv|tlcv)\b", re.IGNORECASE), "tomato leaf curl virus"),

        # Chemical & Bio Inputs
        (re.compile(r"\b(?:mancozeb\s*wp|manco\s*zeb|mankozeb)\b", re.IGNORECASE), "Mancozeb 75% WP"),
        (re.compile(r"\b(?:chlorantranili\w*|chlorantranil)\b", re.IGNORECASE), "Chlorantraniliprole 18.5% SC"),
        (re.compile(r"\b(?:fipronil|fipronil\s*sc|feroni)\b", re.IGNORECASE), "Fipronil 5% SC"),
        (re.compile(r"\b(?:profenofos|profenofos\s*ec|profeno\s*fos)\b", re.IGNORECASE), "Profenofos 50% EC"),
        (re.compile(r"\b(?:thiamethoxam|thiamethoxam\s*fs|thiamethox)\b", re.IGNORECASE), "Thiamethoxam 30% FS"),
        (re.compile(r"\b(?:trichoderma\s*viride|trico\s*derma)\b", re.IGNORECASE), "Trichoderma viride"),
        (re.compile(r"\b(?:bradyrhizobium|brady\s*rhizobium)\b", re.IGNORECASE), "Bradyrhizobium japonicum"),
        (re.compile(r"\b(?:copper\s*oxy\s*chloride|copper\s*oxychloride|coc)\b", re.IGNORECASE), "Copper Oxychloride 50% WP"),

        # Crops & Indian Regional Terms
        (re.compile(r"\b(?:erragada|yerragadda)\b", re.IGNORECASE), "erragadda"),
        (re.compile(r"\b(?:thamarapurugu|thamara\s*purugu)\b", re.IGNORECASE), "thamarapurugu"),
        (re.compile(r"\b(?:gulabi\s*purugu|gulabi\s*kandamu)\b", re.IGNORECASE), "gulabi purugu"),
        (re.compile(r"\b(?:kanda\s*crop|pyaaz\s*crop)\b", re.IGNORECASE), "onion"),
        (re.compile(r"\b(?:tamatar|tamata\s*crop)\b", re.IGNORECASE), "tomato"),
        (re.compile(r"\bpatti\s*crop\b", re.IGNORECASE), "patti"),
        (re.compile(r"\b(?:prathi\s*crop|kapas)\b", re.IGNORECASE), "cotton"),

        # Agronomy & Technical Terms
        (re.compile(r"\b(?:etl\s*level|e\s*t\s*l)\b", re.IGNORECASE), "Economic Threshold Level (ETL)"),
        (re.compile(r"\b(?:fir\s*treatment|f\s*i\s*r)\b", re.IGNORECASE), "FIR seed treatment"),
        (re.compile(r"\b(?:drip\s*irigation|drip\s*erigation)\b", re.IGNORECASE), "drip irrigation"),
    ]

    @classmethod
    def correct_transcript(cls, raw_transcript: str) -> str:
        """Applies vocabulary correction rules to raw ASR transcript."""
        if not raw_transcript:
            return ""

        corrected = raw_transcript
        for pattern, replacement in cls.CORRECTION_RULES:
            corrected = pattern.sub(replacement, corrected)

        # Clean multiple whitespace
        corrected = re.sub(r"\s+", " ", corrected).strip()
        return corrected
