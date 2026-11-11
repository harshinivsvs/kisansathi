"""
Multilingual processor for KisanSathi.
Supports English (en), Hindi (hi), and Telugu (te).
Provides language auto-detection, cross-lingual agronomic term mappings,
and localized response synthesis for low-literacy farmers.
"""

import re
from typing import Dict, List, Optional, Tuple

from backend.config import settings


class LanguageDetector:
    """Detects input language based on Unicode script ranges and agricultural keywords."""

    @staticmethod
    def detect_language(text: str) -> str:
        if not text or not text.strip():
            return "en"

        # Count character occurrences in distinct script Unicode ranges
        telugu_chars = len(re.findall(r"[\u0C00-\u0C7F]", text))
        hindi_chars = len(re.findall(r"[\u0900-\u097F]", text))

        total_indic = telugu_chars + hindi_chars
        if total_indic > 0:
            if telugu_chars >= hindi_chars:
                return "te"
            else:
                return "hi"

        # Check for transliterated Telugu / Hindi keywords (Romanized script) with whole word matching
        words = set(re.findall(r"\b\w+\b", text.lower()))
        telugu_roman = {"patti", "prathi", "ullipaya", "erragadda", "tamata", "thamarapurugu", "tegulu", "purugu", "dharalu"}
        hindi_roman = {"tamatar", "pyaaz", "kapas", "kanda", "keeda", "bimari", "chhidkao", "dawai", "kisan", "fallow"}

        telugu_match = len(words.intersection(telugu_roman))
        hindi_match = len(words.intersection(hindi_roman))

        if telugu_match > hindi_match:
            return "te"
        elif hindi_match > 0:
            return "hi"

        return "en"


class MultilingualAgriThesaurus:
    """Cross-lingual bidirectional mapping for crops, pests, and agronomic concepts."""

    CROP_SYNONYMS = {
        "tomato": {
            "en": ["tomato", "tomatoes"],
            "hi": ["टमाटर", "tamatar"],
            "te": ["టమాటా", "టమోటా", "తక్కాళి", "tamata", "tomota"]
        },
        "onion": {
            "en": ["onion", "onions"],
            "hi": ["प्याज", "कांदा", "pyaaz", "kanda"],
            "te": ["ఉల్లిపాయ", "ఉల్లిగడ్డ", "ఎర్రగడ్డ", "ullipaya", "erragadda", "ulli"]
        },
        "soybean": {
            "en": ["soybean", "soya bean", "soya"],
            "hi": ["सोयाबीन", "soyabean"],
            "te": ["సోయాబీన్", "సోయా", "soyabean", "soya"]
        },
        "cotton": {
            "en": ["cotton"],
            "hi": ["कपास", "kapas"],
            "te": ["పత్తి", "దూది", "patti", "prathi"]
        }
    }

    # Pests and diseases cross-lingual mappings
    PEST_DISEASE_MAPPINGS = {
        "fruit borer": {
            "hi": "फल छेदक कीट (Helicoverpa)",
            "te": "కాయతొలుచు పురుగు (Helicoverpa armigera)"
        },
        "thrips": {
            "hi": "थ्रिप्स / चूसक कीट",
            "te": "తామర పురుగులు (Thrips tabaci)"
        },
        "pink bollworm": {
            "hi": "गुलाबी सुंडी (Pink Bollworm)",
            "te": "గులాబీ రంగు కాయతొలుచు పురుగు (Pink Bollworm)"
        },
        "stem fly": {
            "hi": "तना मक्खी (Stem Fly)",
            "te": "కాండం ఈగ (Stem Fly)"
        },
        "early blight": {
            "hi": "अगेती झुलसा (Early Blight)",
            "te": "ముందస్తు తెగులు / ఆకుమచ్చ తెగులు (Early Blight)"
        },
        "purple blotch": {
            "hi": "बैंगनी धब्बा रोग (Purple Blotch)",
            "te": "ఊదా రంగు మచ్చ తెగులు (Purple Blotch)"
        },
        "seed rate": {
            "hi": "बीज दर",
            "te": "విత్తన మోతాదు"
        },
        "fertilizer": {
            "hi": "उर्वरक / खाद",
            "te": "ఎరువుల యాజమాన్యం"
        }
    }

    # Standard Localized Escapes and Safety Refusals
    LOCALIZED_FALLBACKS = {
        "en": (
            "I do not have verified advisory information for this question in my current Maharashtra database. "
            f"Please consult your local Krishi Vigyan Kendra (KVK) or call the Kisan Call Centre at {settings.KISAN_CALL_CENTRE_HELPLINE}."
        ),
        "hi": (
            "मेरे पास वर्तमान डेटाबेस में इस प्रश्न के लिए सत्यापित कृषि सलाह उपलब्ध नहीं है। "
            f"कृपया अपने नजदीकी कृषि विज्ञान केंद्र (KVK) से संपर्क करें या राष्ट्रीय किसान कॉल सेंटर {settings.KISAN_CALL_CENTRE_HELPLINE} पर कॉल करें।"
        ),
        "te": (
            "నా ప్రస్తుత డేటాబేస్‌లో ఈ ప్రశ్నకు సంబంధించి ధృవీకరించబడిన వ్యవసాయ సలహా అందుబాటులో లేదు. "
            f"దయచేసి మీ స్థానిక కృషి విజ్ఞాన కేంద్రం (KVK) ను సంప్రదించండి లేదా జాతీయ కిసాన్ కాల్ సెంటర్ {settings.KISAN_CALL_CENTRE_HELPLINE} కు కాల్ చేయండి."
        )
    }

    LOCALIZED_SAFETY_REFUSALS = {
        "medical": {
            "en": "I am an agricultural farm advisory system. I cannot provide medical advice. Please consult a qualified doctor or healthcare centre.",
            "hi": "मैं एक कृषि सलाहकार प्रणाली हूँ। मैं मानव चिकित्सा या स्वास्थ्य संबंधी सलाह नहीं दे सकता। कृपया किसी योग्य डॉक्टर या अस्पताल से संपर्क करें।",
            "te": "నేను కేవలం వ్యవసాయ సలహా సహాయకుడిని మాత్రమే. నేను వైద్య లేదా ఆరోగ్య సంబంధిత సలహాలు ఇవ్వలేను. దయచేసి అర్హత కలిగిన వైద్యుడిని సంప్రదించండి."
        },
        "hazard": {
            "en": "Highly toxic chemicals must never be mixed arbitrarily or sprayed without protective equipment (mask, gloves). Follow container label cautions and consult your local KVK.",
            "hi": "अत्यधिक विषैले कीटनाशकों को मनमाने ढंग से न मिलाएं और बिना मास्क व दस्तानों के छिड़काव न करें। हमेशा लेबल पर दिए गए सुरक्षा निर्देशों का पालन करें।",
            "te": "తీవ్రమైన విష రసాయనాలను ఏకపక్షంగా కలపకూడదు మరియు రక్షిత మాస్క్, చేతి తొడుగులు లేకుండా పిచికారీ చేయకూడదు. ఎల్లప్పుడూ లేబుల్ జాగ్రత్తలను పాటించండి."
        },
        "injection": {
            "en": "I am KisanSathi, an agricultural farm advisory assistant. I cannot assist with unsafe requests.",
            "hi": "मैं किसान साथी (KisanSathi) हूँ, एक कृषि सलाहकार सहायक। मैं असुरक्षित या अमान्य अनुरोधों में सहायता नहीं कर सकता।",
            "te": "నేను కిసాన్ సాథి (KisanSathi) వ్యవసాయ సలహాదారుడిని. నేను అసురక్షిత లేదా నిబంధనలకు విరుద్ధమైన అభ్యర్థనలకు సమాధానం ఇవ్వలేను."
        }
    }

    LOCALIZED_LABEL_CAUTION = {
        "en": "Always read container label safety precautions, wear protective gloves/mask, and consult your local KVK.",
        "hi": "सावधानी: कीटनाशक का प्रयोग करते समय सुरक्षा मास्क और दस्ताने पहनें तथा डिब्बे के लेबल पर दिए गए निर्देशों का पालन करें।",
        "te": "గమనిక: పురుగుమందు పిచికారీ చేసేటప్పుడు తప్పనిసరిగా రక్షిత మాస్క్ మరియు చేతి తొడుగులు ధరించండి మరియు డబ్బా లేబుల్ సూచనలను పాటించండి."
    }

    @classmethod
    def detect_crop_multilingual(cls, query: str) -> Optional[str]:
        """Detects crop across English, Hindi, and Telugu scripts and transliterations."""
        q_lower = query.lower()
        for crop, lang_dict in cls.CROP_SYNONYMS.items():
            for lang, synonyms in lang_dict.items():
                for syn in synonyms:
                    if syn in q_lower:
                        return crop
        return None

    @classmethod
    def expand_query_for_retrieval(cls, query: str, lang: str) -> str:
        """Enriches Indic queries with English agronomic synonyms for high-recall vector search."""
        expanded = [query]
        detected_crop = cls.detect_crop_multilingual(query)
        if detected_crop:
            expanded.append(detected_crop)

        # Check Telugu/Hindi pest keywords
        q_lower = query.lower()
        if "తామర" in query or "थ्रिप्स" in query or "thrips" in q_lower:
            expanded.append("thrips")
        if "కాయతొలుచు" in query or "फल छेदक" in query or "borer" in q_lower:
            expanded.append("fruit borer")
        if "గులాబీ" in query or "गुलाबी" in query or "bollworm" in q_lower:
            expanded.append("pink bollworm")
        if "కాండం ఈగ" in query or "तना मक्खी" in query or "stem fly" in q_lower:
            expanded.append("stem fly")
        if "విత్తన" in query or "बीज" in query or "seed" in q_lower:
            expanded.append("seed rate nursery")
        if "ఎరువు" in query or "खाद" in query or "fertilizer" in q_lower:
            expanded.append("fertilizer NPK")

        return " ".join(expanded)
