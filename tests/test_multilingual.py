"""
Unit tests for multilingual language detection and agronomic thesaurus (Telugu, Hindi, English).
"""

from backend.rag.multilingual import LanguageDetector, MultilingualAgriThesaurus


def test_language_detection_telugu():
    assert LanguageDetector.detect_language("టమాటాలో కాయతొలుచు పురుగు నివారణ ఏమిటి?") == "te"
    assert LanguageDetector.detect_language("ఉల్లిలో తామర పురుగులు వస్తున్నాయి") == "te"
    assert LanguageDetector.detect_language("patti crop lo thamarapurugu vachindi") == "te"


def test_language_detection_hindi():
    assert LanguageDetector.detect_language("टमाटर में फल छेदक के लिए क्या छिड़काव करें?") == "hi"
    assert LanguageDetector.detect_language("कपास में गुलाबी सुंडी का नियंत्रण कैसे करें?") == "hi"
    assert LanguageDetector.detect_language("tamatar me keeda lag gaya hai") == "hi"


def test_language_detection_english():
    assert LanguageDetector.detect_language("What is the seed rate for onion per hectare?") == "en"
    assert LanguageDetector.detect_language("How to identify pink bollworm in cotton?") == "en"


def test_multilingual_crop_detection():
    # Telugu
    assert MultilingualAgriThesaurus.detect_crop_multilingual("టమాటా విత్తన మోతాదు") == "tomato"
    assert MultilingualAgriThesaurus.detect_crop_multilingual("ఉల్లిపాయ ఎరువులు") == "onion"
    assert MultilingualAgriThesaurus.detect_crop_multilingual("పత్తిలో గులాబీ పురుగు") == "cotton"
    assert MultilingualAgriThesaurus.detect_crop_multilingual("సోయాబీన్ విత్తన శుద్ధి") == "soybean"

    # Hindi
    assert MultilingualAgriThesaurus.detect_crop_multilingual("टमाटर की खेती") == "tomato"
    assert MultilingualAgriThesaurus.detect_crop_multilingual("प्याज का थ्रिप्स") == "onion"
    assert MultilingualAgriThesaurus.detect_crop_multilingual("कपास का उत्पादन") == "cotton"
    assert MultilingualAgriThesaurus.detect_crop_multilingual("सोयाबीन बीज दर") == "soybean"


def test_localized_fallbacks():
    assert "1800-180-1551" in MultilingualAgriThesaurus.LOCALIZED_FALLBACKS["en"]
    assert "1800-180-1551" in MultilingualAgriThesaurus.LOCALIZED_FALLBACKS["hi"]
    assert "1800-180-1551" in MultilingualAgriThesaurus.LOCALIZED_FALLBACKS["te"]
    assert "కృషి విజ్ఞాన కేంద్రం" in MultilingualAgriThesaurus.LOCALIZED_FALLBACKS["te"]
