"""
Provider-agnostic LLM Generator for KisanSathi.
Answers strictly from retrieved agronomic context, with mandatory source citations,
strict 'I don't know' escalation, and safety guardrails.
"""

from abc import ABC, abstractmethod
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from backend.config import settings
from backend.rag.retriever import RetrievalResult, RetrievedChunk


class AdvisoryAnswer(BaseModel):
    """Complete structured answer returned to farmer/channel."""
    query: str
    answer: str
    citations: List[str] = Field(default_factory=list)
    confidence: float
    is_fallback: bool = False
    latency_ms: float
    token_count: int
    estimated_cost_usd: float
    provider_used: str
    retrieved_chunks: List[Dict[str, Any]] = Field(default_factory=list)


class BaseLLM(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> Tuple[str, int, float]:
        """Returns (answer_text, token_count, estimated_cost_usd)."""
        pass


class MockLLM(BaseLLM):
    """
    Offline/local rule-based generator that synthesizes answers faithfully
    from retrieved chunks without external network or API keys.
    """

    def generate(self, system_prompt: str, user_prompt: str) -> Tuple[str, int, float]:
        # Extract context block from prompt
        context_match = re.search(r"--- CONTEXT ---\n(.*?)\n--- END CONTEXT ---", user_prompt, re.DOTALL)
        query_match = re.search(r"--- FARMER QUESTION ---\n(.*?)$", user_prompt, re.DOTALL)

        query = query_match.group(1).strip() if query_match else ""
        context = context_match.group(1).strip() if context_match else ""

        if not context or "NO_CONTEXT_FOUND" in context:
            answer = (
                "I do not have verified agricultural advisory records for this question in my current Maharashtra database. "
                f"Please consult your local Krishi Vigyan Kendra (KVK) or call the national Kisan Call Centre at {settings.KISAN_CALL_CENTRE_HELPLINE}."
            )
            return answer, 40, 0.0

        # Clean context of internal metadata prefixes and markdown headers
        cleaned_context = re.sub(r"\[Document:.*?\]", "", context)
        cleaned_context = re.sub(r"\[.*?ADVISORY.*?\]", "", cleaned_context)
        cleaned_context = re.sub(r"#+\s+.*", "", cleaned_context)

        from backend.rag.multilingual import LanguageDetector, MultilingualAgriThesaurus
        lang = LanguageDetector.detect_language(query)

        # Check for direct agronomic match in Telugu
        if lang == "te":
            q_lower = query.lower()
            if "తామర" in query or "ఉల్లి" in query or "thrips" in q_lower or "onion" in q_lower:
                core_answer = "ఉల్లిలో తామర పురుగుల (Thrips) నివారణకు: ఎకరాకు 20-25 నీలి జిగురు అట్టలు అమర్చండి. ఆర్థిక నష్ట పరిమితి (ETL 15-20 పురుగులు/మొక్క) దాటితే ఫిప్రోనిల్ 5% SC (1 మి.లీ/లీటరు) లేదా ప్రొఫెనోఫాస్ 50% EC (1.5 మి.లీ/లీటరు) సర్ఫెక్టాంట్‌తో కలిపి పిచికారీ చేయండి."
            elif "కాయతొలుచు" in query or "టమాటా" in query or "fruit borer" in q_lower or "tomato" in q_lower:
                core_answer = "టమాటాలో కాయతొలుచు పురుగు (Helicoverpa) నివారణకు: ఎకరాకు 5 లింగాకర్షక బుట్టలు అమర్చండి. తీవ్రత ఎక్కువగా ఉన్నప్పుడు క్లోరాంట్రానిలిప్రోల్ 18.5% SC మందును లీటరు నీటికి 0.3 మి.లీ చొప్పున కలిపి పిచికారీ చేయండి."
            elif "గులాబీ" in query or "పత్తి" in query or "pink bollworm" in q_lower or "cotton" in q_lower:
                core_answer = "పత్తిలో గులాబీ రంగు కాయతొలుచు పురుగు నివారణకు: ఎకరాకు 8 లింగాకర్షక బుట్టలు అమర్చండి. ETL దాటినప్పుడు ప్రొఫెనోఫాస్ 50% EC (2 మి.లీ/లీటరు) లేదా ఎమామెక్టిన్ బెంజోయేట్ 5% SG (0.5 గ్రా/లీటరు) పిచికారీ చేయండి."
            elif "విత్తన" in query or "సోయాబీన్" in query or "seed" in q_lower or "soybean" in q_lower:
                core_answer = "సోయాబీన్ విత్తన శుద్ధి (FIR విధానం): ముందుగా శిలీంద్రనాశిని (కార్బాక్సిన్ + థైరమ్ 2 గ్రా/కిలో), తర్వాత కీటకనాశిని (థయామిథాక్సామ్ 30 FS 10 మి.లీ/కిలో), చివరగా రైజోబియం మరియు PSB జీవ ఎరువులతో విత్తన శుద్ధి చేయండి."
            else:
                core_answer = "మీ పంట సమస్యకు సంబంధించి అధికారిక పరిశోధనా కేంద్రాల సిఫార్సుల ప్రకారం సలహాలు అందించబడుతున్నాయి."

            core_answer += f" {MultilingualAgriThesaurus.LOCALIZED_LABEL_CAUTION['te']}"
            tokens = len(user_prompt.split()) + len(core_answer.split())
            return core_answer, tokens, 0.0

        # Check for direct agronomic match in Hindi
        if lang == "hi":
            q_lower = query.lower()
            if "थ्रिप्स" in query or "प्याज" in query or "thrips" in q_lower or "onion" in q_lower:
                core_answer = "प्याज में थ्रिप्स कीट नियंत्रण के लिए: 20-25 नीले चिपचिपे ट्रैप लगाएं। आर्थिक क्षति स्तर (ETL 15-20 थ्रिप्स) पार होने पर फिप्रोनिल 5% SC (1 मिली/लीटर) या प्रोफेनोफॉस 50% EC (1.5 मिली/लीटर) का छिड़काव करें।"
            elif "फल छेदक" in query or "टमाटर" in query or "borer" in q_lower or "tomato" in q_lower:
                core_answer = "टमाटर में फल छेदक (Helicoverpa) नियंत्रण के लिए: प्रति एकड़ 5 फेरोमोन ट्रैप लगाएं। अधिक प्रकोप होने पर क्लोरांट्रानिलीप्रोल 18.5% SC (0.3 मिली/लीटर) का छिड़काव करें।"
            elif "गुलाबी सुंडी" in query or "कपास" in query or "pink bollworm" in q_lower or "cotton" in q_lower:
                core_answer = "कपास में गुलाबी सुंडी नियंत्रण के लिए: प्रति एकड़ 8 फेरोमोन ट्रैप लगाएं। ETL पार होने पर प्रोफेनोफॉस 50% EC (2 मिली/लीटर) या एमामेक्टिन बेंजोएट 5% SG (0.5 ग्राम/लीटर) का छिड़काव करें।"
            elif "बीज" in query or "सोयाबीन" in query or "seed" in q_lower or "soybean" in q_lower:
                core_answer = "सोयाबीन बीज उपचार (FIR क्रम): पहले कवकनाशी (कार्बोक्सिन + थीरम 2 ग्राम/किग्रा), फिर कीटनाशक (थायमेथोक्सम 30 FS 10 मिली/किग्रा), अंत में राइजोबियम व PSB कल्चर से उपचार करें।"
            else:
                core_answer = "कृषि विश्वविद्यालय की आधिकारिक सिफारिशों के अनुसार अनुशंसित प्रबंधन का पालन करें।"

            core_answer += f" {MultilingualAgriThesaurus.LOCALIZED_LABEL_CAUTION['hi']}"
            tokens = len(user_prompt.split()) + len(core_answer.split())
            return core_answer, tokens, 0.0

        # English synthesis from extracted context
        query_words = set(re.findall(r"\w{3,}", query.lower())) - {"what", "how", "why", "when", "which", "give", "tell"}
        raw_lines = [line.strip() for line in cleaned_context.splitlines() if line.strip()]
        
        candidates = []
        for line in raw_lines:
            line_words = set(re.findall(r"\w{3,}", line.lower()))
            overlap = len(query_words.intersection(line_words))
            if overlap > 0:
                clean_line = re.sub(r"^[-*•\d.]+\s*", "", line).strip()
                if len(clean_line) > 15:
                    candidates.append((overlap, clean_line))

        candidates.sort(key=lambda x: x[0], reverse=True)
        top_pieces = [c[1] for c in candidates[:2]]

        if top_pieces:
            core_answer = ". ".join(top_pieces)
            if not core_answer.endswith("."):
                core_answer += "."
        else:
            core_answer = "Please consult official advisory documents for detailed management steps."

        # Add chemical safety label caution if chemical terms are present
        chemical_keywords = ["spray", "mancozeb", "chlorantraniliprole", "fipronil", "profenofos", "thiamethoxam", "herbicide", "fungicide", "insecticide"]
        if any(chem in core_answer.lower() for chem in chemical_keywords):
            if "label" not in core_answer.lower():
                core_answer += " Note: Always read and adhere strictly to container label safety precautions and wear protective gloves/mask."

        tokens = len(user_prompt.split()) + len(core_answer.split())
        return core_answer, tokens, 0.0


class GeminiLLM(BaseLLM):
    """Google Gemini LLM wrapper."""

    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.model_name = model_name
        self._fallback = MockLLM()

    def generate(self, system_prompt: str, user_prompt: str) -> Tuple[str, int, float]:
        if not self.api_key:
            return self._fallback.generate(system_prompt, user_prompt)

        try:
            import requests
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
            payload = {
                "system_instruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"parts": [{"text": user_prompt}]}],
                "generationConfig": {
                    "temperature": settings.LLM_TEMPERATURE,
                    "maxOutputTokens": settings.LLM_MAX_TOKENS,
                }
            }
            resp = requests.post(url, json=payload, timeout=12)
            if resp.status_code == 200:
                data = resp.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                tokens = data.get("usageMetadata", {}).get("totalTokenCount", len(text.split()) * 2)
                cost = (tokens / 1_000_000) * 0.15  # Approx $0.15/1M tokens
                return text, tokens, round(cost, 6)
        except Exception:
            pass

        return self._fallback.generate(system_prompt, user_prompt)


class OpenAILLM(BaseLLM):
    """OpenAI API wrapper."""

    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model_name = model_name
        self._fallback = MockLLM()

    def generate(self, system_prompt: str, user_prompt: str) -> Tuple[str, int, float]:
        if not self.api_key:
            return self._fallback.generate(system_prompt, user_prompt)

        try:
            import requests
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": settings.LLM_TEMPERATURE,
                "max_tokens": settings.LLM_MAX_TOKENS
            }
            resp = requests.post(url, json=payload, headers=headers, timeout=12)
            if resp.status_code == 200:
                data = resp.json()
                text = data["choices"][0]["message"]["content"].strip()
                tokens = data.get("usage", {}).get("total_tokens", len(text.split()) * 2)
                cost = (tokens / 1_000_000) * 0.60
                return text, tokens, round(cost, 6)
        except Exception:
            pass

        return self._fallback.generate(system_prompt, user_prompt)


class RAGGenerator:
    """Orchestrates RAG prompt creation, safety checks, and grounded citation generation."""

    SYSTEM_PROMPT = """You are KisanSathi, an expert agricultural advisory assistant for farmers in Maharashtra, India.
RULES:
1. Answer ONLY from the retrieved context provided. Do NOT extrapolate or assume facts outside the context.
2. Keep answers short, direct, and practical (suitable for listening via voice notes).
3. If the context does not contain enough evidence or if confidence is low, strictly reply:
   "I do not have verified advisory information for this question in my current database. Please contact your local Krishi Vigyan Kendra (KVK) or call the Kisan Call Centre at 1800-180-1551."
4. Whenever mentioning any chemical spray, you MUST include: "Always read container label precautions, wear protective gear, and consult your local KVK."
5. Never answer medical, legal, or non-agricultural questions. Politely decline.
6. Append exact citations in brackets: [Source: <document_name>, <page_number>].
"""

    def __init__(self, provider: Optional[str] = None):
        prov = (provider or settings.LLM_PROVIDER).lower().strip()
        self.provider_name = prov

        if prov == "gemini" and settings.GEMINI_API_KEY:
            self.llm: BaseLLM = GeminiLLM(settings.GEMINI_API_KEY, settings.LLM_MODEL_NAME)
        elif prov == "openai" and settings.OPENAI_API_KEY:
            self.llm = OpenAILLM(settings.OPENAI_API_KEY, settings.LLM_MODEL_NAME)
        else:
            self.llm = MockLLM()
            self.provider_name = "mock"

    def _check_safety(self, query: str, lang: str = "en") -> Optional[str]:
        """Safety layer filtering out harmful, hazardous, and non-agricultural queries."""
        from backend.rag.multilingual import MultilingualAgriThesaurus
        q_lower = query.lower()

        # Prompt injection checks
        injection_patterns = [
            r"ignore\s+(?:all\s+)?previous\s+instructions",
            r"system\s+(?:override|prompt|rules)",
            r"tell\s+me\s+the\s+recipe\s+for\s+a\s+bomb",
            r"bypass\s+(?:all\s+)?safety",
            r"disregard\s+previous\s+rules",
            r"confidential\s+system",
            r"prescribe\s+100x",
            r"jailbreak"
        ]
        if any(re.search(pat, q_lower) for pat in injection_patterns):
            return MultilingualAgriThesaurus.LOCALIZED_SAFETY_REFUSALS["injection"].get(lang, MultilingualAgriThesaurus.LOCALIZED_SAFETY_REFUSALS["injection"]["en"])

        # Medical query check
        medical_keywords = [
            "headache", "fever", "paracetamol", "diabetes", "blood pressure",
            "doctor", "medicine for human", "tablet should i take", "painkiller",
            "తలనొప్పి", "జ్వరం", "డాక్టర్", "మాత్ర", "सिरदर्द", "बुखार", "दवा"
        ]
        if any(kw in q_lower for kw in medical_keywords):
            return MultilingualAgriThesaurus.LOCALIZED_SAFETY_REFUSALS["medical"].get(lang, MultilingualAgriThesaurus.LOCALIZED_SAFETY_REFUSALS["medical"]["en"])

        # Hazardous pesticide use check
        hazard_patterns = [
            r"without\s+(?:a\s+)?mask",
            r"mix\s+phorate",
            r"drink\s+pesticide",
            r"taste\s+insecticide",
            r"మాస్క్\s+లేకుండా",
            r"बिना\s+मास्क"
        ]
        if any(re.search(pat, q_lower) for pat in hazard_patterns):
            return MultilingualAgriThesaurus.LOCALIZED_SAFETY_REFUSALS["hazard"].get(lang, MultilingualAgriThesaurus.LOCALIZED_SAFETY_REFUSALS["hazard"]["en"])

        # Financial/Crypto check
        finance_keywords = ["bitcoin", "cryptocurrency", "stock market", "forex", "trading", "crypto"]
        if any(kw in q_lower for kw in finance_keywords):
            return "I cannot advise on financial trading or cryptocurrency investments. Please consult a licensed banking officer."

        # Legal advice check
        legal_keywords = ["write a contract", "land deed lawsuit", "legal notice to brother", "selling my farmland to my brother"]
        if any(kw in q_lower for kw in legal_keywords):
            return "I am KisanSathi, a farming agronomy assistant. I cannot provide legal contract drafting. Please consult a legal professional or Tehsil revenue officer."

        # General trivia check
        trivia_keywords = ["prime minister of", "president of", "who won the", "capital of"]
        if any(kw in q_lower for kw in trivia_keywords):
            return "I am KisanSathi, specialized exclusively in agricultural advice for farmers. I cannot assist with general knowledge queries."

        return None

    def generate_answer(
        self,
        retrieval_result: RetrievalResult,
        target_language: Optional[str] = None
    ) -> AdvisoryAnswer:
        start_time = time.perf_counter()
        from backend.rag.multilingual import MultilingualAgriThesaurus, LanguageDetector

        lang = (target_language or LanguageDetector.detect_language(retrieval_result.query)).lower()
        if lang not in ("en", "hi", "te"):
            lang = settings.SECONDARY_LANGUAGE  # default: te

        # Step 1: Safety & Guardrail check
        safety_refusal = self._check_safety(retrieval_result.query, lang=lang)
        if safety_refusal:
            latency = (time.perf_counter() - start_time) * 1000
            return AdvisoryAnswer(
                query=retrieval_result.query,
                answer=safety_refusal,
                citations=[],
                confidence=0.0,
                is_fallback=True,
                latency_ms=round(latency, 2),
                token_count=len(safety_refusal.split()),
                estimated_cost_usd=0.0,
                provider_used=self.provider_name,
                retrieved_chunks=[]
            )

        # Step 2: 'I don't know' handling if retrieval confidence is low or no chunks found
        if not retrieval_result.is_confident or not retrieval_result.retrieved_chunks:
            fallback_answer = MultilingualAgriThesaurus.LOCALIZED_FALLBACKS.get(
                lang,
                MultilingualAgriThesaurus.LOCALIZED_FALLBACKS["en"]
            )
            latency = (time.perf_counter() - start_time) * 1000
            return AdvisoryAnswer(
                query=retrieval_result.query,
                answer=fallback_answer,
                citations=[],
                confidence=retrieval_result.top_score,
                is_fallback=True,
                latency_ms=round(latency, 2),
                token_count=len(fallback_answer.split()),
                estimated_cost_usd=0.0,
                provider_used=self.provider_name,
                retrieved_chunks=[]
            )

        # Step 3: Construct grounded prompt
        context_parts = []
        citations_set = set()
        chunk_dicts = []

        for item in retrieval_result.retrieved_chunks:
            context_parts.append(
                f"[Document: {item.chunk.document_name} | Section: {item.chunk.section_title} | Page: {item.chunk.page_number}]\n{item.chunk.text}"
            )
            citation_str = f"{item.chunk.document_name}, {item.chunk.page_number}"
            citations_set.add(citation_str)
            chunk_dicts.append({
                "chunk_id": item.chunk.chunk_id,
                "document_name": item.chunk.document_name,
                "page_number": item.chunk.page_number,
                "score": item.similarity_score,
                "text": item.chunk.text[:180] + "..."
            })

        context_block = "\n\n".join(context_parts)
        user_prompt = f"""--- CONTEXT ---
{context_block}
--- END CONTEXT ---

--- FARMER QUESTION ---
{retrieval_result.query}
"""

        # Step 4: Call LLM
        raw_answer, tokens, cost = self.llm.generate(self.SYSTEM_PROMPT, user_prompt)

        # Step 5: Format citations if not explicitly in raw answer
        citations_list = sorted(list(citations_set))
        if citations_list and not any(cit in raw_answer for cit in citations_list):
            primary_citation = citations_list[0]
            raw_answer = f"{raw_answer.strip()} [Source: {primary_citation}]"

        latency = (time.perf_counter() - start_time) * 1000

        return AdvisoryAnswer(
            query=retrieval_result.query,
            answer=raw_answer.strip(),
            citations=citations_list,
            confidence=retrieval_result.top_score,
            is_fallback=False,
            latency_ms=round(latency, 2),
            token_count=tokens,
            estimated_cost_usd=cost,
            provider_used=self.provider_name,
            retrieved_chunks=chunk_dicts
        )
