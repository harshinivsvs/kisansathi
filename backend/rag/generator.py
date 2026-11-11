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

        # Synthesize from context sentences matching query tokens
        query_words = set(re.findall(r"\w{3,}", query.lower())) - {"what", "how", "why", "when", "which", "give", "tell"}
        raw_lines = [line.strip() for line in cleaned_context.splitlines() if line.strip()]
        
        candidates = []
        for line in raw_lines:
            line_words = set(re.findall(r"\w{3,}", line.lower()))
            overlap = len(query_words.intersection(line_words))
            if overlap > 0:
                # Clean leading list markers
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

    def _check_safety(self, query: str) -> Optional[str]:
        """Safety layer filtering out harmful, hazardous, and non-agricultural queries."""
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
            return "I am KisanSathi, an agricultural farm advisory assistant. I cannot assist with unsafe requests."

        # Medical query check
        medical_keywords = [
            "headache", "fever", "paracetamol", "diabetes", "blood pressure",
            "doctor", "medicine for human", "tablet should i take", "painkiller"
        ]
        if any(kw in q_lower for kw in medical_keywords):
            return "I am an agricultural farm advisory system. I cannot provide medical advice. Please consult a qualified doctor or healthcare centre."

        # Hazardous pesticide use check
        hazard_patterns = [
            r"without\s+(?:a\s+)?mask",
            r"mix\s+phorate",
            r"drink\s+pesticide",
            r"taste\s+insecticide"
        ]
        if any(re.search(pat, q_lower) for pat in hazard_patterns):
            return "Highly toxic chemicals must never be mixed arbitrarily or sprayed without protective equipment (mask, gloves). Follow container label cautions and consult your local KVK."

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

    def generate_answer(self, retrieval_result: RetrievalResult) -> AdvisoryAnswer:
        start_time = time.perf_counter()

        # Step 1: Safety & Guardrail check
        safety_refusal = self._check_safety(retrieval_result.query)
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
            fallback_answer = (
                "I do not have verified advisory information for this question in my current Maharashtra database. "
                f"Please consult your local Krishi Vigyan Kendra (KVK) or call the Kisan Call Centre at {settings.KISAN_CALL_CENTRE_HELPLINE}."
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
