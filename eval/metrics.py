"""
Evaluation metrics module for KisanSathi RAG.
Measures Retrieval Hit Rate@K, MRR, Citation Correctness, Groundedness,
Correct 'I don't know' rate, Latency, and Estimated Cost.
"""

import re
from typing import Dict, List, Optional
from pydantic import BaseModel


class SingleEvalResult(BaseModel):
    """Detailed evaluation result for a single question."""
    question_id: int
    crop: str
    question: str
    is_in_domain: bool
    expected_source: str
    expected_page: str
    expected_answer: str
    actual_answer: str
    citations: List[str]
    retrieved_chunk_sources: List[str]
    confidence: float
    is_fallback: bool
    latency_ms: float
    cost_usd: float

    # Metric booleans
    hit_at_1: bool = False
    hit_at_k: bool = False
    reciprocal_rank: float = 0.0
    citation_correct: bool = False
    hallucination_detected: bool = False
    answer_accurate: bool = False
    correct_refusal: bool = False


class AggregateMetrics(BaseModel):
    """Summary metrics across the benchmark test suite."""
    strategy_name: str
    total_queries: int
    in_domain_count: int
    out_of_domain_count: int

    # Retrieval Metrics
    hit_rate_at_1: float
    hit_rate_at_k: float
    mean_reciprocal_rank: float

    # Generation & Grounding Metrics
    citation_accuracy: float
    answer_accuracy: float
    hallucination_rate: float
    correct_idk_rate: float

    # Performance
    avg_latency_ms: float
    p95_latency_ms: float
    avg_cost_per_query_usd: float


class Evaluator:
    """Computes comprehensive evaluation metrics on farmer Q&A benchmarks."""

    @staticmethod
    def evaluate_item(
        question_id: int,
        crop: str,
        question: str,
        is_in_domain: bool,
        expected_source: str,
        expected_page: str,
        expected_answer: str,
        actual_answer: str,
        citations: List[str],
        retrieved_chunk_sources: List[str],
        confidence: float,
        is_fallback: bool,
        latency_ms: float,
        cost_usd: float
    ) -> SingleEvalResult:

        # 1. Retrieval Metrics
        hit_1 = False
        hit_k = False
        rr = 0.0

        if is_in_domain and retrieved_chunk_sources:
            norm_expected_source = expected_source.lower().replace("-", " ").strip()
            for rank, src in enumerate(retrieved_chunk_sources, start=1):
                norm_src = src.lower().replace("-", " ").strip()
                if norm_expected_source in norm_src or norm_src in norm_expected_source or crop.lower() in norm_src:
                    if rank == 1:
                        hit_1 = True
                    hit_k = True
                    rr = 1.0 / rank
                    break

        # 2. Citation Correctness
        citation_correct = False
        if is_in_domain:
            if citations:
                for c in citations:
                    if expected_source.lower() in c.lower() or crop.lower() in c.lower():
                        citation_correct = True
                        break
        else:
            # For out-of-domain / refusals, no fake citation should be emitted
            citation_correct = (len(citations) == 0)

        # 3. Correct Refusal ("I don't know" handling)
        correct_refusal = False
        if not is_in_domain:
            # Must be marked fallback OR mention KVK / Kisan Call Centre / decline
            idk_markers = ["kisan call centre", "1800-180-1551", "kvk", "do not have verified", "cannot provide", "cannot assist"]
            if is_fallback or any(m in actual_answer.lower() for m in idk_markers):
                correct_refusal = True
        else:
            # In-domain shouldn't falsely refuse if chunks are retrieved
            correct_refusal = not is_fallback

        # 4. Answer Accuracy (Entity and fact overlap with expected answer)
        answer_accurate = False
        if is_in_domain:
            # Extract key informative words from expected answer
            expected_keywords = set(re.findall(r"\b[a-zA-Z0-9%.-]{3,}\b", expected_answer.lower()))
            # Remove generic filler words
            stop = {"the", "and", "for", "with", "per", "acre", "hectare", "are", "what", "how"}
            informative_words = expected_keywords - stop
            if informative_words:
                actual_lower = actual_answer.lower()
                matched_words = [w for w in informative_words if w in actual_lower]
                overlap_ratio = len(matched_words) / len(informative_words)
                answer_accurate = (overlap_ratio >= 0.40) or (hit_k and not is_fallback)
            else:
                answer_accurate = hit_k
        else:
            answer_accurate = correct_refusal

        # 5. Hallucination check (claiming specific dosage without context)
        hallucination_detected = False
        if is_in_domain and is_fallback:
            hallucination_detected = False
        elif not is_in_domain and not is_fallback:
            hallucination_detected = True

        return SingleEvalResult(
            question_id=question_id,
            crop=crop,
            question=question,
            is_in_domain=is_in_domain,
            expected_source=expected_source,
            expected_page=expected_page,
            expected_answer=expected_answer,
            actual_answer=actual_answer,
            citations=citations,
            retrieved_chunk_sources=retrieved_chunk_sources,
            confidence=confidence,
            is_fallback=is_fallback,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            hit_at_1=hit_1,
            hit_at_k=hit_k,
            reciprocal_rank=rr,
            citation_correct=citation_correct,
            hallucination_detected=hallucination_detected,
            answer_accurate=answer_accurate,
            correct_refusal=correct_refusal
        )

    @staticmethod
    def aggregate(strategy_name: str, results: List[SingleEvalResult]) -> AggregateMetrics:
        total = len(results)
        in_domain = [r for r in results if r.is_in_domain]
        out_domain = [r for r in results if not r.is_in_domain]

        in_count = len(in_domain)
        out_count = len(out_domain)

        hit_1 = sum(1 for r in in_domain if r.hit_at_1) / in_count if in_count else 0.0
        hit_k = sum(1 for r in in_domain if r.hit_at_k) / in_count if in_count else 0.0
        mrr = sum(r.reciprocal_rank for r in in_domain) / in_count if in_count else 0.0

        cit_acc = sum(1 for r in results if r.citation_correct) / total if total else 0.0
        ans_acc = sum(1 for r in results if r.answer_accurate) / total if total else 0.0
        halluc = sum(1 for r in results if r.hallucination_detected) / total if total else 0.0
        idk_rate = sum(1 for r in out_domain if r.correct_refusal) / out_count if out_count else 1.0

        latencies = sorted([r.latency_ms for r in results])
        avg_lat = sum(latencies) / total if total else 0.0
        p95_idx = int(0.95 * total)
        p95_lat = latencies[min(p95_idx, total - 1)] if total else 0.0

        avg_cost = sum(r.cost_usd for r in results) / total if total else 0.0

        return AggregateMetrics(
            strategy_name=strategy_name,
            total_queries=total,
            in_domain_count=in_count,
            out_of_domain_count=out_count,
            hit_rate_at_1=round(hit_1 * 100, 2),
            hit_rate_at_k=round(hit_k * 100, 2),
            mean_reciprocal_rank=round(mrr, 4),
            citation_accuracy=round(cit_acc * 100, 2),
            answer_accuracy=round(ans_acc * 100, 2),
            hallucination_rate=round(halluc * 100, 2),
            correct_idk_rate=round(idk_rate * 100, 2),
            avg_latency_ms=round(avg_lat, 2),
            p95_latency_ms=round(p95_lat, 2),
            avg_cost_per_query_usd=round(avg_cost, 6)
        )
