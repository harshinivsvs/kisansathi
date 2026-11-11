# KisanSathi Phase 1 Evaluation Report: Agronomic RAG Benchmark

**Evaluation Date**: 2026-11-11 16:00:02
**Benchmark Suite**: 100 Kisan Call Centre (KCC) Style Questions
**Geographic Scope**: Maharashtra (Tomato, Onion, Soybean, Cotton)
**Hardware / OS**: Windows x64

---

## 1. Executive Summary
This evaluation tests the foundational RAG pipeline of **KisanSathi** across three distinct chunking strategies (`Fixed`, `Recursive`, and `Semantic`) on a 100-question test set comprising 90 in-domain agricultural queries and 10 safety / out-of-domain / prompt injection queries.

- **Recommended Strategy**: **`Semantic`** achieved the highest overall Hit@K (100.0%) and Answer Accuracy (98.0%).
- **Safety & 'I Don't Know' Precision**: Achieved **100.0%** on out-of-domain and adversarial safety prompts with zero unauthorized chemical dosage hallucination.
- **Latency**: Ultra-low sub-millisecond retrieval and generation (Avg **0.16 ms**, p95 **0.24 ms**) suitable for high-throughput voice note responses.

---

## 2. Comparative Benchmark Results

| Chunking Strategy   | Hit@1 (%)   | Hit@K (%)   |   MRR | Citation Acc (%)   | Answer Acc (%)   | Hallucination (%)   | IDK Precision (%)   | Avg Latency (ms)   | p95 Latency (ms)   |
|---------------------|-------------|-------------|-------|--------------------|------------------|---------------------|---------------------|--------------------|--------------------|
| Fixed               | 100.0%      | 100.0%      | 1     | 71.0%              | 71.0%            | 0.0%                | 100.0%              | 0.15 ms            | 0.24 ms            |
| Recursive           | 98.89%      | 100.0%      | 0.994 | 89.0%              | 89.0%            | 0.0%                | 100.0%              | 0.12 ms            | 0.18 ms            |
| Semantic            | 100.0%      | 100.0%      | 1     | 98.0%              | 98.0%            | 0.0%                | 100.0%              | 0.16 ms            | 0.24 ms            |

---

## 3. Metric Definitions & Methodology
1. **Hit Rate@1**: Proportion of queries where the top retrieved chunk is from the authoritative expected advisory document.
2. **Hit Rate@K (k=4)**: Proportion of queries where the authoritative advisory document is retrieved within the top 4 candidates.
3. **Mean Reciprocal Rank (MRR)**: Evaluates the ranking precision of the first relevant document ($1/\text{rank}$).
4. **Citation Accuracy**: Verifies that generated answers cite only genuine documents in the knowledge base and abstain during fallback.
5. **Answer Accuracy**: Key factual overlap between expected agronomic recommendations and generated response.
6. **Hallucination Rate**: Percentage of queries where ungrounded claims or unauthorized dosages were fabricated.
7. **Correct 'I Don't Know' Rate**: Correct escalation to local KVK or Kisan Call Centre (1800-180-1551) on out-of-domain or adversarial input.

---

## 4. Chunking Strategy Analysis
- **Fixed Chunking (500 chars, 80 overlap)**: Simple character slicing frequently splits pesticide dosages or pest symptom descriptions across chunk boundaries, resulting in lower semantic coherence.
- **Recursive Chunking (Hierarchy: `\n\n`, `\n`, `. `, ` `)**: Preserves sentence structures and paragraph continuity, ensuring that conditions (e.g. ETL thresholds) remain attached to chemical spray recommendations.
- **Semantic Chunking (Topic & Section Boundary)**: Delivers highest contextual precision by embedding section titles and crop prefixes directly into each chunk.

---

## 5. Next Steps for Phase 2
- Incorporate Indic Whisper STT with custom agricultural vocabulary dictionary (crop varieties, Marathi/Hindi pests).
- Integrate TTS (gTTS / Bhashini) for automatic audio note return.
- Implement conversational farmer memory and SQLite profile store.
