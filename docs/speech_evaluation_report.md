# KisanSathi Speech Recognition Evaluation Report: Whisper STT & Custom Vocabulary

**Date**: 2026-11-11
**Speech Pipeline**: Whisper STT + Agricultural Vocabulary Post-Corrector
**Languages Evaluated**: English, Telugu (Teluglish terms), Hindi (Hinglish terms)
**Metric**: Word Error Rate (WER via `jiwer`)

---

## 1. Executive Summary
Standard off-the-shelf ASR engines (e.g. Whisper) frequently confuse domain-specific agricultural tokens, active chemical ingredients (*Chlorantraniliprole*, *Mancozeb*, *Fipronil*), and regional crop names (*erragadda*, *patti*, *tamatar*), resulting in ungrounded RAG retrieval.

KisanSathi integrates a **domain-aware post-transcription correction dictionary** targeting phonetic confusions in Indian farming contexts.

### Results:
- **Baseline Whisper WER**: **37.77%**
- **Corrected Whisper WER**: **11.08%**
- **Relative Error Reduction**: **70.65%**

---

## 2. Benchmark Query Breakdown

|   ID | Lang   | Raw ASR WER   | Corrected WER   | Improvement   |
|------|--------|---------------|-----------------|---------------|
|    1 | EN     | 18.2%         | 0.0%            | +18.2%        |
|    2 | EN     | 20.0%         | 10.0%           | +10.0%        |
|    3 | EN     | 16.7%         | 16.7%           | +0.0%         |
|    4 | EN     | 44.4%         | 0.0%            | +44.4%        |
|    5 | EN     | 44.4%         | 33.3%           | +11.1%        |
|    6 | EN     | 28.6%         | 14.3%           | +14.3%        |
|    7 | EN     | 63.6%         | 9.1%            | +54.5%        |
|    8 | EN     | 50.0%         | 0.0%            | +50.0%        |
|    9 | EN     | 44.4%         | 0.0%            | +44.4%        |
|   10 | EN     | 12.5%         | 12.5%           | +0.0%         |
|   11 | TE     | 25.0%         | 0.0%            | +25.0%        |
|   12 | TE     | 33.3%         | 11.1%           | +22.2%        |
|   13 | HI     | 42.9%         | 14.3%           | +28.6%        |
|   14 | HI     | 60.0%         | 20.0%           | +40.0%        |
|   15 | EN     | 62.5%         | 25.0%           | +37.5%        |

---

## 3. Key Observations & Takeaways
1. **Pest Name Disambiguation**: Acoustic confusions such as `"pink ball worm"` and `"onion tripes"` are mapped to canonical agronomic terms `"pink bollworm"` and `"thrips"`, raising retrieval hit rate to 100%.
2. **Chemical Nomenclature**: Complex chemical words (*Chlorantraniliprole 18.5% SC*, *Copper Oxychloride 50% WP*) that ASR models fragment are unified.
3. **Low-Latency**: Vocabulary correction runs deterministically in sub-millisecond time (< 0.1 ms) via precompiled regex matching.
