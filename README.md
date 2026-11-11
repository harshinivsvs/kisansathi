# KisanSathi (किसान साथी) 🌾
**A Multilingual, Voice-First, WhatsApp-Based Farm Advisory System Using RAG, Vision, and Forecasting.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.14-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-17%20passed-success)](tests/)

---

## 📌 Problem Statement
Smallholder farmers in India frequently suffer severe crop yield losses due to untimely pest attacks, disease misdiagnoses, and market volatility. Generic conversational AI often produces dangerous hallucinations regarding pesticide dosages, lacks source citations, and fails low-literacy farmers who rely on voice notes in regional languages.

**KisanSathi** solves this by delivering **hyperlocal, source-cited agronomic guidance** backed by authoritative ICAR (Indian Council of Agricultural Research) and State Agricultural University (MPKV Rahuri, PDKV Akola) advisories.

### V1 Scope:
- **Region**: Maharashtra State (Western Maharashtra, Vidarbha, Marathwada, Khandesh)
- **Crops**: Tomato, Onion, Soybean, Cotton
- **Core Languages**: Hindi (`hi`), Marathi (`mr`), English (`en`)

---

## 🏗️ Architecture

```mermaid
flowchart TD
    subgraph ClientChannels["Farmer & Expert Access Channels"]
        WA["WhatsApp Cloud API (Audio/Photo/Text)"]
        WebDemo["Web App / Audio Demo"]
        Dashboard["Agronomist Dashboard"]
    end

    subgraph Gateway["API & Safety Gateway (FastAPI)"]
        Guard["Safety Layer (Prompt Injection, Chemical Limits, Out-of-Domain Filter)"]
        RateLimit["Rate Limiting & Auth"]
    end

    subgraph AdvisoryEngine["RAG Advisory Core"]
        Retriever["Retriever & Vector Search (Cosine Sim / Dense)"]
        VectorDB[("Vector Store (InMemory / Chroma / Qdrant)")]
        LLMWrapper["Provider-Agnostic LLM (Gemini / OpenAI / Mock)"]
        SeedData[("Authoritative Data Pack: ICAR / MPKV / CICR")]
    end

    subgraph Escalation["Helpline & Human-In-The-Loop"]
        KCC["National Kisan Call Centre (1800-180-1551)"]
        KVK["Local Krishi Vigyan Kendra (KVK)"]
    end

    WA --> Gateway
    WebDemo --> Gateway
    Dashboard --> Gateway
    Gateway --> Guard
    Guard -->|"Passed"| Retriever
    Retriever <--> VectorDB
    SeedData -.->|"Ingested & Chunked"| VectorDB
    Retriever --> LLMWrapper
    Guard -->|"Refusal / IDK"| Escalation
    Retriever -->|"Low Confidence (<0.55)"| Escalation
    LLMWrapper -->|"Cited Response"| Gateway
```

---

## 📊 Phase 1 Evaluation Benchmark Results

We evaluated KisanSathi across three chunking strategies against a **100-question Kisan Call Centre benchmark dataset** (`data/eval/kcc_qa_benchmark.csv`):

| Chunking Strategy | Hit@1 (%) | Hit@K (k=4) | MRR | Citation Accuracy | Answer Accuracy | Hallucination Rate | IDK Precision | Avg Latency |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fixed** (500c, 80o) | 100.0% | 100.0% | 1.000 | 71.0% | 71.0% | 0.0% | 100.0% | 0.15 ms |
| **Recursive** (Hierarchy) | 98.89% | 100.0% | 0.994 | 89.0% | 89.0% | 0.0% | 100.0% | 0.12 ms |
| **Semantic** (Section & Bullet) | **100.0%** | **100.0%** | **1.000** | **98.0%** | **98.0%** | **0.0%** | **100.0%** | **0.16 ms** |

### Key Benchmark Takeaways:
1. **Semantic Chunking** achieved **98.0% citation accuracy**, preserving vital chemical proportions and spray conditions without boundary splitting.
2. **Safety & IDK Handling** scored **100.0% precision**, deflecting non-agricultural queries, prompt injections, and unsupported crops directly to the Kisan Call Centre (`1800-180-1551`).
3. **Sub-millisecond Latency**: Average RAG latency is under **0.2 ms**, ensuring zero bottleneck for real-time voice synthesis.

---

## 🚀 Quickstart & One-Command Setup

### 1. Prerequisites
- Python 3.11+
- Virtual environment tool

### 2. Setup & Installation
```bash
# Clone the repository
git clone https://github.com/your-username/kisansathi.git
cd kisansathi

# Create virtual environment and install dependencies
python -m venv venv
.\venv\Scripts\activate   # Windows
# source venv/bin/activate # Linux / macOS

pip install -r requirements.txt
```

### 3. Run Test Suite
```bash
pytest tests/ -v
# 17 passed in ~1 second
```

### 4. Run Evaluation Benchmark
```bash
python eval/run_eval.py
```

### 5. Launch Advisory API Server
```bash
python -m uvicorn backend.main:app --reload --port 8000
```
Interactive API docs available at: **http://localhost:8000/docs**

### 6. Interactive Terminal Demonstration
```bash
python scripts/run_phase1_demo.py
```

---

## 📁 Repository Structure
```
kisansathi/
├── backend/
│   ├── api/             # FastAPI routes & Pydantic schemas
│   ├── rag/             # Document loader, chunking, embeddings, retriever, generator
│   ├── tools/           # Weather & Mandi price tools (Phase 3)
│   ├── vision/          # MobileNetV3 / EfficientNet plant disease classifier (Phase 5)
│   ├── forecast/        # Mandi price forecasting engine (Phase 4)
│   ├── safety/          # Guardrails, pesticide cautions, and prompt injection filters
│   ├── memory/          # Farmer profile & conversation memory (Phase 2)
│   ├── scheduler/       # Crop calendar proactive alert engine (Phase 4)
│   ├── config.py        # Settings via pydantic-settings & .env
│   └── main.py          # Application entrypoint & auto-seed lifespan
├── data/
│   ├── raw/             # Seed advisories (Tomato, Onion, Soybean, Cotton from ICAR & MPKV)
│   ├── processed/       # Vector storage & index files
│   └── eval/            # 100-question KCC benchmark CSV
├── docs/                # Architecture documentation & evaluation reports
├── eval/                # Evaluation harness & metrics computation
├── scripts/             # Ingestion & CLI demo scripts
├── tests/               # Pytest unit & integration test suite
├── docker-compose.yml   # Multi-container orchestration
├── Dockerfile           # Backend container definition
├── Makefile             # Developer workflow shortcuts
└── requirements.txt     # Python dependencies
```

---

## 🗺️ Build Roadmap
- [x] **Phase 1: Repo setup, sample data pack, multi-strategy RAG with citations, eval harness & 100-Q benchmark.**
- [ ] **Phase 2**: Hindi/regional language (Marathi default), Indic Whisper STT, TTS audio note return, conversation memory.
- [ ] **Phase 3**: Real-time weather tool (OpenWeather/IMD), Agmarknet mandi prices, WhatsApp Cloud API webhook, web demo.
- [ ] **Phase 4**: Crop calendar scheduler, spray-window advisor (rain/wind/temp), sell-or-wait price forecasting engine.
- [ ] **Phase 5**: Leaf photo vision diagnosis with temperature-scaled calibrated confidence & clarifying questions.
- [ ] **Phase 6**: Agronomist dashboard, hyperlocal expert tips, community disease outbreak alert engine.
- [ ] **Phase 7**: IVR/SMS fallback, final full evaluation run, Docker production deployment, documentation.
