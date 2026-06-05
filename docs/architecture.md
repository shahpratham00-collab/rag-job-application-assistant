# Architecture — RAG Job Application Assistant

## Overview

The RAG Job Application Assistant uses a Retrieval-Augmented Generation (RAG) pipeline to produce personalised cover letters and skills gap analyses. It is designed to work fully offline (via Ollama) or in the cloud (via HuggingFace Inference API) without code changes.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Streamlit UI (app.py)                  │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐ │
│  │ Job Desc.    │  │  CV / Resume │  │  Settings sidebar  │ │
│  │ text_area    │  │  text_area   │  │  (temperature,     │ │
│  └──────┬───────┘  └──────┬───────┘  │   backend status)  │ │
│         └─────────────────┘          └────────────────────┘ │
│                     │                                        │
│                     ▼                                        │
│              validate_inputs()                               │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   RAG Engine (src/rag_engine.py)             │
│                                                             │
│  1. RecursiveCharacterTextSplitter                          │
│     chunk_size=500, chunk_overlap=50                        │
│     → Document chunks                                       │
│                                                             │
│  2. HuggingFaceEmbeddings (all-MiniLM-L6-v2)               │
│     → dense vectors (384-dim)                               │
│                                                             │
│  3. FAISS vector store                                      │
│     → in-memory index                                       │
│                                                             │
│  4. similarity_search(query, k=4)                           │
│     → retrieved context passages                            │
│                                                             │
│  5. LLMChain (PromptTemplate → LLM → output)               │
│     → cover letter / skills gap JSON                        │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                LLM Selector (src/llm_selector.py)           │
│                                                             │
│  Priority 1: Ollama (local)                                 │
│    GET /api/tags → 200 OK → use Ollama LLM                  │
│    Model: llama3 (configurable via OLLAMA_MODEL)            │
│                                                             │
│  Priority 2: HuggingFace Inference API (cloud)              │
│    HUGGINGFACE_API_TOKEN present → use HuggingFaceEndpoint  │
│    Model: mistralai/Mistral-7B-Instruct-v0.2 (configurable) │
│                                                             │
│  Fallback: RuntimeError with helpful instructions           │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Cover Letter Generation

```
JD + CV text
    │
    ├─► build_vector_store()
    │       └─► chunk → embed → FAISS index
    │
    ├─► retrieve_context(query="relevant experience...")
    │       └─► top-4 chunks by cosine similarity
    │
    └─► LLMChain(COVER_LETTER_PROMPT)
            inputs: job_description, cv_text, context
            └─► cover letter string
```

### Skills Gap Analysis

```
JD + CV text
    │
    └─► LLMChain(SKILLS_GAP_PROMPT, temperature=0.1)
            │
            └─► raw string output
                    │
                    ├─► parse_skills_gap_json()  ← tries 3 extraction strategies
                    │
                    └─► build_fallback_analysis() (if parsing fails)
```

---

## Module Responsibilities

| Module | Responsibility |
|---|---|
| `app.py` | Streamlit UI, orchestration, user feedback |
| `src/rag_engine.py` | Vector store construction, retrieval, chain execution |
| `src/llm_selector.py` | Runtime LLM/embedding selection (Ollama vs HF) |
| `src/prompts.py` | Prompt templates (cover letter, skills gap, summary) |
| `src/parser.py` | Robust JSON extraction from LLM output |
| `src/utils.py` | Input validation, text cleaning, formatting helpers |
| `assets/style.css` | Custom Streamlit component styling |

---

## Embedding Model

**all-MiniLM-L6-v2** (sentence-transformers)
- 384-dimensional dense vectors
- 22M parameters — fast on CPU
- Optimised for semantic similarity tasks
- Runs locally — no API key required

---

## Prompt Strategy

### Cover Letter
- System: expert career coach persona
- Retrieved context injected as a third input variable
- Explicit formatting instructions (length, tone, no placeholders)
- Temperature: user-controlled (default 0.7)

### Skills Gap
- System: senior technical recruiter persona
- Strict JSON output format enforced in prompt
- Temperature: fixed at 0.1 for deterministic structured output
- Three-layer JSON extraction fallback in `parser.py`

---

## Deployment Options

### Local (Ollama)
```bash
# Install Ollama: https://ollama.com
ollama serve
ollama pull llama3
streamlit run app.py
```

### Streamlit Cloud (HuggingFace)
1. Push to GitHub
2. Deploy on share.streamlit.io
3. Add `HUGGINGFACE_API_TOKEN` in Streamlit Secrets

---

## Design Decisions

- **FAISS over Chroma**: No persistence needed; in-memory is faster for single-session use.
- **Dual-backend auto-detection**: Prioritises local privacy (Ollama) with cloud fallback (HF).
- **Low temperature for structured output**: Skills gap JSON is deterministic at 0.1; creativity for cover letters at 0.7.
- **Three-layer JSON parser**: LLMs often wrap JSON in markdown fences or add preambles — the fallback chain handles all common failure modes.
- **`packages.txt`**: Required by Streamlit Cloud for system-level dependencies.
