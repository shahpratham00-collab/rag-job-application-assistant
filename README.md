# 🎯 RAG Job Application Assistant

**Portfolio Project 7 of 7 — Pratham Shah | MSc AI & Data Science, Nottingham Trent University**

A production-quality Retrieval-Augmented Generation (RAG) application that transforms job hunting. Paste a job description and your CV — get a tailored cover letter, a match score, and a personalised skills gap action plan — all powered by LangChain, FAISS, and your choice of LLM backend.

---

## ✨ Features

| Feature | Description |
|---|---|
| **Tailored Cover Letter** | 300-400 word personalised letter, generated from retrieved CV context |
| **Match Score** | 0-100 compatibility score with colour-coded feedback |
| **Skills Gap Analysis** | Matched ✅ / Missing ❌ / Partial 🟡 skill breakdown |
| **Action Plan** | 3-5 concrete, prioritised next steps |
| **Dual LLM Backend** | Auto-selects Ollama (local) or HuggingFace (cloud) |
| **Download** | Export your cover letter as a `.txt` file instantly |

---

## 🏗️ Architecture

```
Input (JD + CV)
    │
    ├─► RecursiveCharacterTextSplitter (chunk_size=500)
    ├─► all-MiniLM-L6-v2 embeddings (local, CPU)
    ├─► FAISS vector store (in-memory)
    ├─► Similarity retrieval (k=4)
    └─► LLMChain → Cover Letter / Skills Gap JSON
```

See [`docs/architecture.md`](docs/architecture.md) for the full system design.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) (recommended for local use) **or** a HuggingFace account

### 1. Clone & Install

```bash
git clone <your-repo-url>
cd Project_7_RAG_Job_Application_Assistant
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings
```

### 3a. Run with Ollama (Local — Recommended)

```bash
# In a separate terminal:
ollama serve
ollama pull llama3

# Then:
streamlit run app.py
```

### 3b. Run with HuggingFace (Cloud)

```bash
# Add your token to .env:
# HUGGINGFACE_API_TOKEN=hf_your_token_here

streamlit run app.py
```

---

## 📁 Project Structure

```
Project_7_RAG_Job_Application_Assistant/
├── app.py                  # Streamlit UI & orchestration
├── requirements.txt        # Python dependencies
├── packages.txt            # Streamlit Cloud system packages
├── .env.example            # Environment variable template
├── src/
│   ├── rag_engine.py       # Core RAG pipeline
│   ├── llm_selector.py     # Ollama / HuggingFace auto-selector
│   ├── prompts.py          # Prompt templates
│   ├── parser.py           # Robust JSON output parser
│   └── utils.py            # Validation & helper functions
├── assets/
│   └── style.css           # Custom UI styling
├── tests/
│   └── test_rag_engine.py  # Unit tests (pytest)
└── docs/
    └── architecture.md     # System design documentation
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Expected output: 25+ tests covering utils, parser, and LLM selector logic.

---

## ☁️ Deploy to Streamlit Cloud

1. Push to a public GitHub repository
2. Visit [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. Set `HUGGINGFACE_API_TOKEN` in **Settings → Secrets**
4. Deploy — the app runs entirely on HuggingFace's hosted inference

---

## 🔧 Configuration

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3` | Ollama model name |
| `HUGGINGFACE_API_TOKEN` | — | HuggingFace API token |
| `HF_MODEL` | `mistralai/Mistral-7B-Instruct-v0.2` | HuggingFace model ID |

---

## 🛠️ Tech Stack

- **[Streamlit](https://streamlit.io)** — UI framework
- **[LangChain](https://langchain.com)** — LLM chain orchestration
- **[FAISS](https://github.com/facebookresearch/faiss)** — Vector similarity search
- **[sentence-transformers](https://www.sbert.net)** — Local text embeddings
- **[Ollama](https://ollama.com)** — Local LLM serving
- **[HuggingFace Hub](https://huggingface.co)** — Cloud LLM inference

---

## 👨‍💻 Author

**Pratham Shah**
MSc Artificial Intelligence & Data Science
Nottingham Trent University

- Email: shahpratham00@gmail.com
- GitHub: [shahpratham00-collab](https://github.com/shahpratham00-collab)

---

*Part of a 7-project AI portfolio demonstrating end-to-end machine learning and generative AI engineering.*
