# 🌿 Floramind — RAG Gardening Chatbot with Google Gemini

Floramind is an intelligent Retrieval-Augmented Generation (RAG) gardening mentor designed for beginners. It combines Hugging Face embeddings, Pinecone vector search, and **Google Gemini** as the primary Large Language Model (LLM) to deliver practical, grounded gardening advice with verified source citations.

---

## 🏛️ Updated AI Architecture

Floramind coordinates retrieval, generation, and citation attribution through a modular, decoupled architecture:

```
                  ┌──────────────────────────────────────────────┐
                  │          FastAPI Web Application             │
                  │ (Single-turn & Multi-turn Chat, Health, UI)  │
                  └──────────────┬───────────────────────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
   ┌───────────────────────────┐   ┌───────────────────────────┐
   │   Hugging Face Model      │   │     Pinecone Vector DB    │
   │ all-MiniLM-L6-v2 (384-dim)│   │ Semantic Cosine Retrieval │
   │ (Query & Doc Embeddings)  │   │  (Top 5 Document Chunks)  │
   └─────────────┬─────────────┘   └─────────────┬─────────────┘
                 │                               │
                 └───────────────┬───────────────┘
                                 ▼
                 ┌───────────────────────────────┐
                 │    Relevance Threshold Gate   │
                 │   (Checks cosine score >= 0.45│
                 └───────────────┬───────────────┘
                                 ▼
                 ┌───────────────────────────────┐
                 │     gemini_service.py         │
                 │  Official Google GenAI SDK    │
                 │   (google-genai / Gemini)     │
                 │    Interactions API Support   │
                 └───────────────┬───────────────┘
                                 ▼
                 ┌───────────────────────────────┐
                 │   Floramind Mentor Answer     │
                 │  + Verified Source Citations  │
                 └───────────────────────────────┘
```

1. **Hugging Face Sentence Transformers (`sentence-transformers/all-MiniLM-L6-v2`):**
   - Converts gardening knowledge documents and user questions into high-density 384-dimensional vector embeddings.
2. **Pinecone Vector Database:**
   - Stores knowledge embeddings and executes semantic similarity searches for the top five most relevant document chunks.
3. **Relevance Threshold Check:**
   - Validates that retrieved chunks actually match the user's intent. If similarity is insufficient, Floramind acknowledges the limitation and prevents source hallucination.
4. **Google Gemini API (`google-genai`):**
   - Serves as the primary LLM inside a dedicated `gemini_service.py` module.
   - Defaults to `gemini-2.5-flash` for responsive, fast, and helpful gardening conversations.
   - Configurable dynamically via the `GEMINI_MODEL` environment variable.
   - Prefers Google's modern Interactions API (`client.interactions.create`) when supported, with automatic fallback to `client.models.generate_content`.
5. **Python FastAPI Backend:**
   - Orchestrates the RAG lifecycle, manages conversation history, formats citations, and serves the web interface.
6. **Bootstrap 5 & Botanical UI:**
   - Fresh, responsive botanical design with glassmorphism, prompt suggestion chips, markdown rendering, and collapsible citation cards.

---

## 🌻 Gemini Gardening Mentor Personality

Configured as **Floramind**, a friendly, encouraging gardening mentor:
- **Beginner-Friendly:** Uses simple English, explaining concepts step-by-step.
- **Practical Guidance:** Practical advice on seeds, soil, watering, sunlight, fertilizers, compost, vegetables, herbs, flowers, pests, and plant diseases.
- **Clarifying Questions:** Asks clarifying questions when plant species, container size, climate, or indoor/outdoor conditions matter.
- **Tropical Gardening:** Offers tailored guidance for humid, tropical environments (monsoon seasons, soil leaching, raised drainage beds, kangkong, bougainvillea, calamansi, hibiscus, orchids).
- **Grounding Safeguards:** Strictly bases answers on retrieved knowledge. Never fabricates citations. If references are insufficient, clearly explains the limitation.
- **Safety & Responsibility:** Emphasizes safety around toxic plants for pets/children (lilies, oleander, dieffenbachia) and prioritizes organic pest control before synthetic chemicals.

---

## ⚙️ Environment Configuration

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

### Environment Variables (`.env.example`)

| Variable | Description | Default / Example |
|---|---|---|
| `GEMINI_API_KEY` | Google Gemini API Key | `your_gemini_api_key` |
| `GEMINI_MODEL` | Selected Gemini Flash Model | `gemini-2.5-flash` |
| `HF_EMBEDDING_MODEL` | Hugging Face Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` |
| `PINECONE_API_KEY` | Pinecone Vector DB Key | `your_pinecone_api_key` |
| `PINECONE_INDEX_NAME` | Pinecone Index Name | `floramind-gardening` |
| `HOST` | Server Host IP | `127.0.0.1` |
| `PORT` | Server Port | `8000` |
| `DEBUG` | Enable Debug Mode | `False` |

> **Security Note:** All credentials are kept on the server and are never exposed in client JavaScript, API responses, or git commits.

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Index Gardening Knowledge into Pinecone
Populate the vector index with the curated gardening knowledge base (`data/gardening_docs.json`):
```bash
python scripts/index_knowledge.py
```
*(If Pinecone is not yet configured, Floramind seamlessly falls back to local semantic search so you can start right away).*

### 3. Launch the Application
```bash
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
Open your browser at: **`http://127.0.0.1:8000`**

---

## 🧪 Automated Testing

Floramind includes an automated test suite with full mocking for Google Gemini and Pinecone so tests run without consuming API quotas:

```bash
python -m pytest tests/ -v
```

### Test Coverage Highlights:
- **`tests/test_gemini_service.py`:** Tests Gemini client initialization, prompt construction, Interactions API fallback to `generate_content`, mock generation, and translation of API errors (401 invalid key, 429 quota exhaustion, 404 missing model).
- **`tests/test_pinecone_service.py`:** Tests vector search querying, local fallback search, and record upserting.
- **`tests/test_rag_service.py`:** Tests full RAG orchestration, grounding verification, citation extraction, empty/insufficient result handling, and empty query validation.
- **`tests/test_api.py`:** Tests FastAPI endpoints (`GET /`, `GET /api/health`, `POST /api/chat`, `POST /api/index-knowledge`).

---

## 📂 Project Structure

```
Floramind/
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore rules protecting credentials
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── data/
│   └── gardening_docs.json   # Curated gardening knowledge base
├── app/
│   ├── __init__.py
│   ├── config.py             # Secure configuration loader
│   ├── main.py               # FastAPI application & endpoints
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py        # Pydantic models (ChatRequest, ChatResponse, Citations)
│   ├── services/
│   │   ├── __init__.py
│   │   ├── embedding_service.py # Hugging Face sentence-transformers
│   │   ├── pinecone_service.py  # Pinecone index integration & fallback
│   │   ├── gemini_service.py    # Dedicated Google Gemini LLM module
│   │   └── rag_service.py       # RAG orchestrator & source attribution
│   ├── static/
│   │   ├── css/style.css     # Botanical aesthetic styling
│   │   └── js/chat.js        # Frontend chat & citation logic
│   └── templates/
│       └── index.html        # Modern Bootstrap 5 single-page interface
├── scripts/
│   └── index_knowledge.py   # CLI script to embed and index documents
└── tests/
    ├── __init__.py
    ├── test_gemini_service.py
    ├── test_pinecone_service.py
    ├── test_rag_service.py
    └── test_api.py
```
