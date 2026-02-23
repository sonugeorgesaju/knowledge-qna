# knowledge-qna

A RAG (Retrieval-Augmented Generation) microservice that lets you query your own documents using natural language.

## What it does

- Ingests `.txt` documents into a FAISS vector index
- Accepts natural language questions via API
- Retrieves the most relevant chunks and passes them to an LLM
- Returns a structured JSON answer with source references

## Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | / | No | Health check |
| POST | /ingest | Yes | Ingest documents from data/documents/ |
| POST | /query | Yes | Ask a question, get a structured answer |

## Stack

- FastAPI + Uvicorn
- FAISS (vector similarity search)
- OpenAI or Ollama — switchable via `.env`, zero code changes required
- Pydantic v2 (structured outputs)
- Python 3.11+

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/sonugeorgesaju/knowledge-qna.git
cd knowledge-qna

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your values
```

## Environment variables

```
API_KEY=your-secret-key

LLM_PROVIDER=openai
EMBEDDING_PROVIDER=openai

OPENAI_API_KEY=your-openai-key
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini

OLLAMA_BASE_URL=http://localhost:11434

FAISS_INDEX_PATH=data/faiss_index
DOCUMENTS_PATH=data/documents
TOP_K=5
```

To use Ollama instead of OpenAI, set `LLM_PROVIDER=ollama` and `EMBEDDING_PROVIDER=ollama`. No other changes needed.

## Run

```bash
uvicorn app.main:app --reload
```

## Run with Docker

```bash
# 1. Build the image
docker build -t knowledge-qna .

# 2. Run the container
docker run -d \
  --name knowledge-qna \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/data/documents:/app/data/documents \
  -v $(pwd)/data/faiss_index:/app/data/faiss_index \
  knowledge-qna
```

Documents and the FAISS index are mounted as volumes so data persists across container restarts.

## Usage

**Ingest documents:**
```bash
# Place .txt files in data/documents/, then:
curl -X POST http://127.0.0.1:8000/ingest \
  -H "X-API-Key: your-secret-key"
```

**Query:**
```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "X-API-Key: your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund policy?"}'
```

**Response:**
```json
{
  "answer": "The refund policy allows returns within 30 days...",
  "sources": [
    {"content": "Returns are accepted within 30 days of purchase...", "score": 0.87}
  ]
}
```

## Project structure

```
app/
├── main.py              — FastAPI app, route definitions
├── auth.py              — X-API-Key header validation
├── config.py            — reads .env, exposes all settings
├── schemas.py           — Pydantic request/response models
├── rag/
│   ├── embeddings.py    — provider abstraction (OpenAI or Ollama)
│   ├── ingest.py        — load, chunk, embed, save FAISS index
│   └── retriever.py     — embed query, search FAISS, return top-K chunks
└── services/
    └── qa_service.py    — chunks + question → LLM → answer
```
