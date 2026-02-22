from dotenv import load_dotenv
import os

load_dotenv()

# Auth
API_KEY = os.getenv("API_KEY", "")

# Providers — set to "openai" or "ollama"
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")
LLM_PROVIDER       = os.getenv("LLM_PROVIDER", "openai")

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL       = os.getenv("LLM_MODEL", "gpt-4o-mini")

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Paths
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")
DOCUMENTS_PATH   = os.getenv("DOCUMENTS_PATH", "data/documents")

# RAG
TOP_K = int(os.getenv("TOP_K", "5"))
