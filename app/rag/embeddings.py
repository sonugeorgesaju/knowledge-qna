from app.config import EMBEDDING_PROVIDER, EMBEDDING_MODEL, OPENAI_API_KEY, OLLAMA_BASE_URL

def get_embedding(text: str) -> list[float]:
    if EMBEDDING_PROVIDER == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
        return response.data[0].embedding

    elif EMBEDDING_PROVIDER == "ollama":
        import ollama
        client = ollama.Client(host=OLLAMA_BASE_URL)
        response = client.embeddings(model=EMBEDDING_MODEL, prompt=text)
        return response["embedding"]

    else:
        raise ValueError(f"Unknown EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER}")
