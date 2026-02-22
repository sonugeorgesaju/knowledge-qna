from app.config import LLM_PROVIDER, LLM_MODEL, OPENAI_API_KEY, OLLAMA_BASE_URL

def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join([c["content"] for c in chunks])
    return f"""You are a helpful assistant. Answer the question using only the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}
Answer:"""

def ask_llm(question: str, chunks: list[dict]) -> str:
    prompt = build_prompt(question, chunks)

    if LLM_PROVIDER == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    elif LLM_PROVIDER == "ollama":
        import ollama
        client = ollama.Client(host=OLLAMA_BASE_URL)
        response = client.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")
