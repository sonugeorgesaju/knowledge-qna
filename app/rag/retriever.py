import os
import faiss
import numpy as np
from app.config import FAISS_INDEX_PATH, TOP_K
from app.rag.embeddings import get_embedding

def load_index():
    index = faiss.read_index(os.path.join(FAISS_INDEX_PATH, "index.faiss"))
    with open(os.path.join(FAISS_INDEX_PATH, "chunks.txt"), "r", encoding="utf-8") as f:
        content = f.read()
    chunks = content.split("\n---CHUNK---\n")
    chunks = [c for c in chunks if c.strip()]
    return index, chunks

def retrieve(question: str) -> list[dict]:
    index, chunks = load_index()
    question_vector = np.array([get_embedding(question)], dtype="float32")
    scores, indices = index.search(question_vector, TOP_K)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(chunks):
            results.append({"content": chunks[idx], "score": float(score)})
    return results
