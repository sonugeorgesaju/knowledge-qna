import os
import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import DOCUMENTS_PATH, FAISS_INDEX_PATH
from app.rag.embeddings import get_embedding

def load_documents() -> list[str]:
    chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    for filename in os.listdir(DOCUMENTS_PATH):
        if filename.endswith(".txt"):
            filepath = os.path.join(DOCUMENTS_PATH, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            chunks.extend(splitter.split_text(text))
    return chunks

def build_index(chunks: list[str]):
    embeddings = [get_embedding(chunk) for chunk in chunks]
    vectors = np.array(embeddings, dtype="float32")
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index

def save_index(index, chunks: list[str]):
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    faiss.write_index(index, os.path.join(FAISS_INDEX_PATH, "index.faiss"))
    with open(os.path.join(FAISS_INDEX_PATH, "chunks.txt"), "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk + "\n---CHUNK---\n")

def run_ingest():
    chunks = load_documents()
    index = build_index(chunks)
    save_index(index, chunks)
    return len(chunks)
