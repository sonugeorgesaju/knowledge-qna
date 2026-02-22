from fastapi import FastAPI, Depends
from app.auth import verify_api_key
from app.schemas import QueryRequest, QueryResponse, SourceChunk
from app.rag.ingest import run_ingest
from app.rag.retriever import retrieve
from app.services.qa_service import ask_llm

app = FastAPI()

@app.get("/")
def health_check():
    return {"status": "running"}

@app.post("/ingest", dependencies=[Depends(verify_api_key)])
def ingest():
    count = run_ingest()
    return {"message": f"Ingested {count} chunks"}

@app.post("/query", dependencies=[Depends(verify_api_key)])
def query(request: QueryRequest) -> QueryResponse:
    chunks = retrieve(request.question)
    answer = ask_llm(request.question, chunks)
    sources = [SourceChunk(content=c["content"], score=c["score"]) for c in chunks]
    return QueryResponse(answer=answer, sources=sources)
