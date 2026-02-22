from pydantic import BaseModel
from typing import List

class QueryRequest(BaseModel):
    question: str

class SourceChunk(BaseModel):
    content: str
    score: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
