from pydantic import BaseModel
from typing import Literal, List

class QueryRequest(BaseModel):
    question: str

class SourceCitation(BaseModel):
    chunk_id: str
    section_title: str
    doc_version: str
    score: float

class QueryResponse(BaseModel):
    answer: str
    confidence_label: Literal["high", "medium", "low"]
    reason_code: str
    sources: List[SourceCitation] = []