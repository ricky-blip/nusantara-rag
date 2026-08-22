from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from app.schemas import QueryRequest, QueryResponse
from app.services.rag import rag_pipeline
from app.services.agent import agent

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- build vector index from doc ---")
    rag_pipeline.build_index()
    yield

app = FastAPI(
    title="NusantaraCare AI Assistant API",
    description="RAG + Agentic assistant untuk Panduan Operasional Internal v2.0",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/", tags=["Health"])
async def health_check():
    return {"status": "healthy", "service": "NusantaraCare RAG API",
            "chunks": len(rag_pipeline.chunks)}

@app.post("/api/v1/query", response_model=QueryResponse, tags=["RAG"])
async def query_document(request: QueryRequest):
    try:
        return QueryResponse(**await agent.process_query(request.question))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))