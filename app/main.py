# app/main.py
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from .schemas import ChatRequest, ChatResponse, SourceSnippet
from rag.pipeline import RAGPipeline
from monitoring.metrics import (
    RAG_REQUESTS,
    RAG_ERRORS,
    RAG_LATENCY,
    RAG_RETRIEVED_CHUNKS,
)

app = FastAPI(title="Everstorm RAG Chatbot")

# Initialize RAG pipeline once (load index, build retriever, etc.)
rag_pipeline = RAGPipeline.from_index()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    endpoint = "/chat"
    RAG_REQUESTS.labels(endpoint=endpoint).inc()
    start = time.perf_counter()

    try:
        answer, meta = rag_pipeline.ask(
            question=req.question,
            chat_history=req.history or [],
            top_k=req.top_k,
            temperature=req.temperature,
        )
    except Exception as e:
        RAG_ERRORS.labels(endpoint=endpoint).inc()
        raise HTTPException(status_code=500, detail=str(e))

    duration = time.perf_counter() - start
    RAG_LATENCY.labels(endpoint=endpoint).observe(duration)

    num_sources = int(meta.get("num_sources", 0))
    RAG_RETRIEVED_CHUNKS.observe(num_sources)

    sources = [SourceSnippet(**s) for s in meta.get("sources", [])]

    return ChatResponse(
        answer=answer,
        num_sources=num_sources,
        sources=sources,
    )

@app.get("/metrics")
def metrics():
    data = generate_latest()
    return PlainTextResponse(data.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)