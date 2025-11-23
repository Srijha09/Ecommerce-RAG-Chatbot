import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from .schemas import ChatRequest, ChatResponse, SourceSnippet
from rag.pipeline import RAGPipeline
from rag.evaluator import InlineLLMJudge

from monitoring.metrics import (
    RAG_REQUESTS,
    RAG_ERRORS,
    RAG_LATENCY,
    RAG_RETRIEVED_CHUNKS,
)

app = FastAPI(title="Everstorm RAG Chatbot")
rag_pipeline = RAGPipeline.from_index()
judge = InlineLLMJudge(max_cycles=3)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest):
    endpoint = "/chat"
    RAG_REQUESTS.labels(endpoint=endpoint).inc()
    start = time.perf_counter()

    try:
        result = judge.evaluate_answer(req.question)
        answer = result["answer"]

        meta = {
            "label": result["label"],
            "cycles": result["cycles"],
            "sources": result["sources"]
        }

    except Exception as e:
        RAG_ERRORS.labels(endpoint=endpoint).inc()
        raise HTTPException(status_code=500, detail=str(e))

    duration = time.perf_counter() - start
    RAG_LATENCY.labels(endpoint=endpoint).observe(duration)
    RAG_RETRIEVED_CHUNKS.observe(len(meta["sources"]))

    return {
        "answer": answer,
        "judge_label": meta["label"],
        "judge_cycles": meta["cycles"],
        "sources": meta["sources"]
    }

@app.get("/metrics")
def metrics():
    data = generate_latest()
    return PlainTextResponse(data.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)