# Ecommerce-RAG-Chatbot

A production-style **Retrieval-Augmented Generation (RAG)** chatbot that answers customer support questions for **Everstorm Outfitters** using their internal policy PDFs.

The system uses:

- **FAISS** vector store for retrieval  
- **Ollama** LLM for answer generation  
- **A separate Ollama LLM as a judge** to evaluate answers (CORRECT / HALLUCINATION / INCOMPLETE)  
- **FastAPI** backend + **Streamlit** UI  
- **Prometheus-compatible metrics** for basic monitoring  

---

## 🎯 Objective

Build an end-to-end RAG system that:

1. Ingests Everstorm policy documents (PDFs)  
2. Chunks and embeds them into a FAISS vector index  
3. Retrieves relevant chunks for each user question  
4. Generates grounded answers with an LLM  
5. Uses **LLM-as-judge** to sanity-check answers and surface evaluation in the UI  
6. Exposes basic metrics for observability  

Target use case: **Customer support** for common questions like returns, refunds, shipping, product care, etc.

---

## 🧱 Project Structure

```text
.
├── app/
│   ├── main.py              # FastAPI app: /chat, /health, /metrics
│   └── schemas.py           # Pydantic request/response models
│
├── rag/
│   ├── config.py            # Settings (paths, model names, hyperparams)
│   ├── data_loader.py       # PDF loading + metadata (source, page)
│   ├── chunker.py           # Text splitting / chunking
│   ├── vectorstore.py       # FAISS index build/load
│   ├── llm.py               # LLM factories (generator + judge)
│   ├── pipeline.py          # RAGPipeline.ask() – main RAG logic
│   ├── evaluator.py         # Inline LLM-as-judge (CORRECT / HALLUCINATION / INCOMPLETE)
│   └── eval_metrics.py      # Classic metrics (BLEU / ROUGE / etc., optional)
│
├── ui/
│   └── app.py               # Streamlit UI (chat + retrieved context + LLM evaluation)
│
├── monitoring/
│   └── metrics.py           # Prometheus metrics: latency, errors, retrieved chunks
│
├── scripts/
│   └── build_index.py       # Offline script to build the FAISS index from PDFs
│
├── data/
│   ├── *.pdf                # Everstorm policy PDFs (input)
│   └── faiss_index/         # Saved FAISS index (output)
│
├── everstorm_eval_dataset.jsonl  # Optional: eval dataset for offline testing



flowchart LR
    subgraph UI["Streamlit UI (ui/app.py)"]
        U[User] -->|Question| ST[Chat Input]
        ST -->|POST /chat| API
    end

    subgraph Backend["FastAPI Backend (app/main.py)"]
        API[POST /chat] --> PIPE[RAGPipeline.ask()]
        API --> METRICS[Prometheus Metrics]
    end

    subgraph RAG["RAG Core (rag/pipeline.py)"]
        PIPE --> RETRIEVE[FAISS Vector Store]
        RETRIEVE --> DOCS[Top-k Chunks]
        DOCS --> PROMPT[Prompt Builder]
        PROMPT --> GEN_LLM[Generator LLM\n(Ollama)]
        GEN_LLM --> ANSWER[Answer]
    end

    subgraph Store["Index & Data"]
        PDF[Everstorm PDFs] --> LOADER[PDF Loader\n+ Chunker]
        LOADER --> INDEX[FAISS Index]
        INDEX -.-> RETRIEVE
    end

    subgraph Judge["LLM-as-Judge (rag/evaluator.py)"]
        ANSWER --> JPROMPT[Judge Prompt\n(CONTEXT + ANSWER)]
        DOCS --> JPROMPT
        JPROMPT --> JLLM[Judge LLM\n(Ollama, smaller)]
        JLLM --> JL[Label:\nCORRECT / HALLUCINATION / INCOMPLETE]
    end

    ANSWER --> API
    JL --> API

    API -->|JSON: answer + sources + judge_label| UI
    UI -->|Show answer, context, LLM eval| U

├── requirements.txt
└── README.md
