# Ecommerce-RAG-Chatbot
A production-style **Retrieval-Augmented Generation (RAG)** chatbot that answers customer support questions for **Everstorm Outfitters** using their internal policy PDFs.

The system uses:

- **FAISS** vector store for retrieval  
- **Ollama** LLM for answer generation  
- **A separate Ollama LLM as a judge** to evaluate answers (CORRECT / HALLUCINATION / INCOMPLETE)  
- **FastAPI** backend + **Streamlit** UI  
- **Prometheus-compatible metrics** for basic monitoring  


## Project Structure

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
├── requirements.txt
└── README.md
```

```mermaid
flowchart LR

    %% UI Layer
    U[User]
    UI[Streamlit UI]
    U --> UI

    %% Backend
    API[FastAPI Chat Endpoint]
    UI -->|POST chat| API

    %% RAG Pipeline
    RAG[RAG Pipeline]
    RET[Retriever]
    CHUNKS[Top K Chunks]
    PROMPT[Prompt Builder]
    GEN[Generator LLM]
    ANSWER[Answer]

    API --> RAG
    RAG --> RET
    RET --> CHUNKS
    CHUNKS --> PROMPT
    PROMPT --> GEN
    GEN --> ANSWER

    %% Index and Data
    PDF[Everstorm PDFs]
    LOADER[PDF Loader and Chunker]
    INDEX[FAISS Index]

    PDF --> LOADER --> INDEX
    INDEX --> RET

    %% Judge
    JUDGE[Judge LLM]
    JPROMPT[Judge Prompt]
    LABEL[Judge Label]

    ANSWER --> JPROMPT
    CHUNKS --> JPROMPT
    JPROMPT --> JUDGE
    JUDGE --> LABEL

    %% Return to UI
    ANSWER --> UI
    LABEL --> UI

```
## ⚙️ Prerequisites
- **Python 3.10+**  
  (Project tested on Python **3.11** and **3.13**)

- **Ollama** installed and running locally  
  https://ollama.com

- **Required models (pull via Ollama):**
  ```bash
  ollama pull llama3.1          # generator LLM
  ollama pull qwen2.5:0.5b      # judge LLM (fast & small)
## 🚀 Setup & Run

### **1. Clone the repo & create a virtual environment**

```bash
git clone <your-repo-url> Ecommerce-RAG-Chatbot
cd Ecommerce-RAG-Chatbot

python -m venv .venv