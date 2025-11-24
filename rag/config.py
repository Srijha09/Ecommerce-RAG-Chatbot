from pathlib import Path

class Settings:
    def __init__(self):
        # Where your PDFs live
        self.data_dir: Path = Path("data")
        print(f"Data directory set to: {self.data_dir}")
        self.index_path: Path = Path("data/faiss_index")

        # Embeddings
        self.embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

        # LLM (if you're using Ollama or similar)
        self.ollama_model_name: str = "gemma2:2b"
        self.llm_temperature: float = 0.1

        self.judge_model_name: str = "tinyllama:latest"
        self.judge_temperature: float = 0.01

        # Retrieval + chunking
        self.top_k: int = 5
        self.chunk_size: int = 512
        self.chunk_overlap: int = 64

# This is what other modules import
settings = Settings()