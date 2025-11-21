from typing import List
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from .config import settings

def _make_embeddings():
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model_name,
        model_kwargs={"device": "cpu"},      # tweak if you have GPU
        encode_kwargs={"normalize_embeddings": True},
    )

def build_vectorstore(chunks: List[Document]) -> FAISS:
    embeddings = _make_embeddings()
    vs = FAISS.from_documents(chunks, embeddings)
    return vs

def save_vectorstore(vs: FAISS, index_path: Path | None = None) -> None:
    index_path = index_path or settings.index_path
    index_path.parent.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(index_path))

def load_vectorstore(index_path: Path | None = None) -> FAISS:
    index_path = index_path or settings.index_path
    embeddings = _make_embeddings()
    vs = FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)
    return vs

def get_retriever(vs: FAISS):
    return vs.as_retriever(search_kwargs={"k": settings.top_k})