from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from .config import settings

def load_policy_pdfs() -> List[Document]:
    data_dir: Path = settings.data_dir
    docs: List[Document] = []

    pdf_paths = list(data_dir.glob("*.pdf"))  
    print(f"[data_loader] Looking for PDFs in: {data_dir.resolve()}")
    print(f"[data_loader] Found {len(pdf_paths)} PDF files: {pdf_paths}")

    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        loaded = loader.load()
        print(f"[data_loader] Loaded {len(loaded)} pages from {pdf_path.name}")
        docs.extend(loaded)

    print(f"[data_loader] Total pages loaded: {len(docs)}")
    return docs