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

    for pdf in data_dir.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf))
        pages = loader.load()

        for page_index, page in enumerate(pages):
            metadata = page.metadata or {}
            metadata.update({
                "source": pdf.name,
                "page_number": page_index + 1   # 1-based page numbering
            })

            docs.append(
                Document(
                    page_content=page.page_content,
                    metadata=metadata
                )
            )


    print(f"[data_loader] Total pages loaded: {len(docs)}")
    return docs