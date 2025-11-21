from rag.data_loader import load_policy_pdfs
from rag.chunker import chunk_documents
from rag.vectorstore import build_vectorstore, save_vectorstore

def main():
    docs = load_policy_pdfs()
    chunks = chunk_documents(docs)
    vs = build_vectorstore(chunks)
    save_vectorstore(vs)
    print("Index built and saved.")

if __name__ == "__main__":
    main()