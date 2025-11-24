from typing import List, Dict, Any
from langchain_core.documents import Document
from .vectorstore import load_vectorstore
from .llm import create_gen_llm
from .config import settings

SYSTEM_TEMPLATE = """
You are a Customer Support Chatbot for Everstorm Outfitters.
Use only the information in <context> to answer.

Rules:
1) Use ONLY the provided <context> to answer.
2) If the answer is not in the context, say:
   "I don't know based on the retrieved documents."
3) Be concise and accurate. Prefer quoting key phrases from the context.
"""

class RAGPipeline:
    def __init__(self, llm, vectorstore):
        # Store both the LLM and the FAISS vectorstore on the instance
        self.llm = llm
        self.vectorstore = vectorstore

    @classmethod
    def from_index(cls) -> "RAGPipeline":
        """
        Load the FAISS index from disk, create an LLM, and build a pipeline instance.
        """
        vs = load_vectorstore()
        llm = create_gen_llm()
        return cls(llm=llm, vectorstore=vs)

    def _build_prompt(self, question: str, docs: List[Document]) -> str:
        """
        Build a prompt using the system instructions + retrieved context + user question.
        """
        context = "\n\n".join(d.page_content for d in docs)
        prompt = f"""{SYSTEM_TEMPLATE}

<context>
{context}
</context>

User question: {question}
"""
        return prompt

    def ask(
        self,
        question: str,
        chat_history: List[tuple[str, str]] | None = None,
        top_k: int | None = None,
        temperature: float | None = None,  # kept for future use
    ) -> tuple[str, Dict[str, Any]]:
        """
        Main RAG entrypoint:
        - retrieves relevant documents from the vectorstore
        - builds a prompt
        - calls the LLM
        - returns answer + metadata (sources)
        """
        chat_history = chat_history or []
        k = top_k or settings.top_k

        # 1) Retrieve relevant documents directly from FAISS
        docs: List[Document] = self.vectorstore.similarity_search(question, k=k)

        # 2) Build prompt
        prompt = self._build_prompt(question, docs)

        # 3) Call LLM (no per-call kwargs for now to keep Ollama happy)
        raw_answer = self.llm.invoke(prompt)

        # Normalize result to string (some LLMs return Message objects)
        if hasattr(raw_answer, "content"):
            answer = raw_answer.content
        else:
            answer = str(raw_answer)

        # 4) Prepare metadata
        meta: Dict[str, Any] = {
            "num_sources": len(docs),
            "sources": [
                {
                    "page_content": d.page_content[:300],
                    "metadata": d.metadata,
                }
                for d in docs
            ],
        }
        return answer, meta