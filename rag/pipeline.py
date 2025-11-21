from typing import List, Dict, Any
from langchain_core.documents import Document
from .vectorstore import load_vectorstore, get_retriever
from .llm import create_llm
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
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    @classmethod
    def from_index(cls) -> "RAGPipeline":
        vs = load_vectorstore()
        retriever = get_retriever(vs)
        llm = create_llm()
        return cls(llm=llm, retriever=retriever)

    def _build_prompt(self, question: str, docs: List[Document]) -> str:
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
        temperature: float | None = None,
    ) -> tuple[str, Dict[str, Any]]:
        chat_history = chat_history or []
        k = top_k or settings.top_k

        # override k on retriever
        self.retriever.search_kwargs["k"] = k
        docs: List[Document] = self.retriever.get_relevant_documents(question)

        prompt = self._build_prompt(question, docs)

        # per-call temperature override if provided
        llm_kwargs: Dict[str, Any] = {}
        if temperature is not None:
            llm_kwargs["temperature"] = temperature

        answer: str = self.llm.invoke(prompt, **llm_kwargs)

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