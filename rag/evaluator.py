from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pprint import pprint
from rag.llm import create_judge_llm
from rag.pipeline import RAGPipeline


@dataclass
class CritiqueResult:
    cycle: int
    answer: str
    critique: Optional[str]
    is_correct: bool


class InlineLLMJudge:

    def __init__(
        self,
        max_cycles: int = 1,
    ):
        self.max_cycles = max_cycles
        self.judge_llm = create_judge_llm()
        self.rag = RAGPipeline.from_index()

    def _build_critic_prompt(self, base_context: str, answer: str) -> str:
        return f"""
        You are an impartial judge. 
        Evaluate whether the assistant’s answer correctly fulfills the user’s request in context.

        Reply with exactly one of these labels—no extra text:

        CORRECT
        HALLUCINATION
        INCOMPLETE

        === CONTEXT ===
        {base_context}

        === ASSISTANT ANSWER ===
        {answer}

        LABEL:
        """

    def evaluate_answer(self, question: str):
        """
        Runs RAG → Judge multi-cycle loop.
        Returns answer + judge evaluation.
        """

        cycles: List[CritiqueResult] = []
        reflection_history = []
        halluc_count = incomplete_count = 0

        # 1. Get RAG answer + sources
        answer, meta = self.rag.ask(
            question=question,
            chat_history=[],
            top_k=None,
            temperature=None
        )
        base_context = "\n\n".join([s["page_content"] for s in meta.get("sources", [])])

        for cycle in range(1, self.max_cycles + 1):

            # 2. Build prompt
            critic_prompt = self._build_critic_prompt(base_context, answer)

            # 3. Call judge model
            feedback_raw = self.judge_llm.invoke(critic_prompt)
            feedback = str(feedback_raw).strip().upper()

            if "LABEL:" in feedback:
                feedback = feedback.split("LABEL:")[-1].strip()

            print("[Judge Feedback Raw]", feedback)

            # 4. Categorize
            is_correct = (feedback == "CORRECT") or (
                "CORRECT" in feedback and "HALLUCINATION" not in feedback and "INCOMPLETE" not in feedback
            )
            is_halluc = feedback == "HALLUCINATION"
            is_incomplete = feedback == "INCOMPLETE"

            cycles.append(
                CritiqueResult(
                    cycle=cycle,
                    answer=answer,
                    critique=None if is_correct else feedback,
                    is_correct=is_correct,
                )
            )

            if is_correct:
                return {
                    "answer": answer,
                    "label": "CORRECT",
                    "cycles": [c.__dict__ for c in cycles],
                    "sources": meta.get("sources", []),
                }

            halluc_count = halluc_count + 1 if is_halluc else 0
            incomplete_count = incomplete_count + 1 if is_incomplete else 0

            if halluc_count >= 2:
                return {
                    "answer": answer,
                    "label": "HALLUCINATION",
                    "cycles": [c.__dict__ for c in cycles],
                    "sources": meta.get("sources", []),
                }

            if incomplete_count >= 2:
                return {
                    "answer": answer,
                    "label": "INCOMPLETE",
                    "cycles": [c.__dict__ for c in cycles],
                    "sources": meta.get("sources", []),
                }

            # If continuing cycles, use judge feedback for reflection
            reflection_history.append(feedback)

        return {
            "answer": answer,
            "label": "MAX_CYCLES",
            "cycles": [c.__dict__ for c in cycles],
            "sources": meta.get("sources", []),
        }