from langchain_community.llms import Ollama
from .config import settings

def create_gen_llm():
    llm = Ollama(
        model=settings.ollama_model_name,
        temperature=settings.llm_temperature,
    )
    return llm

def create_judge_llm():
    """LLM used for LLM-as-judge evaluation."""
    return Ollama(
        model=settings.judge_model_name,
        temperature=settings.judge_temperature,
    )