from langchain_community.llms import Ollama
from .config import settings

def create_llm():
    llm = Ollama(
        model=settings.ollama_model_name,
        temperature=settings.llm_temperature,
    )
    return llm