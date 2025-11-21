from typing import List, Optional, Tuple
from pydantic import BaseModel

class ChatRequest(BaseModel):
    question: str
    history: Optional[List[Tuple[str, str]]] = None  # [(user, bot), ...]
    top_k: Optional[int] = 5
    temperature: Optional[float] = 0.1

class SourceSnippet(BaseModel):
    page_content: str
    metadata: dict

class ChatResponse(BaseModel):
    answer: str
    num_sources: int
    sources: List[SourceSnippet]