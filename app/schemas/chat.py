"""Chat schemas (Pydantic models)."""
from pydantic import BaseModel, Field
from typing import Optional, List


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        description="Your question about the uploaded documents",
        examples=["What is my CGPA in B.Tech?"]
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for continuing a conversation (optional)",
        examples=["550e8400-e29b-41d4-a716-446655440000"]
    )


class ChatResponse(BaseModel):
    response: str
    session_id: str
    context_used: bool
    sources: List[str] = []
    booking_extracted: Optional[dict] = None
    provider_used: Optional[str] = None


class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: List[dict]
