"""Schemas package."""
from app.schemas.document import UploadResponse, ChunkingInfo
from app.schemas.chat import ChatRequest, ChatResponse, ChatHistoryResponse
from app.schemas.booking import BookingResponse

__all__ = [
    "UploadResponse",
    "ChunkingInfo",
    "ChatRequest",
    "ChatResponse",
    "ChatHistoryResponse",
    "BookingResponse",
]
