"""Database module (backward compatibility)."""
from app.models import (
    Base,
    init_db,
    get_session_factory,
    Document,
    DocumentChunk,
    InterviewBooking,
)

__all__ = [
    "Base",
    "init_db",
    "get_session_factory",
    "Document",
    "DocumentChunk",
    "InterviewBooking",
]
