"""Routers package."""
from app.routers.documents import router as documents_router
from app.routers.chat import router as chat_router
from app.routers.bookings import router as bookings_router

__all__ = [
    "documents_router",
    "chat_router",
    "bookings_router",
]
