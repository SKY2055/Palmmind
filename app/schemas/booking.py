"""Booking schemas (Pydantic models)."""
from pydantic import BaseModel


class BookingResponse(BaseModel):
    id: str
    session_id: str
    name: str
    email: str
    date: str
    time: str
    status: str
    created_at: str
