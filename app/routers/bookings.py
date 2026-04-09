"""Bookings router - handles interview booking management."""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List
from sqlalchemy.orm import Session

from app.config import get_settings
from app.services import (
    get_booking_service,
    InterviewBooking,
    init_db,
    get_session_factory,
)

settings = get_settings()

# Initialize database
engine = init_db(settings.DATABASE_URL)
SessionLocal = get_session_factory(engine)

router = APIRouter(prefix="/api/v1/bookings", tags=["bookings"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class BookingResponse(BaseModel):
    id: str
    session_id: str
    name: str
    email: str
    date: str
    time: str
    status: str
    created_at: str


@router.get("/", response_model=List[BookingResponse])
async def get_bookings(
    session_id: Optional[str] = None,
    email: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get interview bookings. Filter by session_id or email."""
    booking_service = get_booking_service()
    bookings = booking_service.get_bookings(db, session_id=session_id, email=email)
    
    return [
        BookingResponse(
            id=b.id,
            session_id=b.session_id,
            name=b.name,
            email=b.email,
            date=b.date,
            time=b.time,
            status=b.status,
            created_at=b.created_at.isoformat() if b.created_at else None
        )
        for b in bookings
    ]


@router.post("/{booking_id}/cancel")
async def cancel_booking(booking_id: str, db: Session = Depends(get_db)):
    """Cancel an interview booking."""
    booking_service = get_booking_service()
    success = booking_service.cancel_booking(db, booking_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Booking not found")
    
    return {"message": f"Booking {booking_id} cancelled successfully"}
