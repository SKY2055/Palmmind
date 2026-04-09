"""Interview booking models."""
from sqlalchemy import Column, String, DateTime
from datetime import datetime
from app.models.base import Base


class InterviewBooking(Base):
    __tablename__ = "interview_bookings"
    
    id = Column(String, primary_key=True)
    session_id = Column(String, nullable=False, index=True)
    name = Column(String, nullable=False)
    email = Column(String, nullable=False)
    date = Column(String, nullable=False)  # YYYY-MM-DD format
    time = Column(String, nullable=False)  # HH:MM format
    status = Column(String, default="confirmed")  # confirmed, cancelled, completed
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
