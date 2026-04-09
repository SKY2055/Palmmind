"""Chat router - handles conversational RAG and interview bookings."""
import uuid
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List
from sqlalchemy.orm import Session

from app.config import get_settings
from app.services import (
    get_chat_memory,
    get_rag_service,
    get_booking_service,
    InterviewBooking,
    init_db,
    get_session_factory,
)

settings = get_settings()

# Initialize database
engine = init_db(settings.DATABASE_URL)
SessionLocal = get_session_factory(engine)

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


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


@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    """
    Conversational RAG endpoint with multi-turn support.
    Creates a new session if session_id is not provided.
    Automatically extracts interview booking information.
    """
    # Generate session ID if not provided
    session_id = request.session_id or str(uuid.uuid4())
    
    # Get services
    rag_service = get_rag_service()
    chat_memory = get_chat_memory()
    booking_service = get_booking_service()
    
    try:
        # Get chat history for context
        chat_history = chat_memory.get_formatted_history(session_id)
        
        # Retrieve document context for booking extraction (to find email)
        rag_service_temp = get_rag_service()
        results = rag_service_temp.retrieve_context(request.message)
        document_context = rag_service_temp.format_context(results)
        
        # Check if this is a booking request and extract info
        booking_info = booking_service.extract_booking_info(request.message, chat_history, document_context)
        booking_data = None
        is_booking_request = False
        
        if booking_info:
            # Store booking in database
            booking = booking_service.create_booking(db, session_id, booking_info)
            booking_data = {
                "id": booking.id,
                "name": booking.name,
                "email": booking.email,
                "date": booking.date,
                "time": booking.time,
                "status": booking.status
            }
            is_booking_request = True
        
        # Get RAG response (pass booking info if available)
        booking_context = None
        if booking_data and is_booking_request:
            booking_context = f"User is requesting to book an interview for {booking_data['name']} on {booking_data['date']} at {booking_data['time']}. The booking has been successfully created."
        
        result = rag_service.chat(session_id, request.message, booking_context=booking_context)
        
        # If booking was extracted, add confirmation to response
        # Only append if the LLM response acknowledges the booking (not contradictory)
        if booking_data and is_booking_request:
            response_lower = result["response"].lower()
            # Only append if response mentions booking, interview, scheduled, or doesn't say "don't have information"
            if any(keyword in response_lower for keyword in ["booked", "scheduled", "confirmed", "interview", "book"]) and "don't have" not in response_lower:
                result["response"] += f"\n\n✅ Interview booked successfully!\n📅 Date: {booking_data['date']}\n🕐 Time: {booking_data['time']}\n📧 Confirmation sent to: {booking_data['email']}"
        
        return ChatResponse(
            response=result["response"],
            session_id=session_id,
            context_used=result["context_used"],
            sources=result.get("sources", []),
            booking_extracted=booking_data,
            provider_used=result.get("provider_used")
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat: {str(e)}"
        )


@router.get("/history/{session_id}", response_model=ChatHistoryResponse)
async def get_chat_history(session_id: str):
    """Get chat history for a session."""
    chat_memory = get_chat_memory()
    messages = chat_memory.get_chat_history(session_id)
    
    return ChatHistoryResponse(
        session_id=session_id,
        messages=messages
    )


@router.delete("/history/{session_id}")
async def clear_chat_history(session_id: str):
    """Clear chat history for a session."""
    chat_memory = get_chat_memory()
    chat_memory.clear_history(session_id)
    return {"message": f"Chat history cleared for session {session_id}"}
