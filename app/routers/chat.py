"""Chat router — conversational RAG + interview booking."""

import re
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
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
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    """Conversational RAG with multi-turn support, disambiguation, and booking."""

    session_id = request.session_id or str(uuid.uuid4())
    rag_service     = get_rag_service()
    chat_memory     = get_chat_memory()
    booking_service = get_booking_service()

    try:
        target_doc_id: Optional[str] = None
        session_meta = chat_memory.get_metadata(session_id)

        if session_meta and session_meta.get("state") == "awaiting_disambiguation":
            candidates = session_meta.get("candidates", [])
            msg_lower  = request.message.lower()

            idx_match = re.search(r"\b([1-9])\b", msg_lower)
            if idx_match:
                idx = int(idx_match.group(1)) - 1
                if 0 <= idx < len(candidates):
                    target_doc_id = candidates[idx]["document_id"]
                    print(f"[Chat] Disambiguation by index {idx + 1}: {target_doc_id}")

            if not target_doc_id:
                for cand in candidates:
                    if cand.get("email", "").lower() in msg_lower:
                        target_doc_id = cand["document_id"]
                        print(f"[Chat] Disambiguation by email: {target_doc_id}")
                        break

            if target_doc_id:
                chat_memory.clear_metadata(session_id)

        retrieval_data = rag_service.retrieve_context(
            request.message, target_doc_id=target_doc_id
        )
        document_context = ""
        if not retrieval_data.get("ambiguous") and retrieval_data["results"]:
            document_context, _ = rag_service.format_context(
                retrieval_data["results"], request.message
            )
        elif not retrieval_data.get("ambiguous") and not retrieval_data["results"]:
            # No documents found - check if query contains a name
            query_tokens = [
                w for w in re.findall(r"[a-zA-Z]+", request.message.lower())
                if len(w) > 2 and w not in {"the", "and", "for", "what", "who", "are", "his", "her", "tell", "about", "give", "me"}
            ]
            if query_tokens and len(query_tokens) >= 2:
                # Likely asking about a specific person who doesn't exist
                return ChatResponse(
                    response="Applicant not found in the uploaded documents. Please upload the applicant's resume or check the name/email.",
                    session_id=session_id,
                    context_used=False,
                    sources=[],
                    booking_extracted=None,
                    provider_used=None,
                )

        chat_history = chat_memory.get_formatted_history(session_id)
        booking_info = booking_service.extract_booking_info(
            request.message, chat_history, document_context
        )

        booking_data:    Optional[dict] = None
        booking_context: Optional[str]  = None

        if booking_info is not None:
            missing = booking_info.get("missing_fields", [])
            if not missing:
                booking = booking_service.create_booking(db, session_id, booking_info)
                booking_data = {
                    "id":     booking.id,
                    "name":   booking.name,
                    "email":  booking.email,
                    "date":   booking.date,
                    "time":   booking.time,
                    "status": booking.status,
                }
                booking_context = (
                    f"The interview for {booking_data['name']} has been successfully "
                    f"booked for {booking_data['date']} at {booking_data['time']}."
                )
            else:
                missing_str = ", ".join(missing)
                booking_context = (
                    f"The user wants to book an interview but these details are missing: "
                    f"{missing_str}. Please ask the user for these details."
                )

        result = rag_service.chat(
            session_id,
            request.message,
            target_doc_id=target_doc_id,
            booking_context=booking_context,
        )

        if result.get("ambiguous"):
            candidates = result["candidates"]
            cand_list = "\n".join(
                f"{i + 1}. {c['name']} ({c['email']})"
                for i, c in enumerate(candidates)
            )
            result["response"] = (
                "I found multiple applicants with that name. "
                "Which one are you referring to?\n\n"
                f"{cand_list}\n\n"
                "Please reply with the number or the email of the correct applicant."
            )
            chat_memory.set_metadata(session_id, {
                "state":      "awaiting_disambiguation",
                "candidates": candidates,
            })
        elif result.get("sources") and len(result["sources"]) > 0:
            # Single match - show which applicant was found
            doc_id = result["sources"][0]
            # Get the candidate info from the retrieval data
            candidate_info = next(
                (c for c in retrieval_data.get("candidates", []) if c.get("document_id") == doc_id),
                None
            )
            if candidate_info:
                result["response"] = (
                    f"Found applicant: {candidate_info.get('name', 'Unknown')} "
                    f"({candidate_info.get('email', 'No email')})\n\n"
                    f"{result['response']}"
                )

        if booking_data and not result.get("ambiguous"):
            resp_lower = result["response"].lower()
            confirmation_keywords = {"booked", "scheduled", "confirmed", "interview"}
            if not any(kw in resp_lower for kw in confirmation_keywords):
                result["response"] += (
                    f"\n\n✅ Interview booked successfully!\n"
                    f"📅 Date: {booking_data['date']}\n"
                    f"🕐 Time: {booking_data['time']}\n"
                    f"📧 Confirmation for: {booking_data['email']}"
                )

        return ChatResponse(
            response=result["response"],
            session_id=session_id,
            context_used=result.get("context_used", False),
            sources=result.get("sources", []),
            booking_extracted=booking_data,
            provider_used=result.get("provider"),
        )

    except Exception as exc:
        print(f"[Chat] Unhandled error: {exc}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing chat: {exc}")


@router.get("/history/{session_id}", response_model=ChatHistoryResponse)
async def get_chat_history(session_id: str):
    """Return full chat history for a session."""
    chat_memory = get_chat_memory()
    messages    = chat_memory.get_chat_history(session_id)
    return ChatHistoryResponse(session_id=session_id, messages=messages)


@router.delete("/history/{session_id}")
async def clear_chat_history(session_id: str):
    """Clear chat history for a session."""
    get_chat_memory().clear_history(session_id)
    return {"message": f"Chat history cleared for session {session_id}"}