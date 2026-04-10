"""Booking service — extracts and persists interview bookings."""

import uuid
import re
import json
from typing import Dict, Optional
from datetime import datetime
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import InterviewBooking
from app.services.llm_client import get_llm_client

settings = get_settings()

_BOOKING_KEYWORDS = {"book", "schedule", "interview", "appointment", "reserve"}

_REQUIRED_FIELDS = ("name", "email", "date", "time")


class BookingService:
    def __init__(self):
        self.llm_client = get_llm_client(settings)

    def extract_booking_info(
        self,
        text: str,
        chat_history: str = "",
        document_context: str = "",
    ) -> Optional[Dict]:
        """
        Extract interview booking details from *text*.

        Returns one of:
          • None                          — not a booking request
          • {"name": …, "email": …, …, "missing_fields": []}   — complete
          • {"name": …, …, "missing_fields": ["email", …]}      — partial
        """
        text_lower = text.lower()
        if not any(kw in text_lower for kw in _BOOKING_KEYWORDS):
            return None

        context_section = (
            f"\n\nDocument Context (use to find email/name if absent from message):\n{document_context}"
            if document_context
            else ""
        )

        prompt = f"""Extract interview booking details from the conversation below.
Return ONLY a raw JSON object — no markdown fences, no explanation.

Schema:
{{
  "name":           "<full name or null>",
  "email":          "<email or null>",
  "date":           "<YYYY-MM-DD or null>",
  "time":           "<HH:MM 24h or null>",
  "missing_fields": ["<field>", ...]   // fields that are null
}}

If this is NOT a booking request at all, return exactly:
{{"is_booking": false}}

Chat history:
{chat_history}
{context_section}

Current message:
{text}

JSON:"""

        system_prompt = (
            "You extract booking information and respond ONLY with a valid JSON object. "
            "No markdown fences. No extra text."
        )

        try:
            result = self.llm_client.generate(prompt, system_prompt)

            if not result["success"]:
                print(f"[Booking] LLM failed: {result.get('errors')}; trying regex")
                return self._extract_with_regex(text)

            raw = result["text"].strip()
            print(f"[Booking] Raw LLM response: {raw[:300]}")

            # ── Strip markdown code fences if the model added them ──
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
            raw = re.sub(r"\s*```$", "", raw)
            raw = raw.strip()

            data = json.loads(raw)

            if data.get("is_booking") is False:
                return None

            missing = [f for f in _REQUIRED_FIELDS if not data.get(f)]
            return {
                "name":           data.get("name"),
                "email":          data.get("email"),
                "date":           data.get("date"),
                "time":           data.get("time"),
                "missing_fields": missing,
            }

        except json.JSONDecodeError as exc:
            print(f"[Booking] JSON parse error ({exc}); trying regex fallback")
            return self._extract_with_regex(text)
        except Exception as exc:
            print(f"[Booking] Unexpected error: {exc}; trying regex fallback")
            return self._extract_with_regex(text)

    def create_booking(
        self, db: Session, session_id: str, booking_info: Dict
    ) -> InterviewBooking:
        booking = InterviewBooking(
            id=str(uuid.uuid4()),
            session_id=session_id,
            name=booking_info["name"],
            email=booking_info["email"],
            date=booking_info["date"],
            time=booking_info["time"],
            status="confirmed",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        db.add(booking)
        db.commit()
        db.refresh(booking)
        print(f"[Booking] Created booking {booking.id} for {booking.name}")
        return booking

    def get_bookings(
        self,
        db: Session,
        session_id: Optional[str] = None,
        email: Optional[str] = None,
    ) -> list:
        query = db.query(InterviewBooking)
        if session_id:
            query = query.filter(InterviewBooking.session_id == session_id)
        if email:
            query = query.filter(InterviewBooking.email == email)
        results = query.all()
        print(f"[Booking] Found {len(results)} bookings")
        return results

    def cancel_booking(self, db: Session, booking_id: str) -> bool:
        booking = db.query(InterviewBooking).filter(InterviewBooking.id == booking_id).first()
        if booking:
            booking.status = "cancelled"
            booking.updated_at = datetime.utcnow()
            db.commit()
            return True
        return False

    def _extract_with_regex(self, text: str) -> Optional[Dict]:
        """Best-effort regex extraction when LLM fails."""
        info: Dict[str, Optional[str]] = {
            "name": None,
            "email": None,
            "date": None,
            "time": None,
        }

        # Email
        m = re.search(r"[\w\.\-]+@[\w\.\-]+\.\w+", text)
        if m:
            info["email"] = m.group(0)

        # Date — try ISO first, then common formats
        for pat in (r"\d{4}-\d{2}-\d{2}", r"\d{2}/\d{2}/\d{4}", r"\d{2}-\d{2}-\d{4}"):
            m = re.search(pat, text)
            if m:
                info["date"] = m.group(0)
                break

        # Time — HH:MM or H:MM optionally followed by am/pm
        m = re.search(r"\b(\d{1,2}:\d{2})\s*(?:am|pm)?\b", text, re.IGNORECASE)
        if m:
            info["time"] = m.group(1)

        # Name
        for pat in (
            r"(?:my name is|i am|name\s*[:\-])\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
            r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b",
        ):
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                info["name"] = m.group(1)
                break

        missing = [f for f in _REQUIRED_FIELDS if not info.get(f)]
        # Only return something useful if at least one field was found
        if len(missing) == len(_REQUIRED_FIELDS):
            return None

        return {**info, "missing_fields": missing}


_booking_service: Optional[BookingService] = None


def get_booking_service() -> BookingService:
    global _booking_service
    if _booking_service is None:
        _booking_service = BookingService()
    return _booking_service