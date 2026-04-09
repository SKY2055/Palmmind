import uuid
import re
import json
from typing import Dict, Optional
from datetime import datetime
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import InterviewBooking
from app.services.llm_client import get_llm_client, reset_llm_client

settings = get_settings()

# Force reset LLM client on module load to pick up config changes
reset_llm_client()


class BookingService:
    def __init__(self):
        self.llm_client = get_llm_client(settings)
    
    def extract_booking_info(self, text: str, chat_history: str = "", document_context: str = "") -> Optional[Dict]:
        """Extract interview booking details from text using LLM with fallback."""
        booking_keywords = ["book", "schedule", "interview", "appointment", "reserve"]
        text_lower = text.lower()
        
        print(f"[Booking] Checking for booking keywords in: {text_lower[:100]}...")
        print(f"[Booking] Keywords found: {[kw for kw in booking_keywords if kw in text_lower]}")
        
        if not any(keyword in text_lower for keyword in booking_keywords):
            print(f"[Booking] No booking keywords found, skipping extraction")
            return None
        
        context_section = ""
        if document_context:
            context_section = f"\n\nDocument Context (use this to find email if not in message):\n{document_context}"
        
        prompt = f"""Extract interview booking information from the following text. Return ONLY a JSON object with these fields: name, email, date (YYYY-MM-DD format), time (HH:MM format). If any field is missing, use null.

Chat history:
{chat_history}
{context_section}

Current message:
{text}

Respond with ONLY a JSON object like:
{{"name": "John Doe", "email": "john@example.com", "date": "2024-12-25", "time": "14:30"}}

If this is NOT a booking request, return: {{"is_booking": false}}"""

        try:
            print(f"[Booking] Attempting LLM extraction...")
            system_prompt = "You extract booking information. Respond only with valid JSON."
            result = self.llm_client.generate(prompt, system_prompt)
            
            if not result["success"]:
                print(f"[Booking] All LLM providers failed: {result.get('errors', [])}")
                return self._extract_with_regex(text)
            
            content = result["text"].strip()
            print(f"[Booking] LLM response: {content[:200]}...")
            
            try:
                data = json.loads(content)
                print(f"[Booking] Parsed JSON: {data}")
                if data.get("is_booking") is False:
                    print(f"[Booking] LLM returned is_booking: false")
                    return None
                
                if all(data.get(field) for field in ["name", "email", "date", "time"]):
                    print(f"[Booking] All fields present, returning booking data")
                    return {
                        "name": data["name"],
                        "email": data["email"],
                        "date": data["date"],
                        "time": data["time"]
                    }
                print(f"[Booking] Missing required fields, returning None")
                return None
            except json.JSONDecodeError as e:
                print(f"[Booking] JSON decode error: {e}, trying regex fallback")
                return self._extract_with_regex(text)
                
        except Exception as e:
            print(f"[Booking] Error extracting booking info: {e}")
            return self._extract_with_regex(text)
    
    def _extract_with_regex(self, text: str) -> Optional[Dict]:
        """Fallback regex extraction for booking details."""
        info = {}
        
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
        if email_match:
            info["email"] = email_match.group(0)
        
        date_patterns = [
            r'(\d{4}-\d{2}-\d{2})',
            r'(\d{2}/\d{2}/\d{4})',
            r'(\d{2}-\d{2}-\d{4})',
        ]
        for pattern in date_patterns:
            date_match = re.search(pattern, text)
            if date_match:
                info["date"] = date_match.group(1)
                break
        
        time_match = re.search(r'(\d{1,2}:\d{2})', text)
        if time_match:
            info["time"] = time_match.group(1)
        
        name_patterns = [
            r'[Mm]y name is ([A-Z][a-z]+ [A-Z][a-z]+)',
            r'[Ii] am ([A-Z][a-z]+ [A-Z][a-z]+)',
            r'[Nn]ame[\s:]+([A-Z][a-z]+ [A-Z][a-z]+)',
        ]
        for pattern in name_patterns:
            name_match = re.search(pattern, text)
            if name_match:
                info["name"] = name_match.group(1)
                break
        
        if all(k in info for k in ["name", "email", "date", "time"]):
            return info
        return None
    
    def create_booking(self, db: Session, session_id: str, booking_info: Dict) -> InterviewBooking:
        """Store booking in database."""
        booking = InterviewBooking(
            id=str(uuid.uuid4()),
            session_id=session_id,
            name=booking_info["name"],
            email=booking_info["email"],
            date=booking_info["date"],
            time=booking_info["time"],
            status="confirmed",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        db.add(booking)
        db.commit()
        db.refresh(booking)
        return booking
    
    def get_bookings(self, db: Session, session_id: str = None, email: str = None) -> list:
        """Get bookings by session or email."""
        query = db.query(InterviewBooking)
        if session_id:
            query = query.filter(InterviewBooking.session_id == session_id)
            print(f"[Booking] Querying by session_id: {session_id}")
        if email:
            query = query.filter(InterviewBooking.email == email)
            print(f"[Booking] Querying by email: {email}")
        results = query.all()
        print(f"[Booking] Found {len(results)} bookings")
        return results
    
    def cancel_booking(self, db: Session, booking_id: str) -> bool:
        """Cancel a booking."""
        booking = db.query(InterviewBooking).filter(InterviewBooking.id == booking_id).first()
        if booking:
            booking.status = "cancelled"
            booking.updated_at = datetime.utcnow()
            db.commit()
            return True
        return False


# Singleton
_booking_service = None


def get_booking_service() -> BookingService:
    global _booking_service
    if _booking_service is None:
        _booking_service = BookingService()
    return _booking_service
