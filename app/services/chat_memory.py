import json
import redis
from typing import List, Dict, Optional
from datetime import datetime
from app.config import get_settings

settings = get_settings()


class ChatMemoryService:
    def __init__(self):
        self.redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        self.ttl = settings.REDIS_CHAT_TTL
    
    def _get_key(self, session_id: str) -> str:
        return f"chat:{session_id}"
    
    def get_chat_history(self, session_id: str, limit: int = None) -> List[Dict]:
        """Retrieve chat history from Redis."""
        key = self._get_key(session_id)
        if not self.redis_client.exists(key):
            return []
        
        # Get all messages (stored as JSON strings in a list)
        messages = self.redis_client.lrange(key, 0, -1)
        history = [json.loads(msg) for msg in messages]
        
        # Apply limit if specified
        if limit and len(history) > limit:
            history = history[-limit:]
        
        return history
    
    def add_message(self, session_id: str, role: str, content: str, metadata: Dict = None):
        """Add a message to chat history."""
        key = self._get_key(session_id)
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        
        # Add to list and set TTL
        self.redis_client.rpush(key, json.dumps(message))
        self.redis_client.expire(key, self.ttl)
        
        # Trim to max history size
        max_history = settings.MAX_CHAT_HISTORY
        current_len = self.redis_client.llen(key)
        if current_len > max_history:
            self.redis_client.ltrim(key, current_len - max_history, -1)
    
    def clear_history(self, session_id: str):
        """Clear chat history for a session."""
        key = self._get_key(session_id)
        self.redis_client.delete(key)
    
    def get_formatted_history(self, session_id: str, limit: int = None) -> str:
        """Get chat history formatted for LLM prompt."""
        history = self.get_chat_history(session_id, limit)
        formatted = []
        for msg in history:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
        return "\n".join(formatted)

    def _get_metadata_key(self, session_id: str) -> str:
        return f"meta:{session_id}"

    def set_metadata(self, session_id: str, data: Dict):
        """Store metadata for a session (e.g., last found candidates)."""
        key = self._get_metadata_key(session_id)
        self.redis_client.set(key, json.dumps(data), ex=self.ttl)

    def get_metadata(self, session_id: str) -> Optional[Dict]:
        """Retrieve metadata for a session."""
        key = self._get_metadata_key(session_id)
        data = self.redis_client.get(key)
        if data:
            return json.loads(data)
        return None

    def clear_metadata(self, session_id: str):
        """Clear metadata for a session."""
        key = self._get_metadata_key(session_id)
        self.redis_client.delete(key)


# Singleton instance
_chat_memory = None


def get_chat_memory() -> ChatMemoryService:
    global _chat_memory
    if _chat_memory is None:
        _chat_memory = ChatMemoryService()
    return _chat_memory
