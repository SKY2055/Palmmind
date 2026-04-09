from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # App
    APP_NAME: str = "PalmMind RAG"
    DEBUG: bool = False
    
    # File Upload
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS: set = {".pdf", ".txt"}
    UPLOAD_DIR: str = "./uploads"
    
    # Vector DB 
    VECTOR_DB_PROVIDER: str = "qdrant"
    
    # Qdrant
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION_NAME: str = "documents"
    
    # Embedding Model
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    
    # Chunking 
    CHUNK_SIZE: int = 1024
    CHUNK_OVERLAP: int = 100
    
    # Database
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/palmmind_db"
    
    # Redis (Chat Memory)
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_CHAT_TTL: int = 3600  
    
    # LLM Provider Priority 
    LLM_PROVIDER_PRIORITY: str = "groq,gemini,deepseek"
    
    # Groq (LLM)
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    GROQ_TEMPERATURE: float = 0.7
    
    # Gemini (Google AI)
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-1.5-flash"
    GEMINI_TEMPERATURE: float = 0.7
    
    # DeepSeek
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_MODEL: str = "deepseek-chat"
    DEEPSEEK_TEMPERATURE: float = 0.7
    
    # RAG Settings
    RAG_TOP_K: int = 8  # Number of chunks to retrieve 
    MAX_CHAT_HISTORY: int = 10  # Messages to keep in context
    
    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
