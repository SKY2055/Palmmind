"""Document models."""
from sqlalchemy import Column, String, DateTime, Integer, Text
from datetime import datetime
from app.models.base import Base


class Document(Base):
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True)
    filename = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False)  # pdf, txt
    file_size = Column(Integer, nullable=False)
    chunking_strategy = Column(String, nullable=False)  # fixed, semantic
    chunk_count = Column(Integer, default=0)
    vector_db_provider = Column(String, nullable=False)
    text_length = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    
    id = Column(String, primary_key=True)
    document_id = Column(String, nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    vector_id = Column(String, nullable=True)  # ID in vector DB
    created_at = Column(DateTime, default=datetime.utcnow)
