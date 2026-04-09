"""Document schemas (Pydantic models)."""
from pydantic import BaseModel, Field
from typing import Optional


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    file_type: str
    chunk_count: int
    chunking_strategy: str
    vector_db_provider: str
    message: str


class ChunkingInfo(BaseModel):
    strategies: list = Field(default=["fixed", "semantic"])
    description: dict = Field(default={
        "fixed": "Fixed-size chunking with character-based splitting",
        "semantic": "Semantic chunking with recursive character splitting (preserves paragraphs/sentences)"
    })
