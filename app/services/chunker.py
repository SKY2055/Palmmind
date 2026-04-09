from typing import List
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)


class ChunkingStrategy:
    FIXED = "fixed"
    SEMANTIC = "semantic"


class TextChunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def fixed_chunking(self, text: str) -> List[str]:
        """Fixed-size chunking with overlap."""
        splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        return splitter.split_text(text)
    
    def semantic_chunking(self, text: str) -> List[str]:
        """Semantic chunking using recursive character splitting."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        )
        return splitter.split_text(text)
    
    def chunk(self, text: str, strategy: str) -> List[str]:
        """Apply selected chunking strategy."""
        if strategy == ChunkingStrategy.FIXED:
            return self.fixed_chunking(text)
        elif strategy == ChunkingStrategy.SEMANTIC:
            return self.semantic_chunking(text)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
