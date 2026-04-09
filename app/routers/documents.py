"""Document router - handles file uploads and document processing."""
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from pydantic import BaseModel, Field
from typing import Optional, Literal
import uuid
import os
import aiofiles
from datetime import datetime
from sqlalchemy.orm import Session

from app.config import get_settings, Settings
from app.services import (
    TextExtractor,
    TextChunker,
    ChunkingStrategy,
    EmbeddingService,
    get_vector_store,
    VectorStore,
    init_db,
    get_session_factory,
    Document,
    DocumentChunk,
)

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

settings = get_settings()
engine = init_db(settings.DATABASE_URL)
SessionLocal = get_session_factory(engine)

embedding_service = None
vector_store = None
chunker = None


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_services():
    global embedding_service, vector_store, chunker
    if embedding_service is None:
        embedding_service = EmbeddingService(settings.EMBEDDING_MODEL)
    if vector_store is None:
        vector_store = get_vector_store(settings)
    if chunker is None:
        chunker = TextChunker(settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    file_type: str
    chunk_count: int
    chunking_strategy: str
    vector_db_provider: str
    message: str


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    chunking_strategy: Literal["fixed", "semantic"] = Form(default="semantic"),
    db: Session = Depends(get_db)
):
    """Upload a PDF or TXT file, extract text, chunk, embed and store."""
    init_services()
    
    file_ext = os.path.splitext(file.filename.lower())[1]
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {settings.ALLOWED_EXTENSIONS}"
        )
    
    content = await file.read()
    
    if len(content) > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {settings.MAX_FILE_SIZE / (1024*1024)}MB"
        )
    
    document_id = str(uuid.uuid4())
    
    try:
        raw_text = TextExtractor.extract(content, file_ext)
        clean_text = TextExtractor.clean_text(raw_text)
        
        if not clean_text.strip():
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from the file"
            )
        
        chunks = chunker.chunk(clean_text, chunking_strategy)
        
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Could not create chunks from the document"
            )
        
        embeddings = embedding_service.embed_texts(chunks)
        
        vector_ids = []
        vector_metadata = []
        for i, chunk_text in enumerate(chunks):
            vector_id = str(uuid.uuid4())
            vector_ids.append(vector_id)
            vector_metadata.append({
                "document_id": document_id,
                "chunk_index": i,
                "chunk_text": chunk_text,
                "chunking_strategy": chunking_strategy,
            })
        
        vector_store.upsert_vectors(vector_ids, embeddings, vector_metadata)
        
        document = Document(
            id=document_id,
            filename=file.filename,
            original_filename=file.filename,
            file_type=file_ext.replace(".", ""),
            file_size=len(content),
            chunking_strategy=chunking_strategy,
            chunk_count=len(chunks),
            vector_db_provider=settings.VECTOR_DB_PROVIDER,
            text_length=len(clean_text),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        db.add(document)
        
        for i, (chunk_text, vector_id) in enumerate(zip(chunks, vector_ids)):
            chunk = DocumentChunk(
                id=str(uuid.uuid4()),
                document_id=document_id,
                chunk_index=i,
                chunk_text=chunk_text,
                vector_id=vector_id,
                created_at=datetime.utcnow()
            )
            db.add(chunk)
        
        db.commit()
        
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        file_path = os.path.join(settings.UPLOAD_DIR, f"{document_id}{file_ext}")
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(content)
        
        return UploadResponse(
            document_id=document_id,
            filename=file.filename,
            file_type=file_ext.replace(".", ""),
            chunk_count=len(chunks),
            chunking_strategy=chunking_strategy,
            vector_db_provider=settings.VECTOR_DB_PROVIDER,
            message="Document uploaded and processed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )


class DocumentInfo(BaseModel):
    id: str
    filename: str
    file_type: str
    file_size: int
    chunk_count: int
    chunking_strategy: str
    vector_db_provider: str
    text_length: int
    created_at: datetime


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total: int


@router.get("/", response_model=DocumentListResponse)
async def list_documents(db: Session = Depends(get_db)):
    """List all uploaded documents."""
    documents = db.query(Document).all()
    return DocumentListResponse(
        documents=[
            DocumentInfo(
                id=d.id,
                filename=d.filename,
                file_type=d.file_type,
                file_size=d.file_size,
                chunk_count=d.chunk_count,
                chunking_strategy=d.chunking_strategy,
                vector_db_provider=d.vector_db_provider,
                text_length=d.text_length,
                created_at=d.created_at
            ) for d in documents
        ],
        total=len(documents)
    )


class DocumentDetail(BaseModel):
    id: str
    filename: str
    file_type: str
    file_size: int
    chunk_count: int
    chunking_strategy: str
    vector_db_provider: str
    text_length: int
    created_at: datetime
    chunks: list[dict]


@router.get("/{document_id}", response_model=DocumentDetail)
async def get_document(document_id: str, db: Session = Depends(get_db)):
    """Get document details including all chunks."""
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    chunks = db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).all()

    return DocumentDetail(
        id=document.id,
        filename=document.filename,
        file_type=document.file_type,
        file_size=document.file_size,
        chunk_count=document.chunk_count,
        chunking_strategy=document.chunking_strategy,
        vector_db_provider=document.vector_db_provider,
        text_length=document.text_length,
        created_at=document.created_at,
        chunks=[{"chunk_index": c.chunk_index, "chunk_text": c.chunk_text} for c in chunks]
    )


@router.delete("/{document_id}")
async def delete_document(document_id: str, db: Session = Depends(get_db)):
    """Delete a document and all its associated data."""
    init_services()

    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        chunks = db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).all()
        vector_ids = [c.vector_id for c in chunks if c.vector_id]
        
        print(f"[Delete] Document {document_id}: Found {len(chunks)} chunks, {len(vector_ids)} vector IDs")
        print(f"[Delete] Vector IDs: {vector_ids[:5]}...")

        if vector_ids:
            try:
                vector_store.delete_vectors(vector_ids)
                print(f"[Delete] Successfully deleted {len(vector_ids)} vectors from Qdrant")
            except Exception as ve:
                print(f"[Delete] Vector store error: {ve}")

        chunk_count = db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete()
        print(f"[Delete] Deleted {chunk_count} chunks from database")

        db.delete(document)
        db.commit()
        print(f"[Delete] Document {document_id} deleted from database")

        file_ext = document.file_type
        if not file_ext.startswith("."):
            file_ext = f".{file_ext}"
        file_path = os.path.join(settings.UPLOAD_DIR, f"{document_id}{file_ext}")
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"[Delete] File deleted: {file_path}")

        return {"message": "Document deleted successfully", "document_id": document_id}

    except Exception as e:
        db.rollback()
        print(f"[Delete] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")
