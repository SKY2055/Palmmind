from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import documents_router, chat_router, bookings_router
from app.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    description="PalmMind Conversational RAG System",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(documents_router)
app.include_router(chat_router)
app.include_router(bookings_router)


@app.get("/")
async def root():
    return {
        "message": "PalmMind RAG",
        "docs": "/docs",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
