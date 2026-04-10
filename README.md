# PalmMind RAG

> **Conversational RAG System with Multi-Provider LLM, Interview Booking, and Document Management**

PalmMind is a production-ready **Retrieval-Augmented Generation (RAG)** API built with FastAPI. It enables you to upload documents (like resumes), ask questions about them, and book interviews through natural conversation.

---

## 📖 What Can You Do?

1. **Upload your resume** (PDF or TXT) to the system
   - Supports image-based PDFs via OCR (tesseract, poppler)
2. **Ask questions** about it - "What is my CGPA?", "What skills do I have?", "What courses have I taken?"
3. **Book an interview** by simply chatting - "I want to book an interview for tomorrow at 3pm"
4. **Get AI responses** based ONLY on your uploaded documents (no hallucinations!)
5. **Handle duplicate names** - Provide email/phone for precise matching when multiple applicants share the same name
6. **Extract comprehensive information** - Name, Address, Contact, Social Media, Summary, Experience, Projects, Education, Skills, Certifications, Courses, Extra-curricular activities, Hobbies, References

---

## � Project Structure

```
PalmMind/
│
├── app/                           # Main application code
│   ├── main.py                    # FastAPI app entry point
│   ├── config.py                  # Settings & environment variables
│   │
│   ├── models/                    # Database models (SQLAlchemy ORM)
│   │   ├── base.py                # Database connection & session
│   │   ├── document.py            # Document & DocumentChunk tables
│   │   └── interview.py           # InterviewBooking table
│   │
│   ├── schemas/                   # Pydantic models (API validation)
│   │   ├── document.py            # Upload/response schemas
│   │   ├── chat.py                # Chat request/response schemas
│   │   └── booking.py             # Booking schemas
│   │
│   ├── routers/                   # API endpoint handlers
│   │   ├── documents.py           # Document upload/list/delete endpoints
│   │   ├── chat.py                # Chat & RAG endpoints
│   │   └── bookings.py            # Interview booking endpoints
│   │
│   └── services/                  # Business logic
│       ├── llm_client.py        # Multi-provider LLM (Groq, Gemini)
│       ├── rag_service.py         # RAG pipeline implementation
│       ├── booking_service.py     # Extract booking details from chat
│       ├── chat_memory.py         # Redis conversation storage
│       ├── vector_store.py        # Qdrant vector database client
│       ├── embeddings.py          # Text embedding service
│       ├── chunker.py             # Document text chunking
│       └── extractor.py           # PDF/TXT text extraction
│
├── .env                           # Environment variables (not in git)
├── .env.example                   # Template for .env
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── .gitignore                     # Files to ignore in git
```

---

## 🧠 How RAG Works (Architecture Diagram)

### What is RAG?

**RAG = Retrieval-Augmented Generation**

Instead of the AI answering from its training data (which might be wrong or outdated), RAG:
1. **Retrieves** relevant text chunks from YOUR uploaded documents
2. **Augments** the AI prompt with that context
3. **Generates** an answer based ONLY on your documents

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER REQUEST                            │
│  "What is my CGPA?"                                            │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI SERVER                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │  EMBEDDING  │  │   VECTOR    │  │     CHAT MEMORY         │ │
│  │   SERVICE   │──│    STORE    │  │      (Redis)            │ │
│  │             │  │  (Qdrant)   │  │                         │ │
│  │ Text →      │  │  Search for │  │  Store conversation     │ │
│  │ Vector      │  │  similar    │  │  history                │ │
│  └─────────────┘  │  chunks     │  └─────────────────────────┘ │
│       │           └──────┬──────┘            │                  │
│       │                  │                   │                  │
│       │                  ▼                   │                  │
│       │           ┌─────────────┐             │                  │
│       │           │  Retrieved  │             │                  │
│       │           │   Context   │             │                  │
│       │           │  (5 chunks) │             │                  │
│       │           └──────┬──────┘             │                  │
│       │                  │                    │                  │
│       └──────────────────┼────────────────────┘                  │
│                          │                                      │
│                          ▼                                      │
│           ┌───────────────────────────────┐                    │
│           │     BUILD RAG PROMPT          │                    │
│           │                                 │                    │
│           │  Context: [Your resume chunks]  │                    │
│           │  History: [Previous messages]   │                    │
│           │  Question: "What is my CGPA?"   │                    │
│           └───────────────┬─────────────────┘                    │
│                           │                                      │
│                           ▼                                      │
│           ┌───────────────────────────────┐                    │
│           │      LLM CLIENT               │                    │
│           │  ┌─────────┐ ┌─────────┐     │                    │
│           │  │  Groq   │ │ Gemini  │     │                    │
│           │  │ (First) │ │(Fallback│     │                    │
│           │  └────┬────┘ └────┬────┘     │                    │
│           │       │           │          │                    │
│           │       ▼           │          │                    │
│           │  Generate Answer   │          │                    │
│           │  "Your B.Tech      │          │                    │
│           │   CGPA is 8.5"    │          │                    │
│           └───────────────────┘          │                    │
│                                          │                    │
└──────────────────────────────────────────┼────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     POSTGRESQL DATABASE                          │
│  ┌─────────────┐  ┌─────────────────────────────────────────┐   │
│  │  Document   │  │        InterviewBooking                 │   │
│  │   Metadata  │  │  ┌─────────┬─────────┬────────┬───────┐│   │
│  │             │  │  │  Name   │  Email  │  Date  │ Time  ││   │
│  │  id,        │  │  │ John    │john@... │2024-01 │ 14:00 ││   │
│  │  filename,  │  │  │  Doe    │         │ -20    │       ││   │
│  │  chunks     │  │  └─────────┴─────────┴────────┴───────┘│   │
│  └─────────────┘  └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow Steps

**When you UPLOAD a document:**
```
PDF/TXT File → Text Extraction → Chunking (1024 chars) → Embeddings → Qdrant Vector Store
```

**When you ASK a question:**
```
1. Your Question → Embedding
2. Search Qdrant → Find similar chunks (top 8)
3. Build Prompt = Context + History + Question
4. Send to LLM (Groq → Gemini fallback)
5. Store in Redis memory
6. Return answer + session_id
```

**When you BOOK an interview:**
```
Chat Message → LLM extracts (name, email, date, time) → Save to PostgreSQL
```

---

## �🚀 Quick Start (5 Minutes)

### Step 1: Install Python & Dependencies

**You need:** Python 3.9 or higher

```bash
# Check Python version
python --version  # Should be 3.9+

# Clone the repository
git clone <repository-url>
cd PalmMind

# Create virtual environment (recommended)
python -m venv .venv

# Activate it
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

**System Dependencies for OCR Support (macOS):**
```bash
# Install tesseract (OCR engine) and poppler (for PDF processing)
brew install tesseract poppler
```

**System Dependencies for OCR Support (Linux/Ubuntu):**
```bash
sudo apt-get install tesseract-ocr poppler-utils
```

**System Dependencies for OCR Support (Windows):**
1. Download Tesseract from https://github.com/UB-Mannheim/tesseract/wiki
2. Download Poppler from https://github.com/oschwartz10612/poppler-windows
3. Add both to your system PATH

### Step 4: Start Docker Services (Qdrant, Redis, PostgreSQL)

**Before running the FastAPI server, you must start the required Docker services.**

**First, make sure Docker Desktop is running:**
- On macOS: Open Docker Desktop from Applications or run `open /Applications/Docker.app`
- Wait for the Docker icon to appear in the menu bar
- Verify Docker is running with `docker version`

```bash
# Start Qdrant (Vector Database)
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant

# Start Redis (Chat Memory)
docker run -d -p 6379:6379 --name redis redis:7-alpine

# Start PostgreSQL (Database)
docker run -d -p 5432:5432 --name postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=palmmind_db \
  postgres:15-alpine
```

**To check if they're running:**
```bash
docker ps
```

You should see `qdrant`, `redis`, and `postgres` containers running.

### Step 5: Configure Environment Variables

```bash
# Copy the example file
cp .env.example .env

# Open .env in your editor
nano .env  # or use VS Code, Notepad, etc.
```

**Fill in these REQUIRED values:**

```bash
# Qdrant (use local Docker we started in Step 2)
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=                    # Leave empty for local Docker

# PostgreSQL (use local Docker we started in Step 3)
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/palmmind_db

# Redis (use local Docker we started in Step 2)
REDIS_URL=redis://localhost:6379/0

# LLM Provider (Get FREE key from https://console.groq.com/keys)
LLM_PROVIDER_PRIORITY=groq
GROQ_API_KEY=gsk_your_actual_key_here
```

**How to get a FREE Groq API Key:**
1. Go to https://console.groq.com/keys
2. Sign up (it's free)
3. Click "Create API Key"
4. Copy the key and paste it as `GROQ_API_KEY`

### Step 6: Start the Application

**IMPORTANT: Make sure all Docker containers (qdrant, redis, postgres) are running before starting the server.**

```bash
# Start the FastAPI server
uvicorn app.main:app --reload --port 8005
```

**You should see:**
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8005
```

**The API is now running!** 🎉

---

## 📱 How to Use the API (Interactive Guide)

Once the server is running, open **http://localhost:8005/docs** in your browser.

This is the **Swagger UI** - an interactive interface where you can test all API endpoints.

### Step-by-Step: Upload Your Resume and Chat

#### Step 1: Upload a Document

1. In Swagger UI (`http://localhost:8005/docs`), scroll to **Documents** section
2. Click `POST /api/v1/documents/upload`
3. Click **"Try it out"** button
4. Fill the form:
   - **file**: Click "Choose File" and select your `resume.pdf`
   - **chunking_strategy**: Select `semantic` (recommended for resumes)
5. Click **"Execute"**

**What you should see:**
```json
{
  "message": "File uploaded and processed successfully",
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "resume.pdf",
  "chunk_count": 5
}
```

**Save the `document_id`** - you'll need it later!

#### Step 2: Ask Questions About Your Resume

1. Scroll to **Chat** section
2. Click `POST /api/v1/chat/`
3. Click **"Try it out"**
4. In the request body, enter:

```json
{
  "message": "What is my CGPA in B.Tech?"
}
```

5. Click **"Execute"**

**Response example:**
```json
{
  "response": "Your B.Tech CGPA is 8.5 from XYZ University.",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "context_used": true,
  "provider_used": "groq"
}
```

**Save the `session_id`** to continue the conversation!

#### Step 3: Continue the Conversation

Use the same `session_id` to ask follow-up questions:

```json
{
  "message": "What projects did I mention?",
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

#### Step 4: Book an Interview

You can book an interview by just chatting:

```json
{
  "message": "I want to book an interview. My name is John Doe, email john@example.com, tomorrow at 3pm"
}
```

**Response:**
```json
{
  "response": "I've booked your interview for tomorrow at 3:00 PM...",
  "booking_extracted": {
    "id": "booking_12345",
    "candidate_name": "John Doe",
    "email": "john@example.com",
    "interview_date": "2024-01-20",
    "interview_time": "15:00"
  }
}
```

---

## 🔧 Using cURL (Command Line)

If you prefer command line over Swagger UI:

### Upload Document
```bash
curl -X POST \
  -F "file=@/path/to/your/resume.pdf" \
  -F "chunking_strategy=semantic" \
  http://localhost:8005/api/v1/documents/upload
```

### Chat
```bash
# First message
curl -X POST http://localhost:8005/api/v1/chat/ \
  -H "Content-Type: application/json" \
  -d '{"message": "What is my CGPA?"}'

# Continue conversation (use session_id from response)
curl -X POST http://localhost:8005/api/v1/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Tell me about my projects",
    "session_id": "550e8400-e29b-41d4-a716-446655440000"
  }'
```

### Check Your Bookings
```bash
curl "http://localhost:8005/api/v1/bookings/?email=john@example.com"
```
---

## 📚 Complete API Reference

### Documents API

| Endpoint | Method | What it does |
|----------|--------|--------------|
| `/api/v1/documents/upload` | POST | Upload PDF/TXT file |
| `/api/v1/documents/` | GET | List all your documents |
| `/api/v1/documents/{id}` | GET | View document details |
| `/api/v1/documents/{id}` | DELETE | Delete a document |

**POST Request Body (Upload):**
```json
{
  "file": "your_resume.pdf",           // Required: PDF or TXT file
  "chunking_strategy": "semantic"      // Optional: "semantic" or "fixed"
}
```

**Response (Upload Success):**
```json
{
  "message": "File uploaded successfully",
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "resume.pdf",
  "chunk_count": 5,
  "text_length": 2450
}
```

### Chat API

| Endpoint | Method | What it does |
|----------|--------|--------------|
| `/api/v1/chat/` | POST | Send a message |
| `/api/v1/chat/history/{session_id}` | GET | View chat history |
| `/api/v1/chat/history/{session_id}` | DELETE | Clear chat history |

**POST Request Body (Chat):**
```json
{
  "message": "What is my CGPA in B.Tech?",                    // Required: Your question
  "session_id": "550e8400-e29b-41d4-a716-446655440000"       // Optional: For continuing chat
}
```

**Response (Chat):**
```json
{
  "response": "Your B.Tech CGPA is 8.5 from XYZ University.",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "context_used": true,
  "sources": ["doc_id_1"],
  "booking_extracted": null,
  "provider_used": "groq"
}
```

### Bookings API

| Endpoint | Method | What it does |
|----------|--------|--------------|
| `/api/v1/bookings/?email={email}` | GET | List your bookings |
| `/api/v1/bookings/{id}/cancel` | POST | Cancel a booking |

**Query Parameters (List Bookings):**
- `email` (required): Your email address

**Response (List Bookings):**
```json
{
  "bookings": [
    {
      "id": "booking_12345",
      "candidate_name": "John Doe",
      "email": "john@example.com",
      "interview_date": "2024-01-20",
      "interview_time": "14:00",
      "status": "confirmed"
    }
  ]
}
```

---

## ❓ Troubleshooting

### "Failed to connect to Qdrant"
**Problem:** Qdrant Docker container is not running
**Solution:**
```bash
docker start qdrant
# Or if it doesn't exist:
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant
```

### "Failed to connect to Redis"
**Problem:** Redis Docker container is not running
**Solution:**
```bash
docker start redis
# Or if it doesn't exist:
docker run -d -p 6379:6379 --name redis redis:7-alpine
```

### "No LLM providers available"
**Problem:** API key is missing or invalid
**Solution:**
1. Check your `.env` file has `GROQ_API_KEY` filled in
2. Get a free key from https://console.groq.com/keys
3. Restart the server after updating `.env`

### "I uploaded my resume but the AI gives wrong information"
**Problem:** Old chunks from previous upload
**Solution:**
1. Delete the old document via Swagger UI or API
2. Re-upload with new chunk settings (1024 chars)
3. Ask questions again

### "No text could be extracted from the file"
**Problem:** PDF is corrupted, password-protected, or image-based without OCR support
**Solution:**
1. Verify PDF is not corrupted by opening it in a PDF viewer
2. Ensure tesseract and poppler are installed (see System Dependencies above)
3. For image-based PDFs, OCR will automatically attempt extraction
4. If PDF is password-protected, remove the password before uploading

### "OCR extraction failed"
**Problem:** Tesseract or poppler not installed or not in PATH
**Solution:**
```bash
# Check if tesseract is installed
tesseract --version

# Check if pdftoppm (poppler) is installed
pdftoppm --version

# If not found, install them:
# macOS
brew install tesseract poppler

# Linux/Ubuntu
sudo apt-get install tesseract-ocr poppler-utils
```

### Server won't start
**Checklist:**
- ✅ Python 3.12+ installed
- ✅ All Docker containers running (`docker ps`)
- ✅ PostgreSQL database exists
- ✅ `.env` file created and filled
- ✅ Virtual environment activated

---

## ⚙️ Environment Variables Reference

| Variable | Example Value | Required | Description |
|----------|---------------|----------|-------------|
| `QDRANT_URL` | `http://localhost:6333` | Yes | Qdrant server URL |
| `QDRANT_API_KEY` | (leave empty for local) | For Cloud | Qdrant API key |
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/palmmind_db` | Yes | PostgreSQL connection |
| `REDIS_URL` | `redis://localhost:6379/0` | Yes | Redis connection |
| `LLM_PROVIDER_PRIORITY` | `groq` | Yes | Provider fallback order |
| `GROQ_API_KEY` | `gsk_abc123...` | Yes (if using Groq) | Get free key from groq.com |
| `GEMINI_API_KEY` | `AIzaSyC2g2...` | No | Google Gemini API key |
| `RAG_TOP_K` | `8` | No | Chunks to retrieve |

**Complete `.env` example for local development:**

```bash
# Vector Database (Local Docker)
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=

# PostgreSQL (Local Docker)
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/palmmind_db

# Redis (Local Docker)
REDIS_URL=redis://localhost:6379/0

# LLM (Groq - FREE tier)
LLM_PROVIDER_PRIORITY=groq
GROQ_API_KEY=gsk_your_actual_key_here

# Optional
RAG_TOP_K=8
MAX_CHAT_HISTORY=10
```

---

## 🤝 Tips for Best Results

- **Use semantic chunking** for resumes (better context preservation)
- **Ask specific questions** - "What is my B.Tech CGPA?" not "Tell me about my education"
- **Save session_id** to continue conversations
- **Include name + email + date + time** when booking interviews
- **Check terminal logs** to see which LLM provider responded
- **For duplicate names:** Provide email or phone in your query for precise matching
  - Example: "What are John Doe's certifications? Email: john@example.com"
  - System now uses email-based matching when email is provided in the query
- **OCR Support:** System automatically attempts OCR for image-based PDFs if tesseract/poppler are installed
- **Comprehensive Extraction:** System extracts all resume sections including courses, references, and extra-curricular activities

---

**Built with ❤️ using FastAPI, Qdrant, and LangChain**
