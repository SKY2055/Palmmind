from typing import List, Dict, Optional
from app.config import get_settings
from app.services.embeddings import EmbeddingService
from app.services.vector_store import get_vector_store
from app.services.chat_memory import ChatMemoryService
from app.services.llm_client import get_llm_client, reset_llm_client

settings = get_settings()

# Force reset LLM client on module load to pick up config changes
reset_llm_client()


class RAGService:
    def __init__(self):
        self.embedding_service = EmbeddingService(settings.EMBEDDING_MODEL)
        self.vector_store = get_vector_store(settings)
        self.chat_memory = ChatMemoryService()
        self.llm_client = get_llm_client(settings)
    
    def retrieve_context(self, query: str, top_k: int = None) -> List[Dict]:
        """Retrieve relevant document chunks from Qdrant."""
        top_k = top_k or settings.RAG_TOP_K
        
        print(f"[RAG] retrieve_context called with query: {query}")
        
        query_embedding = self.embedding_service.embed_query(query)
        
        # For hard-to-retrieve sections (certifications, references, etc.), retrieve all chunks from the target document
        query_lower = query.lower()
        hard_to_retrieve = ["certification", "certifications", "reference", "references", "extracurricular", "extra curricular", "hobby", "hobbies", "education"]
        if any(kw in query_lower for kw in hard_to_retrieve):
            print(f"[RAG] Query about hard-to-retrieve section, retrieving all chunks")
            
            semantic_results = self.vector_store.search(
                query_vector=query_embedding,
                top_k=10
            )
            if semantic_results:
                # Check for unique identifiers (email, phone) in query to disambiguate duplicate names
                import re
                email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
                phone_pattern = r'\+?\d{9,15}'
                email_match = re.search(email_pattern, query_lower)
                phone_match = re.search(phone_pattern, query_lower)
                
                if email_match or phone_match:
                    target_doc_id = None
                    for result in semantic_results:
                        doc_id = result.get("metadata", {}).get("document_id")
                        chunk_text = result.get("metadata", {}).get("chunk_text", "").lower()
                        if email_match and email_match.group() in chunk_text:
                            target_doc_id = doc_id
                            print(f"[RAG] Found target document {doc_id} based on email match")
                            break
                        elif phone_match and phone_match.group() in chunk_text:
                            target_doc_id = doc_id
                            print(f"[RAG] Found target document {doc_id} based on phone match")
                            break
                    
                    if target_doc_id:
                        all_points = self.vector_store.client.scroll(
                            collection_name=self.vector_store.collection_name,
                            limit=1000,
                            with_payload=True
                        )[0]
                        all_results = [
                            {"id": p.id, "score": 0.0, "metadata": p.payload}
                            for p in all_points
                            if p.payload.get("document_id") == target_doc_id
                        ]
                        print(f"[RAG] Retrieved {len(all_results)} chunks from document {target_doc_id}")
                        return all_results
                else:
                    # No unique identifier provided - find all documents matching the name
                    matching_doc_ids = []
                    words = query_lower.split()
                    for result in semantic_results:
                        doc_id = result.get("metadata", {}).get("document_id")
                        chunk_text = result.get("metadata", {}).get("chunk_text", "").lower()
                        for word in words:
                            if len(word) > 3 and word in chunk_text:
                                if doc_id not in matching_doc_ids:
                                    matching_doc_ids.append(doc_id)
                                    print(f"[RAG] Found matching document {doc_id} based on name match: {word}")
                                break
                    
                    if matching_doc_ids:
                        if len(matching_doc_ids) > 1:
                            print(f"[RAG] Multiple documents match name: {matching_doc_ids}. Retrieving from all.")
                        else:
                            print(f"[RAG] Single document matches name: {matching_doc_ids[0]}")
                        
                        all_points = self.vector_store.client.scroll(
                            collection_name=self.vector_store.collection_name,
                            limit=1000,
                            with_payload=True
                        )[0]
                        all_results = [
                            {"id": p.id, "score": 0.0, "metadata": p.payload}
                            for p in all_points
                            if p.payload.get("document_id") in matching_doc_ids
                        ]
                        print(f"[RAG] Retrieved {len(all_results)} chunks from {len(matching_doc_ids)} documents")
                        return all_results
        
        # Standard semantic search
        results = self.vector_store.search(
            query_vector=query_embedding,
            top_k=top_k
        )
        print(f"[RAG] Semantic search returned {len(results)} results")
        
        return results
    
    def format_context(self, results: List[Dict], query: str = None) -> tuple:
        """Format retrieved chunks into context string. Returns (context, filtered_results)."""
        if not results:
            print("[RAG] No chunks retrieved from vector store")
            return "No relevant documents found.", []
        
        print(f"[RAG] Original retrieval: {len(results)} chunks")
        for i, result in enumerate(results, 1):
            metadata = result.get("metadata", {})
            text = metadata.get("chunk_text", "")
            doc_id = metadata.get("document_id", "unknown")
            if text:
                print(f"[RAG] Original Chunk {i} (doc: {doc_id}, length: {len(text)} chars): {text[:300]}...")
        
        target_doc_id = None
        if query:
            # Identify which document to use based on name in query
            query_lower = query.lower()
            doc_chunks = {}
            for result in results:
                doc_id = result.get("metadata", {}).get("document_id", "unknown")
                if doc_id not in doc_chunks:
                    doc_chunks[doc_id] = []
                doc_chunks[doc_id].append(result)
            
            # If multiple documents, filter by name match
            if len(doc_chunks) > 1:
                for doc_id, chunks in doc_chunks.items():
                    doc_text = " ".join([c.get("metadata", {}).get("chunk_text", "") for c in chunks])
                    words = query_lower.split()
                    for word in words:
                        if len(word) > 3 and word in doc_text.lower():
                            target_doc_id = doc_id
                            print(f"[RAG] Filtering to document {doc_id} based on name match: {word}")
                            break
                    if target_doc_id:
                        break
                
                if not target_doc_id:
                    target_doc_id = max(doc_chunks.keys(), key=lambda k: len(doc_chunks[k]))
                    print(f"[RAG] No name match, using document with most chunks: {target_doc_id}")
        
        filtered_results = results
        if target_doc_id:
            filtered_results = [r for r in results if r.get("metadata", {}).get("document_id") == target_doc_id]
            print(f"[RAG] Filtered from {len(results)} to {len(filtered_results)} chunks for document {target_doc_id}")
        
        print(f"[RAG] Retrieved {len(filtered_results)} chunks")
        context_parts = []
        for i, result in enumerate(filtered_results, 1):
            metadata = result.get("metadata", {})
            text = metadata.get("chunk_text", "")
            doc_id = metadata.get("document_id", "unknown")
            if text:
                print(f"[RAG] Chunk {i} (doc: {doc_id}, length: {len(text)} chars): {text[:200]}...")
                context_parts.append(f"[Document {i}]\n{text}")
        
        return "\n\n".join(context_parts), filtered_results
    
    def build_prompt(self, query: str, context: str, chat_history: str, booking_context: str = None) -> str:
        """Build the RAG prompt with context and history."""
        booking_instruction = ""
        if booking_context:
            booking_instruction = f"\n\nBOOKING CONTEXT: {booking_context}\nIMPORTANT: When responding, acknowledge that the interview booking has been confirmed and provide the booking details. Do NOT say 'I don't have that information' when a booking has been made."
        
        if "PROJECTS" in context:
            print(f"[RAG] Context contains PROJECTS section")
        if "EXPERIENCE" in context:
            print(f"[RAG] Context contains EXPERIENCE section")
        if "SKILLS" in context:
            print(f"[RAG] Context contains SKILLS section")
        if "CERTIFICATIONS" in context or "CERTIFICATION" in context:
            print(f"[RAG] Context contains CERTIFICATIONS section")
        
        prompt = f"""You are a helpful AI assistant for PalmMind. Answer the user's question based on the provided context from uploaded documents.

IMPORTANT INSTRUCTIONS:
- Use ONLY the information from the "Context from documents" section below
- Extract ALL relevant information exactly as it appears in the documents
- For names, use the EXACT spelling from the document (do not correct or change names)
- For education details, include: degree name, university/college name (exact), CGPA/grades, years. ALSO look for certifications mixed with education - extract any certifications mentioned in education sections.
- For skills: If you see "SKILLS" in the context, extract EVERY skill listed. There may be many skills separated by pipes or commas. List ALL of them - do not skip any. Include both technical skills and soft skills.
- For projects, list ALL projects mentioned with their descriptions and technologies
- For experience/work history: If you see "EXPERIENCE" in the context, extract EVERY job entry listed. There may be multiple jobs separated by company names or dates. List ALL of them with their roles, companies, dates, and responsibilities. DO NOT say experience is not mentioned if you see "EXPERIENCE" or job titles in the context.
- For certifications: CRITICAL - Search the ENTIRE context for ANY mention of certifications. Look for:
  * Words like "Certification", "Certified", "Certify" (case-insensitive)
  * Patterns like "Certification — Institute", "Certification from", "Certified in"
  * Certifications may be in a dedicated CERTIFICATIONS section OR mixed with EDUCATION section
  * Extract EVERY certification you find with: name, organization/institute, and dates
  * DO NOT skip certifications even if they appear with education
  * If you see "Data Analytics Certification", "Cloud Computing Certification", or similar, extract them
  
  EXAMPLES of certifications to extract:
  - "Data Analytics Certification — Institute of Analytics March 2024 – May 2025" → Data Analytics Certification from Institute of Analytics (March 2024 – May 2025)
  - "Cloud Computing Certification – Intern Certify" → Cloud Computing Certification from Intern Certify
  - "IoT Devices – University of Illinois Urbana -Champaign via Coursera" → IoT Devices Certification from University of Illinois Urbana-Champaign via Coursera

- For courses/training: Search the ENTIRE context for ANY mention of courses, training programs, workshops, or online learning. Look for:
  * Words like "Course", "Training", "Workshop", "Program", "NPTEL", "Coursera", "edX", "Udemy" (case-insensitive)
  * Patterns like "Course — Institute", "Training from", "Workshop on"
  * Courses may be in a dedicated TRAINING/COURSES section OR mixed with EDUCATION section
  * Extract EVERY course/training you find with: name, organization/institute, duration/dates
  * DO NOT skip courses even if they appear with education
  * If you see "8 Weeks Introduction to Machine Learning", "Big Data Analytics Using Spark", or similar, extract them
  
- For references: If you see "REFERENCES" or "Reference" in the context, extract EVERY reference listed with their names, titles, companies, and contact information.
- If the context contains relevant information, provide a comprehensive and accurate answer
- If the context is empty or doesn't contain the answer AND there is no booking context, say "I don't have that information in the uploaded documents."
- Do NOT use your general knowledge or make up information
- Do NOT guess or infer information not present in the context
- When asked about projects, search through ALL document chunks for any section labeled "PROJECTS" or containing project descriptions
- When asked about experience/work history, if you see "EXPERIENCE" in the context, extract ALL job entries - there may be multiple, list them ALL
- When asked about skills, if you see "SKILLS" in the context, extract ALL skills - there may be many, list them ALL
- When asked about certifications, search through ALL document chunks for the word "certification" (case-insensitive) and extract ALL certifications - they may be in a dedicated section or mixed with education. Look specifically for patterns like "Certification — Institute". Use the examples above as a guide.
- When asked about references, if you see "REFERENCES" or "Reference" in the context, extract ALL references - there may be multiple, list them ALL{booking_instruction}

Context from documents:
{context}

Previous conversation:
{chat_history}

User: {query}
Assistant:"""
        return prompt
    
    def generate_response(self, prompt: str) -> tuple:
        """Generate response using multi-provider LLM with fallback."""
        system_prompt = "You are PalmMind AI assistant. Answer ONLY based on the provided document context. If information is not in the context, say so clearly."
        
        result = self.llm_client.generate(prompt, system_prompt)
        
        if result["success"]:
            return result["text"], result["provider"]
        else:
            return result["text"], None
    
    def chat(self, session_id: str, query: str, booking_context: str = None) -> Dict:
        """Process a chat query with RAG."""
        chat_history = self.chat_memory.get_formatted_history(session_id)
        
        results = self.retrieve_context(query)
        context, filtered_results = self.format_context(results, query)
        
        prompt = self.build_prompt(query, context, chat_history, booking_context)
        
        response, provider = self.generate_response(prompt)
        
        # Fallback: if LLM didn't find certifications, try pattern matching
        query_lower = query.lower()
        if "certification" in query_lower and "i don't have that information" in response.lower():
            print("[RAG] LLM didn't find certifications, trying pattern matching")
            certifications = self.extract_certifications_pattern(context)
            if certifications:
                response = f"Based on the documents, the certifications are:\n\n{certifications}"
        
        self.chat_memory.add_message(session_id, "user", query)
        self.chat_memory.add_message(session_id, "assistant", response)
        
        return {
            "response": response,
            "session_id": session_id,
            "context_used": len(filtered_results) > 0,
            "sources": [r.get("metadata", {}).get("document_id") for r in filtered_results],
            "provider": provider
        }
    
    def extract_certifications_pattern(self, context: str) -> str:
        """Extract certifications using pattern matching as fallback."""
        import re
        
        pattern = r'([A-Za-z\s]+Certification)\s*[—-]\s*([A-Za-z\s]+(?:of\s+[A-Za-z\s]+)?\s+(?:March|January|February|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s*\d{4}\s*[–-]\s*(?:March|January|February|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s*\d{4})'
        
        certifications = []
        
        matches = re.findall(pattern, context, re.IGNORECASE)
        for match in matches:
            cert_name = match[0].strip()
            details = match[1].strip()
            certifications.append(f"- {cert_name} — {details}")
        
        if certifications:
            return "\n".join(certifications)
        return None


# Singleton
_rag_service = None


def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
