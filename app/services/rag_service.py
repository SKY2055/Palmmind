"""RAG Service — Retrieval-Augmented Generation pipeline."""

import re
from typing import List, Dict, Optional, Tuple

from app.config import get_settings
from app.services.embeddings import EmbeddingService
from app.services.vector_store import get_vector_store
from app.services.chat_memory import ChatMemoryService
from app.services.llm_client import get_llm_client

settings = get_settings()


class RAGService:
    def __init__(self):
        self.embedding_service = EmbeddingService(settings.EMBEDDING_MODEL)
        self.vector_store = get_vector_store(settings)
        self.chat_memory = ChatMemoryService()
        self.llm_client = get_llm_client(settings)

    def retrieve_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        target_doc_id: Optional[str] = None,
    ) -> Dict:
        """
        Retrieve relevant document chunks from Qdrant.

        Returns:
            {
                "results":    List[Dict],   # chunk dicts
                "ambiguous":  bool,
                "candidates": List[Dict],   # only when ambiguous
            }
        """
        top_k = top_k or settings.RAG_TOP_K

        # ── If a specific document is already selected, return ALL its chunks ──
        if target_doc_id:
            results = self._fetch_all_chunks_for_doc(target_doc_id)
            print(f"[RAG] Forced retrieval from {target_doc_id}: {len(results)} chunks")
            return {"results": results, "ambiguous": False, "candidates": []}

        # ── Semantic search ──
        query_embedding = self.embedding_service.embed_query(query)
        semantic_results = self.vector_store.search(
            query_vector=query_embedding,
            top_k=top_k * 3,  # over-fetch so we can filter properly
        )

        if not semantic_results:
            return {"results": [], "ambiguous": False, "candidates": []}

        print(f"[RAG] Semantic search returned {len(semantic_results)} chunks from top_k={top_k * 3}")
        for i, result in enumerate(semantic_results[:10], 1):
            doc_id = result.get("metadata", {}).get("document_id")
            print(f"[RAG] Result {i}: doc_id={doc_id}")

        # ── Collect unique documents from results ──
        doc_info: Dict[str, Dict] = {}
        for result in semantic_results:
            doc_id = result.get("metadata", {}).get("document_id")
            if doc_id and doc_id not in doc_info:
                chunk_text = result.get("metadata", {}).get("chunk_text", "")
                extracted_name = self._extract_name(chunk_text)
                extracted_email = self._extract_email(chunk_text)
                doc_info[doc_id] = {
                    "document_id": doc_id,
                    "name": extracted_name,
                    "email": extracted_email,
                }
                print(f"[RAG] Document {doc_id}: name='{extracted_name}', email='{extracted_email}'")

        # ── Name-based disambiguation ──
        # Extract meaningful tokens from query (length > 2, not stop-words)
        STOP = {"the", "and", "for", "what", "who", "are", "his", "her", "tell", "about", "give", "me"}
        query_tokens = [
            w for w in re.findall(r"[a-zA-Z]+", query.lower())
            if len(w) > 2 and w not in STOP
        ]
        
        # Check if query contains email - use it for exact matching
        query_email = self._extract_email(query)
        if query_email != "Not provided":
            print(f"[RAG] Query contains email: {query_email}, searching all documents")
            # Search ALL chunks in the vector store to find document with this email
            points, _ = self.vector_store.client.scroll(
                collection_name=self.vector_store.collection_name,
                limit=2000,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                chunk_text = point.payload.get("chunk_text", "")
                if query_email in chunk_text:
                    doc_id = point.payload.get("document_id")
                    results = self._fetch_all_chunks_for_doc(doc_id)
                    print(f"[RAG] Email match in chunk → {doc_id} ({len(results)} chunks)")
                    return {"results": results, "ambiguous": False, "candidates": []}
            # Email in query but not found in any document - return empty
            print(f"[RAG] Email {query_email} not found in any document")
            return {"results": [], "ambiguous": False, "candidates": []}

        matching_docs: List[Dict] = []
        for info in doc_info.values():
            name_lower = info["name"].lower()
            # Fuzzy matching: check if any query token is a substring of the name or vice versa
            for tok in query_tokens:
                if len(tok) > 3 and (tok in name_lower or name_lower in tok):
                    matching_docs.append(info)
                    break

        # Multiple distinct people match → ambiguous
        if len(matching_docs) > 1:
            print(f"[RAG] Ambiguity: {len(matching_docs)} candidates match query tokens")
            return {"results": [], "ambiguous": True, "candidates": matching_docs}

        # Exactly one person matched → return ALL their chunks for full coverage
        if len(matching_docs) == 1:
            doc_id = matching_docs[0]["document_id"]
            results = self._fetch_all_chunks_for_doc(doc_id)
            print(f"[RAG] Single name match → {doc_id} ({len(results)} chunks)")
            return {"results": results, "ambiguous": False, "candidates": matching_docs}

        # No name match - if query contains name-like tokens, return empty; otherwise use semantic results
        if query_tokens and len(query_tokens) >= 2:
            print(f"[RAG] Query contains name tokens but no match found - returning empty results")
            return {"results": [], "ambiguous": False, "candidates": []}

        # No name match — return top semantic results as-is
        print(f"[RAG] No name match; using top-{top_k} semantic results")
        return {
            "results": semantic_results[:top_k],
            "ambiguous": False,
            "candidates": [],
        }

    def format_context(
        self, results: List[Dict], query: Optional[str] = None
    ) -> Tuple[str, List[Dict]]:
        """
        Format retrieved chunks into a single context string.

        Returns: (context_string, filtered_results)
        """
        if not results:
            print("[RAG] No chunks to format")
            return "No relevant documents found.", []

        # If chunks span multiple documents (shouldn't happen after retrieval fix,
        # but guard anyway) keep only the document with the most chunks.
        doc_chunks: Dict[str, List[Dict]] = {}
        for r in results:
            doc_id = r.get("metadata", {}).get("document_id", "unknown")
            doc_chunks.setdefault(doc_id, []).append(r)

        if len(doc_chunks) > 1:
            best_doc = max(doc_chunks, key=lambda k: len(doc_chunks[k]))
            print(
                f"[RAG] Multiple docs in results; keeping {best_doc} "
                f"({len(doc_chunks[best_doc])} chunks)"
            )
            results = doc_chunks[best_doc]

        context_parts: List[str] = []
        for i, result in enumerate(results, 1):
            text = result.get("metadata", {}).get("chunk_text", "").strip()
            doc_id = result.get("metadata", {}).get("document_id", "unknown")
            if text:
                print(f"[RAG] Chunk {i} (doc={doc_id}, len={len(text)}): {text[:120]}…")
                context_parts.append(f"[Chunk {i}]\n{text}")

        if not context_parts:
            return "No text found in retrieved chunks.", []

        return "\n\n".join(context_parts), results

    def build_prompt(
        self,
        query: str,
        context: str,
        chat_history: str,
        booking_context: Optional[str] = None,
    ) -> str:
        booking_instruction = ""
        if booking_context:
            booking_instruction = (
                f"\n\nBOOKING CONTEXT:\n{booking_context}\n"
                "IMPORTANT: Acknowledge the booking status above. "
                "Do NOT say 'I don't have that information' if booking context is provided."
            )

        return f"""You are PalmMind AI, a specialized assistant for analysing resumes and candidate documents.
Answer ONLY from the DOCUMENT CONTEXT below. If something is not in the context, say so clearly.

DOCUMENT CONTEXT:
{context}

PREVIOUS CONVERSATION:
{chat_history if chat_history else "None"}

USER QUERY: {query}
{booking_instruction}

INSTRUCTIONS:
1. Be exhaustive — scan every chunk for the requested information.
2. For skills, list ALL found (technical, soft, tools).
3. For experience, include: title, company, dates, and all bullet points.
4. For education, include: degree, institution, CGPA/grades, graduation date.
5. For certifications/courses, include: title, provider, date.
6. Use clear headings and bullet points.
7. If information is genuinely absent, say "I don't have information about [X] in the uploaded documents."

Answer:"""

    def generate_response(self, prompt: str) -> Tuple[str, Optional[str]]:
        """Call multi-provider LLM; return (text, provider_name)."""
        system_prompt = (
            "You are PalmMind AI. Answer ONLY from the provided document context. "
            "If information is not present, say so clearly."
        )
        result = self.llm_client.generate(prompt, system_prompt)
        if result["success"]:
            return result["text"], result["provider"]
        return result["text"], None

    def chat(
        self,
        session_id: str,
        query: str,
        target_doc_id: Optional[str] = None,
        booking_context: Optional[str] = None,
    ) -> Dict:
        """Full RAG pipeline for one conversational turn."""
        chat_history = self.chat_memory.get_formatted_history(session_id)

        retrieval_data = self.retrieve_context(query, target_doc_id=target_doc_id)

        if retrieval_data.get("ambiguous"):
            return {
                "response": (
                    "I found multiple applicants matching that name. "
                    "Which one are you referring to?"
                ),
                "session_id": session_id,
                "ambiguous": True,
                "candidates": retrieval_data["candidates"],
                "context_used": False,
                "sources": [],
                "provider": None,
            }

        context, filtered_results = self.format_context(
            retrieval_data["results"], query
        )
        prompt = self.build_prompt(query, context, chat_history, booking_context)
        response, provider = self.generate_response(prompt)

        if (
            "certification" in query.lower()
            and "i don't have" in response.lower()
        ):
            certs = self._extract_certifications_pattern(context)
            if certs:
                response = f"Based on the documents, the certifications found are:\n\n{certs}"

        sources = [r.get("metadata", {}).get("document_id") for r in filtered_results]
        context_used = bool(context) and context != "No relevant documents found."

        self.chat_memory.add_message(session_id, "user", query)
        self.chat_memory.add_message(session_id, "assistant", response)

        return {
            "response": response,
            "session_id": session_id,
            "context_used": context_used,
            "sources": sources,
            "provider": provider,
        }

    def _fetch_all_chunks_for_doc(self, doc_id: str) -> List[Dict]:
        """Scroll the entire collection and return chunks for one document."""
        points, _ = self.vector_store.client.scroll(
            collection_name=self.vector_store.collection_name,
            limit=2000,
            with_payload=True,
            with_vectors=False,
        )
        return [
            {"id": p.id, "score": 1.0, "metadata": p.payload}
            for p in points
            if p.payload.get("document_id") == doc_id
        ]

    @staticmethod
    def _extract_name(text: str) -> str:
        email_match = re.search(r"[\w\.\-]+@[\w\.\-]+\.\w+", text)
        if email_match:
            email_pos = email_match.start()
            text_before = text[max(0, email_pos - 150):email_pos]
            patterns = [
                r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b",
                r"\b([A-Z]{2,}(?:\s+[A-Z]{2,})+)\b",
            ]
            for pattern in patterns:
                match = re.search(pattern, text_before)
                if match:
                    name = match.group(1)
                    if name not in ["Data Science Engineer", "Data Analytics Certification", "Curricular Activities", "Git Web", "John Doe Email", "Marital Status", "Online Certification"]:
                        return name
    
        first_part = text[:200]
        patterns = [
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b",
            r"\b([A-Z]{2,}(?:\s+[A-Z]{2,})+)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, first_part)
            if match:
                name = match.group(1)
                if name not in ["Data Science Engineer", "Data Analytics Certification", "Curricular Activities", "Git Web", "John Doe Email", "Marital Status", "Online Certification"]:
                    return name
        return "Unknown"

    @staticmethod
    def _extract_email(text: str) -> str:
        first_part = text[:300]
        match = re.search(r"[\w\.\-]+@[\w\.\-]+\.\w+", first_part)
        if match:
            return match.group(0)
        match = re.search(r"[\w\.\-]+@[\w\.\-]+\.\w+", text)
        return match.group(0) if match else "Not provided"

    @staticmethod
    def _extract_certifications_pattern(context: str) -> Optional[str]:
        months = (
            "January|February|March|April|May|June|July|August|September|"
            "October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
        )
        pattern = (
            rf"([A-Za-z][A-Za-z\s]+?(?:Certification|Certificate|Certified))"
            rf"\s*[—\-–]\s*"
            rf"([A-Za-z\s]+(?:{months})\s*\d{{4}})"
        )
        matches = re.findall(pattern, context, re.IGNORECASE)
        if matches:
            return "\n".join(f"- {m[0].strip()} — {m[1].strip()}" for m in matches)
        return None


_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service