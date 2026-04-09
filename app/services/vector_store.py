from typing import List, Optional
from abc import ABC, abstractmethod


class VectorStore(ABC):
    @abstractmethod
    def upsert_vectors(self, ids: List[str], vectors: List[List[float]], metadata: List[dict]) -> bool:
        """Upsert vectors into the store."""
        pass
    
    @abstractmethod
    def delete_vectors(self, ids: List[str]) -> bool:
        """Delete vectors by IDs."""
        pass
    
    @abstractmethod
    def search(self, query_vector: List[float], top_k: int = 10, filters: Optional[dict] = None) -> List[dict]:
        """Search similar vectors."""
        pass


class QdrantStore(VectorStore):
    def __init__(self, url: str, api_key: str, collection_name: str, dimension: int):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        
        self.client = QdrantClient(url=url, api_key=api_key if api_key else None)
        self.collection_name = collection_name
        
        # Create collection if not exists
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if collection_name not in collection_names:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
            )
    
    def upsert_vectors(self, ids: List[str], vectors: List[List[float]], metadata: List[dict]) -> bool:
        from qdrant_client.models import PointStruct
        
        points = [
            PointStruct(id=id_, vector=vec, payload=meta)
            for id_, vec, meta in zip(ids, vectors, metadata)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
        return True
    
    def delete_vectors(self, ids: List[str]) -> bool:
        """Delete vectors by their IDs."""
        if not ids:
            return True
        
        # Qdrant expects list of IDs directly for point deletion
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=ids
        )
        return True
    
    def search(self, query_vector: List[float], top_k: int = 10, filters: Optional[dict] = None) -> List[dict]:
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=filters
        )
        return [{"id": r.id, "score": r.score, "metadata": r.payload} for r in results]
    
    def hybrid_search(self, query_vector: List[float], query_text: str, top_k: int = 10, filters: Optional[dict] = None, semantic_weight: float = 0.7, keyword_weight: float = 0.3) -> List[dict]:
        """Hybrid search combining semantic and keyword search."""
        try:
            print(f"[Hybrid] Starting hybrid search for query: {query_text}")
            
            # Semantic search
            semantic_results = self.search(query_vector, top_k=top_k * 2, filters=filters)
            print(f"[Hybrid] Semantic search returned {len(semantic_results)} results")
            
            # Keyword search (simple text matching on chunk_text in metadata)
            all_points = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                with_payload=True
            )[0]
            
            print(f"[Hybrid] Scrolled {len(all_points)} points from collection")
            
            query_lower = query_text.lower()
            keyword_scores = {}
            for point in all_points:
                chunk_text = point.payload.get("chunk_text", "").lower()
                if chunk_text:
                    # Simple keyword matching
                    score = 0
                    words = query_lower.split()
                    for word in words:
                        if len(word) > 2:  # Skip very short words
                            if word in chunk_text:
                                score += chunk_text.count(word)  # Count occurrences
                    if score > 0:
                        keyword_scores[point.id] = score
                        print(f"[Hybrid] Keyword match: {point.id} (score: {score}) - contains query words")
            
            print(f"[Hybrid] Found {len(keyword_scores)} chunks with keyword matches")
            
            # Normalize keyword scores
            if keyword_scores:
                max_keyword_score = max(keyword_scores.values())
                keyword_scores = {k: v / max_keyword_score for k, v in keyword_scores.items()}
            
            # Combine results using reciprocal rank fusion (RRF)
            combined_scores = {}
            # Add semantic scores
            for i, result in enumerate(semantic_results):
                rrf_score = 1 / (i + 1 + 60)  # RRF with k=60
                combined_scores[result["id"]] = combined_scores.get(result["id"], 0) + rrf_score * semantic_weight
            
            # Add keyword scores
            for point_id, score in keyword_scores.items():
                if point_id in combined_scores:
                    combined_scores[point_id] += score * keyword_weight
                    print(f"[Hybrid] Boosting chunk {point_id} with keyword score: {score}")
                else:
                    combined_scores[point_id] = score * keyword_weight
                    print(f"[Hybrid] Adding chunk {point_id} with keyword score: {score}")
            
            # Sort by combined score
            sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Get metadata for top results
            result_ids = [item[0] for item in sorted_results[:top_k]]
            final_results = []
            for result in semantic_results:
                if result["id"] in result_ids:
                    result["score"] = combined_scores[result["id"]]
                    final_results.append(result)
            
            print(f"[Hybrid] Returning {len(final_results)} results")
            return final_results[:top_k]
        except Exception as e:
            print(f"[Hybrid] Error in hybrid search: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to semantic search
            return self.search(query_vector, top_k=top_k, filters=filters)


def get_vector_store(config) -> VectorStore:
    """Factory function to get Qdrant vector store."""
    return QdrantStore(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        collection_name=config.QDRANT_COLLECTION_NAME,
        dimension=config.EMBEDDING_DIMENSION
    )
