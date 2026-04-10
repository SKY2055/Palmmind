"""Vector store abstraction — Qdrant implementation."""

from typing import List, Optional
from abc import ABC, abstractmethod


class VectorStore(ABC):
    @abstractmethod
    def upsert_vectors(
        self,
        ids: List[str],
        vectors: List[List[float]],
        metadata: List[dict],
    ) -> bool:
        pass

    @abstractmethod
    def delete_vectors(self, ids: List[str]) -> bool:
        pass

    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[dict] = None,
    ) -> List[dict]:
        pass


class QdrantStore(VectorStore):
    def __init__(
        self,
        url: str,
        api_key: str,
        collection_name: str,
        dimension: int,
    ):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.client = QdrantClient(
            url=url,
            api_key=api_key if api_key else None,
        )
        self.collection_name = collection_name

        existing = [c.name for c in self.client.get_collections().collections]
        if collection_name not in existing:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )
            print(f"[Qdrant] Created collection '{collection_name}' (dim={dimension})")
        else:
            print(f"[Qdrant] Using existing collection '{collection_name}'")

    # ------------------------------------------------------------------

    def upsert_vectors(
        self,
        ids: List[str],
        vectors: List[List[float]],
        metadata: List[dict],
    ) -> bool:
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(id=id_, vector=vec, payload=meta)
            for id_, vec, meta in zip(ids, vectors, metadata)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
        return True

    def delete_vectors(self, ids: List[str]) -> bool:
        """
        Delete points by ID list.

        FIX: Qdrant requires a PointIdsList selector — passing raw strings
        directly raises a validation error.
        """
        if not ids:
            return True

        from qdrant_client.models import PointIdsList

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=ids),
        )
        return True

    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[dict] = None,
    ) -> List[dict]:
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=filters,
        )
        return [
            {"id": r.id, "score": r.score, "metadata": r.payload}
            for r in results
        ]

    # ------------------------------------------------------------------
    # Hybrid search (semantic + keyword RRF)
    # ------------------------------------------------------------------

    def hybrid_search(
        self,
        query_vector: List[float],
        query_text: str,
        top_k: int = 10,
        filters: Optional[dict] = None,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ) -> List[dict]:
        """Hybrid search combining cosine similarity and keyword frequency."""
        try:
            semantic_results = self.search(query_vector, top_k=top_k * 2, filters=filters)

            # Scroll for keyword matching
            all_points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=2000,
                with_payload=True,
                with_vectors=False,
            )

            query_lower = query_text.lower()
            query_words = [w for w in query_lower.split() if len(w) > 2]

            keyword_scores: dict = {}
            for point in all_points:
                chunk_text = point.payload.get("chunk_text", "").lower()
                if not chunk_text:
                    continue
                score = sum(chunk_text.count(w) for w in query_words)
                if score > 0:
                    keyword_scores[point.id] = score

            # Normalise keyword scores to [0, 1]
            if keyword_scores:
                max_ks = max(keyword_scores.values())
                keyword_scores = {k: v / max_ks for k, v in keyword_scores.items()}

            # RRF fusion
            combined: dict = {}
            for rank, result in enumerate(semantic_results):
                rrf = 1.0 / (rank + 1 + 60)
                combined[result["id"]] = combined.get(result["id"], 0) + rrf * semantic_weight

            for pid, ks in keyword_scores.items():
                combined[pid] = combined.get(pid, 0) + ks * keyword_weight

            # Sort and return top-k with metadata
            sorted_ids = sorted(combined, key=lambda x: combined[x], reverse=True)[:top_k]
            id_set = set(sorted_ids)

            final = [r for r in semantic_results if r["id"] in id_set]
            for r in final:
                r["score"] = combined[r["id"]]

            return final[:top_k]

        except Exception as exc:
            print(f"[Qdrant] hybrid_search error: {exc}; falling back to semantic")
            return self.search(query_vector, top_k=top_k, filters=filters)


def get_vector_store(config) -> QdrantStore:
    return QdrantStore(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        collection_name=config.QDRANT_COLLECTION_NAME,
        dimension=config.EMBEDDING_DIMENSION,
    )