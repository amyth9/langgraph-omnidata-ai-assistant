from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchRequest,
    ScrollRequest,
)
from app.config import config


def qdrant_client() -> QdrantClient:
    return QdrantClient(url=config.qdrant.endpoint, api_key=config.qdrant.api_key)


class QdrantInterface:
    def __init__(self):
        self.client = qdrant_client()
        self.collection_name = config.qdrant.collection_name
        self.embedding_dimension = 768

    def create_collection(self) -> None:
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dimension, distance=Distance.COSINE
                    ),
                )

        except Exception as e:
            raise Exception(f"Error creating collection: {str(e)}")

    def delete_collection(self) -> None:
        try:
            self.client.delete_collection(collection_name=self.collection_name)
        except Exception as e:
            raise Exception(f"Error deleting collection: {str(e)}")

    def upsert_documents(
        self, documents: List[Dict[str, Any]], batch_size: int = 100
    ) -> None:
        try:
            self.create_collection()

            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i : i + batch_size]

                points = []
                for doc in batch_docs:
                    point = PointStruct(
                        id=doc["id"],
                        vector=doc["embedding"],
                        payload={
                            "content": doc["content"],
                            "metadata": doc.get("metadata", {}),
                            "source": doc.get("source", "unknown"),
                            "chunk_index": doc.get("metadata", {}).get(
                                "chunk_index", 0
                            ),
                        },
                    )
                    points.append(point)

                self.client.upsert(collection_name=self.collection_name, points=points)

        except Exception as e:
            raise Exception(f"Error upserting documents: {str(e)}")

    def search_similar(
        self, query_embedding: List[float], limit: int = 5, score_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        try:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
            )

            results = []
            for result in search_result:
                results.append(
                    {
                        "id": result.id,
                        "score": result.score,
                        "content": result.payload.get("content", ""),
                        "metadata": result.payload.get("metadata", {}),
                        "source": result.payload.get("source", "unknown"),
                    }
                )

            return results

        except Exception as e:
            raise Exception(f"Error searching similar documents: {str(e)}")

    def search_by_source(self, source: str, limit: int = 100) -> List[Dict[str, Any]]:
        try:
            search_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="source", match=MatchValue(value=source))]
                ),
                limit=limit,
            )

            results = []
            for result in search_result[0]:
                results.append(
                    {
                        "id": result.id,
                        "content": result.payload.get("content", ""),
                        "metadata": result.payload.get("metadata", {}),
                        "source": result.payload.get("source", "unknown"),
                    }
                )

            return results

        except Exception as e:
            return self._search_by_source_fallback(source, limit)

    def _search_by_source_fallback(
        self, source: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        try:
            search_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
            )

            results = []
            for result in search_result[0]:
                if result.payload.get("source") == source:
                    results.append(
                        {
                            "id": result.id,
                            "content": result.payload.get("content", ""),
                            "metadata": result.payload.get("metadata", {}),
                            "source": result.payload.get("source", "unknown"),
                        }
                    )

            return results

        except Exception as e:
            raise Exception(f"Error in fallback search by source: {str(e)}")

    def get_collection_info(self) -> Dict[str, Any]:
        try:
            collection_info = self.client.get_collection(self.collection_name)
            collection_stats = self.client.get_collection(self.collection_name)

            return {
                "name": collection_info.name,
                "status": collection_info.status,
                "points_count": collection_stats.points_count,
                "vectors_count": collection_stats.vectors_count,
                "segments_count": collection_stats.segments_count,
            }

        except Exception as e:
            raise Exception(f"Error getting collection info: {str(e)}")

    def delete_documents_by_source(self, source: str) -> None:
        try:
            documents = self.search_by_source(source, limit=1000)
            if documents:
                point_ids = [doc["id"] for doc in documents]
                self.client.delete(
                    collection_name=self.collection_name, points_selector=point_ids
                )

        except Exception as e:
            raise Exception(f"Error deleting documents by source: {str(e)}")

    def clear_collection(self) -> None:
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            self.create_collection()

        except Exception as e:
            raise Exception(f"Error clearing collection: {str(e)}")

    def collection_exists(self) -> bool:
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            return self.collection_name in collection_names
        except Exception:
            return False
