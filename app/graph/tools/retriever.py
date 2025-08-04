from typing import List, Dict, Any
from app.interfaces.embeddings_interface import EmbeddingsInterface
from app.interfaces.qdrant_interface import QdrantInterface


class RetrieverTool:
    def __init__(self):
        self.embeddings_interface = EmbeddingsInterface()
        self.qdrant_interface = QdrantInterface()

    async def retrieve_relevant_documents(
        self, query: str, limit: int = 5, score_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        try:
            query_embedding = await self.embeddings_interface.generate_single_embedding(
                query
            )

            results = self.qdrant_interface.search_similar(
                query_embedding=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
            )

            return results

        except Exception as e:
            raise Exception(f"Error retrieving documents: {str(e)}")

    async def get_documents_by_source(
        self, source: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        try:
            results = self.qdrant_interface.search_by_source(source, limit=limit)
            return results
        except Exception as e:
            raise Exception(f"Error retrieving documents by source: {str(e)}")

    def format_retrieved_documents(self, documents: List[Dict[str, Any]]) -> str:
        if not documents:
            return "No relevant documents found."

        formatted_content = []
        for i, doc in enumerate(documents, 1):
            content = doc.get("content", "")
            source = doc.get("source", "Unknown")
            score = doc.get("score", 0)

            if len(content) > 500:
                content = content[:500] + "..."

            formatted_content.append(
                f"[{i}] Source: {source} (Relevance: {score:.2f})\n"
                f"Content: {content}\n"
            )

        return "\n".join(formatted_content)

    async def get_rag_context(self, query: str, max_chunks: int = 3) -> Dict[str, Any]:
        try:
            documents = await self.retrieve_relevant_documents(
                query=query,
                limit=max_chunks,
                score_threshold=0.6,
            )

            if not documents:
                return {
                    "relevant_chunks": [],
                    "sources": [],
                    "total_results": 0,
                    "average_score": 0,
                }

            relevant_chunks = [doc.get("content", "") for doc in documents]
            sources = list(set([doc.get("source", "Unknown") for doc in documents]))
            scores = [doc.get("score", 0) for doc in documents]

            return {
                "relevant_chunks": relevant_chunks,
                "sources": sources,
                "total_results": len(documents),
                "average_score": sum(scores) / len(scores) if scores else 0,
                "documents": documents,
            }

        except Exception as e:
            raise Exception(f"Error getting RAG context: {str(e)}")

    def get_retrieval_stats(self) -> Dict[str, Any]:
        try:
            collection_info = self.qdrant_interface.get_collection_info()
            return {
                "total_documents": collection_info["points_count"],
                "collection_name": collection_info["name"],
                "vectors_count": collection_info["vectors_count"],
            }
        except Exception as e:
            raise Exception(f"Error getting retrieval stats: {str(e)}")
