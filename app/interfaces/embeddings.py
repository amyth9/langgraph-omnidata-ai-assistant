from typing import List, Dict, Any
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import config


class EmbeddingsInterface:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", google_api_key=config.google_ai.api_key
        )

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        try:
            embeddings = await self.embeddings.aembed_documents(texts)
            return embeddings
        except Exception as e:
            raise Exception(f"Error generating embeddings: {str(e)}")

    async def generate_single_embedding(self, text: str) -> List[float]:
        try:
            embedding = await self.embeddings.aembed_query(text)
            return embedding
        except Exception as e:
            raise Exception(f"Error generating single embedding: {str(e)}")

    def batch_embed_documents(
        self, documents: List[Dict[str, Any]], batch_size: int = 100
    ) -> List[Dict[str, Any]]:
        try:
            texts = [doc["content"] for doc in documents]

            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                batch_embeddings = self.embeddings.embed_documents(batch_texts)
                all_embeddings.extend(batch_embeddings)

            for i, doc in enumerate(documents):
                doc["embedding"] = all_embeddings[i]

            return documents

        except Exception as e:
            raise Exception(f"Error in batch embedding: {str(e)}")

    def get_embedding_dimension(self) -> int:
        return 768
