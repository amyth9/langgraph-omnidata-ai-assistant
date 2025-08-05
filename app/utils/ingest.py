import hashlib
import uuid
from typing import List, Dict, Any, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.parsers.pdf import PDFPlumberParser
from langchain_core.document_loaders import Blob
from app.interfaces.embeddings import EmbeddingsInterface
from app.interfaces.qdrant import QdrantInterface
from app.config import config


class PDFIngester:
    def __init__(self):
        self.embeddings_interface = EmbeddingsInterface()
        self.qdrant_interface = QdrantInterface()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            parser = PDFPlumberParser()
            blob = Blob(path=pdf_path)
            document = parser.lazy_parse(blob)

            pdf_text = ""
            for page_content in document:
                pdf_text += page_content.page_content

            return pdf_text

        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")

    def chunk_text(self, text: str) -> List[str]:
        try:
            chunks = self.text_splitter.split_text(text)
            return chunks
        except Exception as e:
            raise Exception(f"Error chunking text: {str(e)}")

    def create_document_chunks(
        self,
        chunks: List[str],
        source_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        documents = []
        metadata = metadata or {}

        for i, chunk in enumerate(chunks):
            chunk_id = str(uuid.uuid4())

            document = {
                "id": chunk_id,
                "content": chunk,
                "source": source_name,
                "metadata": {
                    **metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk),
                },
            }
            documents.append(document)

        return documents

    async def process_pdf(
        self,
        pdf_path: str,
        source_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            if source_name is None:
                import os

                source_name = os.path.basename(pdf_path)

            text = self.extract_text_from_pdf(pdf_path)
            return await self.process_text(text, source_name, metadata)

        except Exception as e:
            raise Exception(f"Error processing PDF {pdf_path}: {str(e)}")

    async def process_text(
        self, text: str, source_name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        try:
            chunks = self.chunk_text(text)
            documents = self.create_document_chunks(chunks, source_name, metadata)

            embeddings = await self.embeddings_interface.generate_embeddings(
                [doc["content"] for doc in documents]
            )

            for i, doc in enumerate(documents):
                doc["embedding"] = embeddings[i]

            self.qdrant_interface.upsert_documents(documents)

            return {
                "source_name": source_name,
                "total_chunks": len(chunks),
                "total_characters": len(text),
                "average_chunk_size": (
                    sum(len(chunk) for chunk in chunks) / len(chunks) if chunks else 0
                ),
                "metadata": metadata or {},
            }

        except Exception as e:
            raise Exception(f"Error processing text for {source_name}: {str(e)}")

    def get_processing_stats(self) -> Dict[str, Any]:
        try:
            return self.qdrant_interface.get_collection_info()
        except Exception as e:
            raise Exception(f"Error getting processing stats: {str(e)}")

    def delete_source_documents(self, source_name: str) -> None:
        try:
            self.qdrant_interface.delete_documents_by_source(source_name)
        except Exception as e:
            raise Exception(f"Error deleting source documents: {str(e)}")
