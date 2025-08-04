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
