"""
Test cases for RetrieverTool functionality.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from app.graph.tools.retriever import RetrieverTool


class TestRetrieverTool:
    """Test cases for RetrieverTool."""

    def test_retriever_tool_initialization(self):
        """Test RetrieverTool initialization."""
        with patch(
            "app.graph.tools.retriever.EmbeddingsInterface"
        ) as mock_embeddings, patch(
            "app.graph.tools.retriever.QdrantInterface"
        ) as mock_qdrant:

            retriever_tool = RetrieverTool()

            assert retriever_tool is not None
            assert hasattr(retriever_tool, "embeddings_interface")
            assert hasattr(retriever_tool, "qdrant_interface")

    @pytest.mark.asyncio
    async def test_retrieve_relevant_documents_success(
        self, mock_embeddings_interface, mock_qdrant_interface
    ):
        """Test successful document retrieval."""
        with patch(
            "app.graph.tools.retriever.EmbeddingsInterface",
            return_value=mock_embeddings_interface,
        ), patch(
            "app.graph.tools.retriever.QdrantInterface",
            return_value=mock_qdrant_interface,
        ):

            retriever_tool = RetrieverTool()

            # Mock embeddings response
            mock_embeddings_interface.generate_single_embedding.return_value = [
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
            ]

            # Mock Qdrant search response
            mock_qdrant_interface.search_similar.return_value = [
                {
                    "content": "This is a sample document chunk about AI and machine learning.",
                    "source": "sample_doc.pdf",
                    "score": 0.9,
                },
                {
                    "content": "Another relevant chunk about natural language processing.",
                    "source": "ai_guide.pdf",
                    "score": 0.8,
                },
            ]

            result = await retriever_tool.retrieve_relevant_documents("What is AI?")

            assert len(result) == 2
            assert (
                result[0]["content"]
                == "This is a sample document chunk about AI and machine learning."
            )
            assert result[0]["source"] == "sample_doc.pdf"
            assert result[0]["score"] == 0.9
            assert (
                result[1]["content"]
                == "Another relevant chunk about natural language processing."
            )
            assert result[1]["source"] == "ai_guide.pdf"
            assert result[1]["score"] == 0.8

    @pytest.mark.asyncio
    async def test_get_documents_by_source_success(self, mock_qdrant_interface):
        """Test retrieving documents by source."""
        with patch("app.graph.tools.retriever.EmbeddingsInterface"), patch(
            "app.graph.tools.retriever.QdrantInterface",
            return_value=mock_qdrant_interface,
        ):

            retriever_tool = RetrieverTool()

            # Mock Qdrant search by source response
            mock_qdrant_interface.search_by_source.return_value = [
                {
                    "content": "Document from specific source.",
                    "source": "sample_doc.pdf",
                    "score": 0.8,
                }
            ]

            result = await retriever_tool.get_documents_by_source("sample_doc.pdf")

            assert len(result) == 1
            assert result[0]["content"] == "Document from specific source."
            assert result[0]["source"] == "sample_doc.pdf"
            assert result[0]["score"] == 0.8

    @pytest.mark.asyncio
    async def test_get_documents_by_source_error(self, mock_qdrant_interface):
        """Test retrieving documents by source when Qdrant fails."""
        with patch("app.graph.tools.retriever.EmbeddingsInterface"), patch(
            "app.graph.tools.retriever.QdrantInterface",
            return_value=mock_qdrant_interface,
        ):

            retriever_tool = RetrieverTool()

            # Mock Qdrant to raise an exception
            mock_qdrant_interface.search_by_source.side_effect = Exception(
                "Qdrant API error"
            )

            with pytest.raises(
                Exception,
                match="Error retrieving documents by source: Qdrant API error",
            ):
                await retriever_tool.get_documents_by_source("sample_doc.pdf")

    def test_format_retrieved_documents_with_documents(
        self, mock_embeddings_interface, mock_qdrant_interface
    ):
        """Test formatting retrieved documents."""
        with patch(
            "app.graph.tools.retriever.EmbeddingsInterface",
            return_value=mock_embeddings_interface,
        ), patch(
            "app.graph.tools.retriever.QdrantInterface",
            return_value=mock_qdrant_interface,
        ):

            retriever_tool = RetrieverTool()

            documents = [
                {
                    "content": "This is a sample document chunk about AI and machine learning.",
                    "source": "sample_doc.pdf",
                    "score": 0.9,
                },
                {
                    "content": "Another relevant chunk about natural language processing.",
                    "source": "ai_guide.pdf",
                    "score": 0.8,
                },
            ]

            formatted_result = retriever_tool.format_retrieved_documents(documents)

            assert isinstance(formatted_result, str)
            assert "sample_doc.pdf" in formatted_result
            assert "ai_guide.pdf" in formatted_result
            assert "0.90" in formatted_result
            assert "0.80" in formatted_result
            assert "AI and machine learning" in formatted_result
            assert "natural language processing" in formatted_result

    def test_format_retrieved_documents_empty_list(
        self, mock_embeddings_interface, mock_qdrant_interface
    ):
        """Test formatting empty document list."""
        with patch(
            "app.graph.tools.retriever.EmbeddingsInterface",
            return_value=mock_embeddings_interface,
        ), patch(
            "app.graph.tools.retriever.QdrantInterface",
            return_value=mock_qdrant_interface,
        ):

            retriever_tool = RetrieverTool()

            formatted_result = retriever_tool.format_retrieved_documents([])

            assert formatted_result == "No relevant documents found."

    def test_format_retrieved_documents_long_content(
        self, mock_embeddings_interface, mock_qdrant_interface
    ):
        """Test formatting documents with long content."""
        with patch(
            "app.graph.tools.retriever.EmbeddingsInterface",
            return_value=mock_embeddings_interface,
        ), patch(
            "app.graph.tools.retriever.QdrantInterface",
            return_value=mock_qdrant_interface,
        ):

            retriever_tool = RetrieverTool()

            # Create a document with very long content
            long_content = "A" * 600  # Longer than 500 characters
            documents = [
                {"content": long_content, "source": "long_doc.pdf", "score": 0.9}
            ]

            formatted_result = retriever_tool.format_retrieved_documents(documents)

            assert "..." in formatted_result  # Should be truncated
            assert (
                len(formatted_result) < len(long_content) + 100
            )  # Should be significantly shorter

    def test_get_retrieval_stats_success(self, mock_qdrant_interface):
        """Test getting retrieval statistics."""
        with patch("app.graph.tools.retriever.EmbeddingsInterface"), patch(
            "app.graph.tools.retriever.QdrantInterface",
            return_value=mock_qdrant_interface,
        ):

            retriever_tool = RetrieverTool()

            # Mock Qdrant collection info
            mock_qdrant_interface.get_collection_info.return_value = {
                "points_count": 100,
                "name": "test_collection",
                "vectors_count": 100,
            }

            stats = retriever_tool.get_retrieval_stats()

            assert stats["total_documents"] == 100
            assert stats["collection_name"] == "test_collection"
            assert stats["vectors_count"] == 100

    def test_get_retrieval_stats_error(self, mock_qdrant_interface):
        """Test getting retrieval statistics when Qdrant fails."""
        with patch("app.graph.tools.retriever.EmbeddingsInterface"), patch(
            "app.graph.tools.retriever.QdrantInterface",
            return_value=mock_qdrant_interface,
        ):

            retriever_tool = RetrieverTool()

            # Mock Qdrant to raise an exception
            mock_qdrant_interface.get_collection_info.side_effect = Exception(
                "Qdrant API error"
            )

            with pytest.raises(
                Exception, match="Error getting retrieval stats: Qdrant API error"
            ):
                retriever_tool.get_retrieval_stats()

    @pytest.mark.asyncio
    async def test_retrieve_relevant_documents_with_custom_parameters(
        self, mock_embeddings_interface, mock_qdrant_interface
    ):
        """Test document retrieval with custom limit and score threshold."""
        with patch(
            "app.graph.tools.retriever.EmbeddingsInterface",
            return_value=mock_embeddings_interface,
        ), patch(
            "app.graph.tools.retriever.QdrantInterface",
            return_value=mock_qdrant_interface,
        ):

            retriever_tool = RetrieverTool()

            # Mock embeddings response
            mock_embeddings_interface.generate_single_embedding.return_value = [
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
            ]

            # Mock Qdrant search response
            mock_qdrant_interface.search_similar.return_value = [
                {"content": "Sample document", "source": "sample.pdf", "score": 0.9}
            ]

            result = await retriever_tool.retrieve_relevant_documents(
                query="What is AI?", limit=10, score_threshold=0.8
            )

            # Verify that Qdrant was called with correct parameters
            mock_qdrant_interface.search_similar.assert_called_with(
                query_embedding=[0.1, 0.2, 0.3, 0.4, 0.5], limit=10, score_threshold=0.8
            )

            assert len(result) == 1
            assert result[0]["content"] == "Sample document"

    @pytest.mark.asyncio
    async def test_get_documents_by_source_with_custom_limit(
        self, mock_qdrant_interface
    ):
        """Test retrieving documents by source with custom limit."""
        with patch("app.graph.tools.retriever.EmbeddingsInterface"), patch(
            "app.graph.tools.retriever.QdrantInterface",
            return_value=mock_qdrant_interface,
        ):

            retriever_tool = RetrieverTool()

            # Mock Qdrant search by source response
            mock_qdrant_interface.search_by_source.return_value = [
                {
                    "content": "Document from source",
                    "source": "sample_doc.pdf",
                    "score": 0.8,
                }
            ]

            result = await retriever_tool.get_documents_by_source(
                "sample_doc.pdf", limit=5
            )

            # Verify that Qdrant was called with correct parameters
            mock_qdrant_interface.search_by_source.assert_called_with(
                "sample_doc.pdf", limit=5
            )

            assert len(result) == 1
            assert result[0]["content"] == "Document from source"
