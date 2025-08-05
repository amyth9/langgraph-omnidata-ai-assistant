"""
Test cases for RAGNode functionality.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from app.graph.nodes.rag import RAGNode
from app.graph.state import AssistantState, QueryType, MessageRole, RAGResult


class TestRAGNode:
    """Test cases for RAGNode."""

    def test_rag_node_initialization(self):
        """Test RAGNode initialization."""
        with patch("app.graph.nodes.rag.RetrieverTool") as mock_retriever_tool, patch(
            "app.graph.nodes.rag.LLMInterface"
        ) as mock_llm_interface:

            rag_node = RAGNode()

            assert rag_node is not None
            assert hasattr(rag_node, "retriever_tool")
            assert hasattr(rag_node, "llm_interface")

    @pytest.mark.asyncio
    async def test_process_rag_query_success(
        self, sample_assistant_state, mock_retriever_tool, mock_llm_interface
    ):
        """Test successful RAG query processing."""
        with patch(
            "app.graph.nodes.rag.RetrieverTool", return_value=mock_retriever_tool
        ), patch("app.graph.nodes.rag.LLMInterface", return_value=mock_llm_interface):

            rag_node = RAGNode()

            # Mock retriever tool response
            mock_retriever_tool.get_rag_context.return_value = {
                "relevant_chunks": [
                    "This is a sample document chunk about AI and machine learning.",
                    "Another relevant chunk about natural language processing.",
                ],
                "sources": ["sample_doc.pdf", "ai_guide.pdf"],
                "total_results": 2,
                "average_score": 0.85,
            }

            # Mock LLM response
            mock_llm_interface.summarize_rag_results.return_value = "Based on the documents, AI and machine learning are important technologies."

            result = await rag_node.process_rag_query(sample_assistant_state)

            assert result.error_message is None
            assert result.rag_result is not None
            assert result.rag_result.query == "What's the weather like in London?"
            assert len(result.rag_result.relevant_chunks) == 2
            assert (
                result.rag_result.summary
                == "Based on the documents, AI and machine learning are important technologies."
            )
            assert len(result.rag_result.sources) == 2
            assert result.processing_time is not None

            # Check that assistant message was added
            assistant_messages = [
                msg for msg in result.messages if msg.role == MessageRole.ASSISTANT
            ]
            assert len(assistant_messages) == 1
            assert (
                "Based on the documents, AI and machine learning"
                in assistant_messages[0].content
            )

    @pytest.mark.asyncio
    async def test_process_rag_query_no_query_provided(
        self, mock_retriever_tool, mock_llm_interface
    ):
        """Test RAG processing with no query."""
        with patch(
            "app.graph.nodes.rag.RetrieverTool", return_value=mock_retriever_tool
        ), patch("app.graph.nodes.rag.LLMInterface", return_value=mock_llm_interface):

            rag_node = RAGNode()

            # Create state with no query
            state = AssistantState()

            result = await rag_node.process_rag_query(state)

            assert result.error_message == "No query provided for RAG processing"
            assert result.rag_result is None

    @pytest.mark.asyncio
    async def test_process_rag_query_uses_last_user_message(
        self, mock_retriever_tool, mock_llm_interface
    ):
        """Test RAG processing uses last user message when no current query."""
        with patch(
            "app.graph.nodes.rag.RetrieverTool", return_value=mock_retriever_tool
        ), patch("app.graph.nodes.rag.LLMInterface", return_value=mock_llm_interface):

            rag_node = RAGNode()

            # Create state with messages but no current query
            state = AssistantState()
            state.add_message(MessageRole.USER, "What is artificial intelligence?")

            mock_retriever_tool.get_rag_context.return_value = {
                "relevant_chunks": ["AI is artificial intelligence"],
                "sources": ["ai_doc.pdf"],
                "total_results": 1,
                "average_score": 0.9,
            }

            mock_llm_interface.summarize_rag_results.return_value = (
                "AI refers to artificial intelligence."
            )

            result = await rag_node.process_rag_query(state)

            # Should use the last user message
            mock_retriever_tool.get_rag_context.assert_called_with(
                "What is artificial intelligence?"
            )
            assert result.rag_result.query == "What is artificial intelligence?"

    @pytest.mark.asyncio
    async def test_process_rag_query_no_relevant_chunks(
        self, sample_assistant_state, mock_retriever_tool, mock_llm_interface
    ):
        """Test RAG processing when no relevant chunks are found."""
        with patch(
            "app.graph.nodes.rag.RetrieverTool", return_value=mock_retriever_tool
        ), patch("app.graph.nodes.rag.LLMInterface", return_value=mock_llm_interface):

            rag_node = RAGNode()

            # Mock retriever tool to return no relevant chunks
            mock_retriever_tool.get_rag_context.return_value = {
                "relevant_chunks": [],
                "sources": [],
                "total_results": 0,
                "average_score": 0,
            }

            result = await rag_node.process_rag_query(sample_assistant_state)

            assert result.error_message == "No relevant documents found for your query"
            assert result.rag_result is None

    @pytest.mark.asyncio
    async def test_process_rag_query_processing_time(
        self, sample_assistant_state, mock_retriever_tool, mock_llm_interface
    ):
        """Test that processing time is recorded."""
        with patch(
            "app.graph.nodes.rag.RetrieverTool", return_value=mock_retriever_tool
        ), patch("app.graph.nodes.rag.LLMInterface", return_value=mock_llm_interface):

            rag_node = RAGNode()

            mock_retriever_tool.get_rag_context.return_value = {
                "relevant_chunks": ["Sample chunk"],
                "sources": ["sample.pdf"],
                "total_results": 1,
                "average_score": 0.9,
            }

            mock_llm_interface.summarize_rag_results.return_value = "RAG summary"

            result = await rag_node.process_rag_query(sample_assistant_state)

            assert result.processing_time is not None
            assert isinstance(result.processing_time, float)
            assert result.processing_time >= 0

    @pytest.mark.asyncio
    async def test_process_rag_query_preserves_state_data(
        self, sample_assistant_state, mock_retriever_tool, mock_llm_interface
    ):
        """Test that RAG processing preserves existing state data."""
        with patch(
            "app.graph.nodes.rag.RetrieverTool", return_value=mock_retriever_tool
        ), patch("app.graph.nodes.rag.LLMInterface", return_value=mock_llm_interface):

            rag_node = RAGNode()

            # Add some existing data to state
            sample_assistant_state.session_id = "test_session"
            sample_assistant_state.user_id = "test_user"
            sample_assistant_state.add_message(MessageRole.USER, "Previous message")

            mock_retriever_tool.get_rag_context.return_value = {
                "relevant_chunks": ["Sample chunk"],
                "sources": ["sample.pdf"],
                "total_results": 1,
                "average_score": 0.9,
            }

            mock_llm_interface.summarize_rag_results.return_value = "RAG summary"

            result = await rag_node.process_rag_query(sample_assistant_state)

            # Check that existing data is preserved
            assert result.session_id == "test_session"
            assert result.user_id == "test_user"
            assert (
                len(result.messages) > 1
            )  # Should have previous message plus new assistant message

    def test_format_rag_response(self, sample_assistant_state, sample_rag_context):
        """Test formatting RAG response."""
        with patch("app.graph.nodes.rag.RetrieverTool"), patch(
            "app.graph.nodes.rag.LLMInterface"
        ):

            rag_node = RAGNode()

            # Set RAG result in state
            sample_assistant_state.rag_result = RAGResult(
                query="What is AI?",
                relevant_chunks=sample_rag_context["relevant_chunks"],
                summary="AI refers to artificial intelligence technologies.",
                sources=sample_rag_context["sources"],
            )

            formatted_response = rag_node.format_rag_response(sample_assistant_state)

            assert isinstance(formatted_response, str)
            # Should contain information about the response

    def test_format_rag_response_no_rag_result(self, sample_assistant_state):
        """Test formatting RAG response when no RAG result exists."""
        with patch("app.graph.nodes.rag.RetrieverTool"), patch(
            "app.graph.nodes.rag.LLMInterface"
        ):

            rag_node = RAGNode()

            formatted_response = rag_node.format_rag_response(sample_assistant_state)

            assert isinstance(formatted_response, str)
            # Should handle the case gracefully

    def test_get_retrieval_stats(self, sample_assistant_state, sample_rag_context):
        """Test getting retrieval statistics."""
        with patch("app.graph.nodes.rag.RetrieverTool"), patch(
            "app.graph.nodes.rag.LLMInterface"
        ):

            rag_node = RAGNode()

            # Set RAG result in state
            sample_assistant_state.rag_result = RAGResult(
                query="What is AI?",
                relevant_chunks=sample_rag_context["relevant_chunks"],
                summary="AI refers to artificial intelligence technologies.",
                sources=sample_rag_context["sources"],
            )

            stats = rag_node.get_retrieval_stats(sample_assistant_state)

            assert isinstance(stats, dict)
            assert "total_chunks" in stats
            assert "sources" in stats
            assert "query" in stats
            assert stats["total_chunks"] == 3
            assert len(stats["sources"]) == 2
            assert stats["query"] == "What is AI?"

    def test_get_retrieval_stats_no_rag_result(self, sample_assistant_state):
        """Test getting retrieval statistics when no RAG result exists."""
        with patch("app.graph.nodes.rag.RetrieverTool"), patch(
            "app.graph.nodes.rag.LLMInterface"
        ):

            rag_node = RAGNode()

            stats = rag_node.get_retrieval_stats(sample_assistant_state)

            assert isinstance(stats, dict)
            assert stats == {}  # Should return empty dict when no RAG result

    @pytest.mark.asyncio
    async def test_process_rag_query_empty_string_query(
        self, mock_retriever_tool, mock_llm_interface
    ):
        """Test RAG processing with empty string query."""
        with patch(
            "app.graph.nodes.rag.RetrieverTool", return_value=mock_retriever_tool
        ), patch("app.graph.nodes.rag.LLMInterface", return_value=mock_llm_interface):

            rag_node = RAGNode()

            state = AssistantState(current_query="")

            result = await rag_node.process_rag_query(state)

            assert result.error_message == "No query provided for RAG processing"
            assert result.rag_result is None

    @pytest.mark.asyncio
    async def test_process_rag_query_assistant_message_format(
        self, sample_assistant_state, mock_retriever_tool, mock_llm_interface
    ):
        """Test that assistant message is properly formatted."""
        with patch(
            "app.graph.nodes.rag.RetrieverTool", return_value=mock_retriever_tool
        ), patch("app.graph.nodes.rag.LLMInterface", return_value=mock_llm_interface):

            rag_node = RAGNode()

            mock_retriever_tool.get_rag_context.return_value = {
                "relevant_chunks": ["This is a sample document chunk about AI."],
                "sources": ["sample_doc.pdf"],
                "total_results": 1,
                "average_score": 0.9,
            }

            mock_llm_interface.summarize_rag_results.return_value = (
                "Based on the documents, AI is artificial intelligence."
            )

            result = await rag_node.process_rag_query(sample_assistant_state)

            assistant_messages = [
                msg for msg in result.messages if msg.role == MessageRole.ASSISTANT
            ]
            assert len(assistant_messages) == 1

            assistant_message = assistant_messages[0]
            assert assistant_message.role == MessageRole.ASSISTANT
            assert (
                "Based on the documents, AI is artificial intelligence"
                in assistant_message.content
            )
            assert assistant_message.timestamp is not None

    @pytest.mark.asyncio
    async def test_process_rag_query_with_multiple_chunks(
        self, sample_assistant_state, mock_retriever_tool, mock_llm_interface
    ):
        """Test RAG processing with multiple relevant chunks."""
        with patch(
            "app.graph.nodes.rag.RetrieverTool", return_value=mock_retriever_tool
        ), patch("app.graph.nodes.rag.LLMInterface", return_value=mock_llm_interface):

            rag_node = RAGNode()

            mock_retriever_tool.get_rag_context.return_value = {
                "relevant_chunks": [
                    "First chunk about AI",
                    "Second chunk about machine learning",
                    "Third chunk about deep learning",
                ],
                "sources": ["doc1.pdf", "doc2.pdf", "doc3.pdf"],
                "total_results": 3,
                "average_score": 0.85,
            }

            mock_llm_interface.summarize_rag_results.return_value = (
                "Comprehensive summary of AI technologies."
            )

            result = await rag_node.process_rag_query(sample_assistant_state)

            assert result.rag_result is not None
            assert len(result.rag_result.relevant_chunks) == 3
            assert len(result.rag_result.sources) == 3
            assert (
                result.rag_result.summary == "Comprehensive summary of AI technologies."
            )

    @pytest.mark.asyncio
    async def test_process_rag_query_with_single_chunk(
        self, sample_assistant_state, mock_retriever_tool, mock_llm_interface
    ):
        """Test RAG processing with single relevant chunk."""
        with patch(
            "app.graph.nodes.rag.RetrieverTool", return_value=mock_retriever_tool
        ), patch("app.graph.nodes.rag.LLMInterface", return_value=mock_llm_interface):

            rag_node = RAGNode()

            mock_retriever_tool.get_rag_context.return_value = {
                "relevant_chunks": ["Single relevant chunk about AI"],
                "sources": ["single_doc.pdf"],
                "total_results": 1,
                "average_score": 0.95,
            }

            mock_llm_interface.summarize_rag_results.return_value = (
                "Summary based on single document."
            )

            result = await rag_node.process_rag_query(sample_assistant_state)

            assert result.rag_result is not None
            assert len(result.rag_result.relevant_chunks) == 1
            assert len(result.rag_result.sources) == 1
            assert result.rag_result.summary == "Summary based on single document."
