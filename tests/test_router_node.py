"""
Test cases for RouterNode functionality.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from app.graph.nodes.router import RouterNode
from app.graph.state import AssistantState, QueryType, MessageRole


class TestRouterNode:
    """Test cases for RouterNode."""

    def test_router_node_initialization(self):
        """Test RouterNode initialization."""
        router = RouterNode()

        assert router is not None
        assert hasattr(router, "llm_interface")

    @pytest.mark.asyncio
    async def test_route_query_weather_classification(
        self, sample_assistant_state, mock_llm_interface
    ):
        """Test routing a weather query."""
        with patch(
            "app.graph.nodes.router.LLMInterface", return_value=mock_llm_interface
        ):
            router = RouterNode()

            # Mock the classify_query method
            mock_llm_interface.classify_query.return_value = "weather"

            result = await router.route_query(sample_assistant_state)

            assert result.query_type == QueryType.WEATHER
            assert result.error_message is None
            assert result.processing_time is not None
            assert len(result.messages) > 0

            # Check that a system message was added
            system_messages = [
                msg for msg in result.messages if msg.role == MessageRole.SYSTEM
            ]
            assert len(system_messages) == 1
            assert "Query classified as: weather" in system_messages[0].content

    @pytest.mark.asyncio
    async def test_route_query_rag_classification(
        self, sample_assistant_state, mock_llm_interface
    ):
        """Test routing a RAG query."""
        with patch(
            "app.graph.nodes.router.LLMInterface", return_value=mock_llm_interface
        ):
            router = RouterNode()

            # Mock the classify_query method
            mock_llm_interface.classify_query.return_value = "rag"

            result = await router.route_query(sample_assistant_state)

            assert result.query_type == QueryType.RAG
            assert result.error_message is None
            assert result.processing_time is not None

    @pytest.mark.asyncio
    async def test_route_query_no_query_provided(self, mock_llm_interface):
        """Test routing with no query provided."""
        with patch(
            "app.graph.nodes.router.LLMInterface", return_value=mock_llm_interface
        ):
            router = RouterNode()

            # Create state with no query
            state = AssistantState()

            result = await router.route_query(state)

            assert result.error_message == "No query provided"
            assert result.query_type == QueryType.UNKNOWN

    @pytest.mark.asyncio
    async def test_route_query_uses_last_user_message(self, mock_llm_interface):
        """Test routing uses last user message when no current query."""
        with patch(
            "app.graph.nodes.router.LLMInterface", return_value=mock_llm_interface
        ):
            router = RouterNode()

            # Create state with messages but no current query
            state = AssistantState()
            state.add_message(MessageRole.USER, "What's the weather in Paris?")
            state.add_message(MessageRole.ASSISTANT, "Let me check that for you.")

            mock_llm_interface.classify_query.return_value = "weather"

            result = await router.route_query(state)

            # Should use the last user message
            mock_llm_interface.classify_query.assert_called_with(
                "What's the weather in Paris?"
            )
            assert result.query_type == QueryType.WEATHER

    @pytest.mark.asyncio
    async def test_route_query_llm_error_handling(
        self, sample_assistant_state, mock_llm_interface
    ):
        """Test error handling when LLM classification fails."""
        with patch(
            "app.graph.nodes.router.LLMInterface", return_value=mock_llm_interface
        ):
            router = RouterNode()

            # Mock LLM to raise an exception
            mock_llm_interface.classify_query.side_effect = Exception("LLM API error")

            result = await router.route_query(sample_assistant_state)

            assert result.error_message == "Error in router node: LLM API error"
            assert result.query_type == QueryType.UNKNOWN

    @pytest.mark.asyncio
    async def test_route_query_classification_edge_cases(
        self, sample_assistant_state, mock_llm_interface
    ):
        """Test routing with various classification responses."""
        with patch(
            "app.graph.nodes.router.LLMInterface", return_value=mock_llm_interface
        ):
            router = RouterNode()

            # Test with "weather" in classification
            mock_llm_interface.classify_query.return_value = "weather information"
            result = await router.route_query(sample_assistant_state)
            assert result.query_type == QueryType.WEATHER

            # Test with "rag" in classification
            mock_llm_interface.classify_query.return_value = "rag retrieval"
            result = await router.route_query(sample_assistant_state)
            assert result.query_type == QueryType.RAG

            # Test with unknown classification
            mock_llm_interface.classify_query.return_value = "unknown query type"
            result = await router.route_query(sample_assistant_state)
            assert result.query_type == QueryType.RAG  # Default to RAG

    @pytest.mark.asyncio
    async def test_route_query_processing_time(
        self, sample_assistant_state, mock_llm_interface
    ):
        """Test that processing time is recorded."""
        with patch(
            "app.graph.nodes.router.LLMInterface", return_value=mock_llm_interface
        ):
            router = RouterNode()

            mock_llm_interface.classify_query.return_value = "weather"

            result = await router.route_query(sample_assistant_state)

            assert result.processing_time is not None
            assert isinstance(result.processing_time, float)
            assert result.processing_time >= 0

    @pytest.mark.asyncio
    async def test_route_query_preserves_state_data(
        self, sample_assistant_state, mock_llm_interface
    ):
        """Test that routing preserves existing state data."""
        with patch(
            "app.graph.nodes.router.LLMInterface", return_value=mock_llm_interface
        ):
            router = RouterNode()

            # Add some existing data to state
            sample_assistant_state.session_id = "test_session"
            sample_assistant_state.user_id = "test_user"
            sample_assistant_state.add_message(MessageRole.USER, "Previous message")

            mock_llm_interface.classify_query.return_value = "weather"

            result = await router.route_query(sample_assistant_state)

            # Check that existing data is preserved
            assert result.session_id == "test_session"
            assert result.user_id == "test_user"
            assert (
                len(result.messages) > 1
            )  # Should have previous message plus new system message

    @pytest.mark.asyncio
    async def test_route_query_empty_string_query(self, mock_llm_interface):
        """Test routing with empty string query."""
        with patch(
            "app.graph.nodes.router.LLMInterface", return_value=mock_llm_interface
        ):
            router = RouterNode()

            state = AssistantState(current_query="")

            result = await router.route_query(state)

            assert result.error_message == "No query provided"
            assert result.query_type == QueryType.UNKNOWN

    @pytest.mark.asyncio
    async def test_route_query_system_message_format(
        self, sample_assistant_state, mock_llm_interface
    ):
        """Test that system message is properly formatted."""
        with patch(
            "app.graph.nodes.router.LLMInterface", return_value=mock_llm_interface
        ):
            router = RouterNode()

            mock_llm_interface.classify_query.return_value = "weather"

            result = await router.route_query(sample_assistant_state)

            system_messages = [
                msg for msg in result.messages if msg.role == MessageRole.SYSTEM
            ]
            assert len(system_messages) == 1

            system_message = system_messages[0]
            assert system_message.role == MessageRole.SYSTEM
            assert "Query classified as: weather" in system_message.content
            assert system_message.timestamp is not None
