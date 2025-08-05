"""
Test cases for pipeline integration and end-to-end functionality.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from app.graph.nodes.router import RouterNode
from app.graph.nodes.weather import WeatherNode
from app.graph.nodes.rag import RAGNode
from app.graph.state import AssistantState, QueryType, MessageRole


class TestPipelineIntegration:
    """Test cases for pipeline integration."""

    @pytest.mark.asyncio
    async def test_complete_weather_pipeline(
        self, mock_llm_interface, mock_weather_tool
    ):
        """Test complete weather pipeline flow."""
        with patch(
            "app.graph.nodes.router.LLMInterface", return_value=mock_llm_interface
        ), patch(
            "app.graph.nodes.weather.WeatherTool", return_value=mock_weather_tool
        ), patch(
            "app.graph.nodes.weather.LLMInterface", return_value=mock_llm_interface
        ):

            # Create initial state
            state = AssistantState(current_query="What's the weather in London?")

            # Step 1: Router classification
            router = RouterNode()
            mock_llm_interface.classify_query.return_value = "weather"

            state = await router.route_query(state)

            assert state.query_type == QueryType.WEATHER
            assert state.error_message is None

            # Step 2: Weather processing
            weather_node = WeatherNode()
            mock_weather_tool.get_weather_for_query.return_value = {
                "city": "London",
                "temperature": 15.5,
                "feels_like": 14.2,
                "description": "Partly cloudy",
                "humidity": 65,
                "wind_speed": 12.3,
                "pressure": 1013,
            }

            mock_llm_interface.summarize_weather_data.return_value = (
                "The weather in London is partly cloudy with a temperature of 15.5°C."
            )

            state = await weather_node.process_weather_query(state)

            assert state.weather_data is not None
            assert state.weather_data.city == "London"
            assert state.weather_data.temperature == 15.5
            assert state.error_message is None

            # Check that messages were added
            assert len(state.messages) > 0
            system_messages = [
                msg for msg in state.messages if msg.role == MessageRole.SYSTEM
            ]
            assistant_messages = [
                msg for msg in state.messages if msg.role == MessageRole.ASSISTANT
            ]
            assert len(system_messages) > 0
            assert len(assistant_messages) > 0

    @pytest.mark.asyncio
    async def test_complete_rag_pipeline(self, mock_llm_interface, mock_retriever_tool):
        """Test complete RAG pipeline flow."""
        with patch(
            "app.graph.nodes.router.LLMInterface", return_value=mock_llm_interface
        ), patch(
            "app.graph.nodes.rag.RetrieverTool", return_value=mock_retriever_tool
        ), patch(
            "app.graph.nodes.rag.LLMInterface", return_value=mock_llm_interface
        ):

            # Create initial state
            state = AssistantState(current_query="What is artificial intelligence?")

            # Step 1: Router classification
            router = RouterNode()
            mock_llm_interface.classify_query.return_value = "rag"

            state = await router.route_query(state)

            assert state.query_type == QueryType.RAG
            assert state.error_message is None

            # Step 2: RAG processing
            rag_node = RAGNode()
            mock_retriever_tool.get_rag_context.return_value = {
                "relevant_chunks": [
                    "This is a sample document chunk about AI and machine learning.",
                    "Another relevant chunk about natural language processing.",
                ],
                "sources": ["sample_doc.pdf", "ai_guide.pdf"],
                "total_results": 2,
                "average_score": 0.85,
            }

            mock_llm_interface.summarize_rag_results.return_value = "Based on the documents, AI and machine learning are important technologies."

            state = await rag_node.process_rag_query(state)

            assert state.rag_result is not None
            assert state.rag_result.query == "What is artificial intelligence?"
            assert len(state.rag_result.relevant_chunks) == 2
            assert state.error_message is None

            # Check that messages were added
            assert len(state.messages) > 0
            system_messages = [
                msg for msg in state.messages if msg.role == MessageRole.SYSTEM
            ]
            assistant_messages = [
                msg for msg in state.messages if msg.role == MessageRole.ASSISTANT
            ]
            assert len(system_messages) > 0
            assert len(assistant_messages) > 0

    @pytest.mark.asyncio
    async def test_pipeline_state_preservation(
        self, mock_llm_interface, mock_weather_tool
    ):
        """Test that pipeline preserves state data throughout processing."""
        with patch(
            "app.graph.nodes.router.LLMInterface", return_value=mock_llm_interface
        ), patch(
            "app.graph.nodes.weather.WeatherTool", return_value=mock_weather_tool
        ), patch(
            "app.graph.nodes.weather.LLMInterface", return_value=mock_llm_interface
        ):

            # Create state with additional data
            state = AssistantState(
                current_query="What's the weather in London?",
                session_id="test_session_123",
                user_id="user_456",
            )
            state.add_message(MessageRole.USER, "Previous message")

            # Router processing
            router = RouterNode()
            mock_llm_interface.classify_query.return_value = "weather"

            state = await router.route_query(state)

            # Check that state data is preserved
            assert state.session_id == "test_session_123"
            assert state.user_id == "user_456"
            assert (
                len(state.messages) > 1
            )  # Should have previous message plus new system message

            # Weather processing
            weather_node = WeatherNode()
            mock_weather_tool.get_weather_for_query.return_value = {
                "city": "London",
                "temperature": 15.5,
                "feels_like": 14.2,
                "description": "Partly cloudy",
                "humidity": 65,
                "wind_speed": 12.3,
                "pressure": 1013,
            }

            mock_llm_interface.summarize_weather_data.return_value = "Weather summary"

            state = await weather_node.process_weather_query(state)

            # Check that state data is still preserved
            assert state.session_id == "test_session_123"
            assert state.user_id == "user_456"
            assert (
                len(state.messages) > 2
            )  # Should have previous messages plus new assistant message

    @pytest.mark.asyncio
    async def test_pipeline_processing_time_tracking(
        self, mock_llm_interface, mock_weather_tool
    ):
        """Test that pipeline tracks processing time."""
        with patch(
            "app.graph.nodes.router.LLMInterface", return_value=mock_llm_interface
        ), patch(
            "app.graph.nodes.weather.WeatherTool", return_value=mock_weather_tool
        ), patch(
            "app.graph.nodes.weather.LLMInterface", return_value=mock_llm_interface
        ):

            # Create initial state
            state = AssistantState(current_query="What's the weather in London?")

            # Router processing
            router = RouterNode()
            mock_llm_interface.classify_query.return_value = "weather"

            state = await router.route_query(state)

            assert state.processing_time is not None
            assert isinstance(state.processing_time, float)
            assert state.processing_time >= 0

            # Weather processing
            weather_node = WeatherNode()
            mock_weather_tool.get_weather_for_query.return_value = {
                "city": "London",
                "temperature": 15.5,
                "feels_like": 14.2,
                "description": "Partly cloudy",
                "humidity": 65,
                "wind_speed": 12.3,
                "pressure": 1013,
            }

            mock_llm_interface.summarize_weather_data.return_value = "Weather summary"

            state = await weather_node.process_weather_query(state)

            assert state.processing_time is not None
            assert isinstance(state.processing_time, float)
            assert state.processing_time >= 0

    @pytest.mark.asyncio
    async def test_pipeline_message_flow(self, mock_llm_interface, mock_weather_tool):
        """Test that pipeline properly manages message flow."""
        with patch(
            "app.graph.nodes.router.LLMInterface", return_value=mock_llm_interface
        ), patch(
            "app.graph.nodes.weather.WeatherTool", return_value=mock_weather_tool
        ), patch(
            "app.graph.nodes.weather.LLMInterface", return_value=mock_llm_interface
        ):

            # Create initial state
            state = AssistantState(current_query="What's the weather in London?")

            # Router processing
            router = RouterNode()
            mock_llm_interface.classify_query.return_value = "weather"

            state = await router.route_query(state)

            # Check that system message was added
            system_messages = [
                msg for msg in state.messages if msg.role == MessageRole.SYSTEM
            ]
            assert len(system_messages) == 1
            assert "Query classified as: weather" in system_messages[0].content

            # Weather processing
            weather_node = WeatherNode()
            mock_weather_tool.get_weather_for_query.return_value = {
                "city": "London",
                "temperature": 15.5,
                "feels_like": 14.2,
                "description": "Partly cloudy",
                "humidity": 65,
                "wind_speed": 12.3,
                "pressure": 1013,
            }

            mock_llm_interface.summarize_weather_data.return_value = "Weather summary"

            state = await weather_node.process_weather_query(state)

            # Check that assistant message was added
            assistant_messages = [
                msg for msg in state.messages if msg.role == MessageRole.ASSISTANT
            ]
            assert len(assistant_messages) == 1
            assert "Weather summary" in assistant_messages[0].content

            # Check total message count
            assert len(state.messages) == 2  # 1 system + 1 assistant

    @pytest.mark.asyncio
    async def test_pipeline_edge_case_empty_query(self, mock_llm_interface):
        """Test pipeline behavior with empty query."""
        with patch(
            "app.graph.nodes.router.LLMInterface", return_value=mock_llm_interface
        ):

            # Create state with empty query
            state = AssistantState(current_query="")

            # Router processing
            router = RouterNode()

            state = await router.route_query(state)

            # Should handle empty query gracefully
            assert state.query_type == QueryType.UNKNOWN
            assert state.error_message is not None

    @pytest.mark.asyncio
    async def test_pipeline_edge_case_unknown_classification(
        self, mock_llm_interface, mock_retriever_tool
    ):
        """Test pipeline behavior with unknown query classification."""
        with patch(
            "app.graph.nodes.router.LLMInterface", return_value=mock_llm_interface
        ), patch(
            "app.graph.nodes.rag.RetrieverTool", return_value=mock_retriever_tool
        ), patch(
            "app.graph.nodes.rag.LLMInterface", return_value=mock_llm_interface
        ):

            # Create initial state
            state = AssistantState(current_query="Random query")

            # Router processing with unknown classification
            router = RouterNode()
            mock_llm_interface.classify_query.return_value = "unknown"

            state = await router.route_query(state)

            # Should default to RAG processing
            assert state.query_type == QueryType.RAG

            # RAG processing
            rag_node = RAGNode()
            mock_retriever_tool.get_rag_context.return_value = {
                "relevant_chunks": ["Sample document"],
                "sources": ["sample.pdf"],
                "total_results": 1,
                "average_score": 0.8,
            }

            mock_llm_interface.summarize_rag_results.return_value = "Sample response"

            state = await rag_node.process_rag_query(state)

            assert state.rag_result is not None
            assert state.error_message is None
