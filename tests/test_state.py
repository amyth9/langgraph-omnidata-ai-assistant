"""
Test cases for state management and data models.
"""

import pytest
from datetime import datetime
from typing import Dict, Any

from app.graph.state import (
    AssistantState,
    QueryType,
    MessageRole,
    WeatherData,
    RAGResult,
    Message,
)


class TestMessage:
    """Test cases for Message model."""

    def test_message_creation(self):
        """Test creating a message with required fields."""
        message = Message(role=MessageRole.USER, content="Hello, world!")

        assert message.role == MessageRole.USER
        assert message.content == "Hello, world!"
        assert message.timestamp is None

    def test_message_with_timestamp(self):
        """Test creating a message with timestamp."""
        timestamp = datetime.now().isoformat()
        message = Message(
            role=MessageRole.ASSISTANT, content="Response message", timestamp=timestamp
        )

        assert message.role == MessageRole.ASSISTANT
        assert message.content == "Response message"
        assert message.timestamp == timestamp


class TestWeatherData:
    """Test cases for WeatherData model."""

    def test_weather_data_creation(self, sample_weather_data):
        """Test creating WeatherData from sample data."""
        weather_data = WeatherData(**sample_weather_data)

        assert weather_data.city == "London"
        assert weather_data.temperature == 15.5
        assert weather_data.feels_like == 14.2
        assert weather_data.description == "Partly cloudy"
        assert weather_data.humidity == 65
        assert weather_data.wind_speed == 12.3
        assert weather_data.pressure == 1013
        assert isinstance(weather_data.raw_data, dict)

    def test_weather_data_defaults(self):
        """Test WeatherData with minimal data."""
        minimal_data = {
            "city": "Paris",
            "temperature": 20.0,
            "feels_like": 19.0,
            "description": "Sunny",
            "humidity": 50,
            "wind_speed": 5.0,
            "pressure": 1015,
            "raw_data": {},
        }

        weather_data = WeatherData(**minimal_data)
        assert weather_data.city == "Paris"
        assert weather_data.temperature == 20.0


class TestRAGResult:
    """Test cases for RAGResult model."""

    def test_rag_result_creation(self):
        """Test creating RAGResult with required fields."""
        rag_result = RAGResult(
            query="What is AI?",
            relevant_chunks=[
                "AI is artificial intelligence",
                "Machine learning is a subset of AI",
            ],
            summary="AI refers to artificial intelligence technologies.",
            sources=["doc1.pdf", "doc2.pdf"],
        )

        assert rag_result.query == "What is AI?"
        assert len(rag_result.relevant_chunks) == 2
        assert (
            rag_result.summary == "AI refers to artificial intelligence technologies."
        )
        assert len(rag_result.sources) == 2

    def test_rag_result_default_sources(self):
        """Test RAGResult with default empty sources."""
        rag_result = RAGResult(
            query="Test query", relevant_chunks=["Test chunk"], summary="Test summary"
        )

        assert rag_result.sources == []


class TestAssistantState:
    """Test cases for AssistantState model."""

    def test_assistant_state_creation(self):
        """Test creating AssistantState with default values."""
        state = AssistantState()

        assert state.messages == []
        assert state.current_query == ""
        assert state.query_type == QueryType.UNKNOWN
        assert state.weather_data is None
        assert state.rag_result is None
        assert state.error_message is None
        assert state.processing_time is None
        assert state.session_id is None
        assert state.user_id is None

    def test_assistant_state_with_initial_data(self):
        """Test creating AssistantState with initial data."""
        state = AssistantState(
            current_query="What's the weather?",
            query_type=QueryType.WEATHER,
            session_id="test_session_123",
            user_id="user_456",
        )

        assert state.current_query == "What's the weather?"
        assert state.query_type == QueryType.WEATHER
        assert state.session_id == "test_session_123"
        assert state.user_id == "user_456"

    def test_add_message(self):
        """Test adding messages to the state."""
        state = AssistantState()

        state.add_message(MessageRole.USER, "Hello")
        state.add_message(MessageRole.ASSISTANT, "Hi there!")

        assert len(state.messages) == 2
        assert state.messages[0].role == MessageRole.USER
        assert state.messages[0].content == "Hello"
        assert state.messages[1].role == MessageRole.ASSISTANT
        assert state.messages[1].content == "Hi there!"
        assert state.messages[0].timestamp is not None
        assert state.messages[1].timestamp is not None

    def test_get_last_user_message(self):
        """Test getting the last user message."""
        state = AssistantState()

        # No messages
        assert state.get_last_user_message() is None

        # Add messages
        state.add_message(MessageRole.USER, "First user message")
        state.add_message(MessageRole.ASSISTANT, "Assistant response")
        state.add_message(MessageRole.USER, "Second user message")

        assert state.get_last_user_message() == "Second user message"

    def test_get_last_user_message_no_user_messages(self):
        """Test getting last user message when no user messages exist."""
        state = AssistantState()
        state.add_message(MessageRole.ASSISTANT, "Only assistant message")

        assert state.get_last_user_message() is None

    def test_get_conversation_history(self):
        """Test getting conversation history."""
        state = AssistantState()

        # Add multiple messages
        state.add_message(MessageRole.USER, "User message 1")
        state.add_message(MessageRole.ASSISTANT, "Assistant response 1")
        state.add_message(MessageRole.USER, "User message 2")
        state.add_message(MessageRole.ASSISTANT, "Assistant response 2")
        state.add_message(MessageRole.SYSTEM, "System message")

        history = state.get_conversation_history()

        # Should contain all messages in order
        assert "user: User message 1" in history
        assert "assistant: Assistant response 1" in history
        assert "user: User message 2" in history
        assert "assistant: Assistant response 2" in history
        assert "system: System message" in history

    def test_get_conversation_history_limit(self):
        """Test conversation history is limited to last 10 messages."""
        state = AssistantState()

        # Add more than 10 messages
        for i in range(15):
            state.add_message(MessageRole.USER, f"Message {i}")

        history = state.get_conversation_history()

        # Should only contain last 10 messages
        lines = history.split("\n")
        assert len(lines) == 10
        assert "Message 5" in history  # Should include some early messages
        assert "Message 14" in history  # Should include latest messages

    def test_clear_processing_data(self):
        """Test clearing processing data."""
        state = AssistantState()

        # Set some processing data
        state.weather_data = WeatherData(
            city="Test",
            temperature=20.0,
            feels_like=19.0,
            description="Test",
            humidity=50,
            wind_speed=5.0,
            pressure=1015,
            raw_data={},
        )
        state.rag_result = RAGResult(
            query="Test", relevant_chunks=["Test"], summary="Test"
        )
        state.error_message = "Test error"
        state.processing_time = 1.5

        # Clear processing data
        state.clear_processing_data()

        assert state.weather_data is None
        assert state.rag_result is None
        assert state.error_message is None
        assert state.processing_time is None

    def test_state_with_weather_data(self, sample_weather_data):
        """Test state with weather data."""
        state = AssistantState()
        weather_data = WeatherData(**sample_weather_data)
        state.weather_data = weather_data

        assert state.weather_data is not None
        assert state.weather_data.city == "London"
        assert state.weather_data.temperature == 15.5

    def test_state_with_rag_result(self):
        """Test state with RAG result."""
        state = AssistantState()
        rag_result = RAGResult(
            query="Test query",
            relevant_chunks=["Test chunk"],
            summary="Test summary",
            sources=["test.pdf"],
        )
        state.rag_result = rag_result

        assert state.rag_result is not None
        assert state.rag_result.query == "Test query"
        assert len(state.rag_result.relevant_chunks) == 1
        assert state.rag_result.summary == "Test summary"
        assert state.rag_result.sources == ["test.pdf"]


class TestQueryType:
    """Test cases for QueryType enum."""

    def test_query_type_values(self):
        """Test QueryType enum values."""
        assert QueryType.WEATHER.value == "weather"
        assert QueryType.RAG.value == "rag"
        assert QueryType.UNKNOWN.value == "unknown"

    def test_query_type_comparison(self):
        """Test QueryType comparison."""
        assert QueryType.WEATHER == QueryType.WEATHER
        assert QueryType.WEATHER != QueryType.RAG
        assert QueryType.RAG != QueryType.UNKNOWN


class TestMessageRole:
    """Test cases for MessageRole enum."""

    def test_message_role_values(self):
        """Test MessageRole enum values."""
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"
        assert MessageRole.SYSTEM.value == "system"

    def test_message_role_comparison(self):
        """Test MessageRole comparison."""
        assert MessageRole.USER == MessageRole.USER
        assert MessageRole.USER != MessageRole.ASSISTANT
        assert MessageRole.ASSISTANT != MessageRole.SYSTEM
