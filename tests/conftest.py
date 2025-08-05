"""
Pytest configuration and fixtures for AI Pipeline tests.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from app.graph.state import (
    AssistantState,
    QueryType,
    MessageRole,
    WeatherData,
    RAGResult,
)
from app.graph.nodes.router import RouterNode
from app.graph.nodes.weather import WeatherNode
from app.graph.nodes.rag import RAGNode
from app.graph.tools.weather import WeatherTool
from app.graph.tools.retriever import RetrieverTool


@pytest.fixture
def sample_weather_data() -> Dict[str, Any]:
    """Sample weather data for testing."""
    return {
        "city": "London",
        "temperature": 15.5,
        "feels_like": 14.2,
        "description": "Partly cloudy",
        "humidity": 65,
        "wind_speed": 12.3,
        "pressure": 1013,
        "raw_data": {
            "main": {
                "temp": 15.5,
                "feels_like": 14.2,
                "humidity": 65,
                "pressure": 1013,
            },
            "weather": [{"description": "partly cloudy"}],
            "wind": {"speed": 12.3},
        },
    }


@pytest.fixture
def sample_rag_context() -> Dict[str, Any]:
    """Sample RAG context for testing."""
    return {
        "relevant_chunks": [
            "This is a sample document chunk about AI and machine learning.",
            "Another relevant chunk about natural language processing.",
            "Third chunk about deep learning and neural networks.",
        ],
        "sources": ["sample_doc.pdf", "ai_guide.pdf"],
        "total_results": 3,
        "average_score": 0.85,
        "documents": [
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
            {
                "content": "Third chunk about deep learning and neural networks.",
                "source": "sample_doc.pdf",
                "score": 0.85,
            },
        ],
    }


@pytest.fixture
def sample_assistant_state() -> AssistantState:
    """Sample assistant state for testing."""
    state = AssistantState(
        current_query="What's the weather like in London?", query_type=QueryType.UNKNOWN
    )
    state.add_message(MessageRole.USER, "What's the weather like in London?")
    return state


@pytest.fixture
def mock_llm_interface():
    """Mock LLM interface for testing."""
    mock = Mock()
    mock.classify_query = AsyncMock(return_value="weather")
    mock.summarize_weather_data = AsyncMock(
        return_value="The weather in London is partly cloudy with a temperature of 15.5°C."
    )
    mock.summarize_rag_results = AsyncMock(
        return_value="Based on the documents, AI and machine learning are important technologies."
    )
    mock.generate_response = AsyncMock(return_value="london")
    return mock


@pytest.fixture
def mock_weather_tool():
    """Mock weather tool for testing."""
    mock = Mock()
    mock.extract_city_from_query = AsyncMock(return_value="london")
    mock.get_weather_for_query = AsyncMock(
        return_value={
            "city": "London",
            "temperature": 15.5,
            "feels_like": 14.2,
            "description": "Partly cloudy",
            "humidity": 65,
            "wind_speed": 12.3,
            "pressure": 1013,
        }
    )
    return mock


@pytest.fixture
def mock_retriever_tool():
    """Mock retriever tool for testing."""
    mock = Mock()
    mock.get_rag_context = AsyncMock(
        return_value={
            "relevant_chunks": [
                "This is a sample document chunk about AI and machine learning.",
                "Another relevant chunk about natural language processing.",
            ],
            "sources": ["sample_doc.pdf", "ai_guide.pdf"],
            "total_results": 2,
            "average_score": 0.85,
        }
    )
    mock.retrieve_relevant_documents = AsyncMock(
        return_value=[
            {
                "content": "This is a sample document chunk about AI and machine learning.",
                "source": "sample_doc.pdf",
                "score": 0.9,
            }
        ]
    )
    return mock


@pytest.fixture
def mock_embeddings_interface():
    """Mock embeddings interface for testing."""
    mock = Mock()
    mock.generate_single_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4, 0.5])
    return mock


@pytest.fixture
def mock_qdrant_interface():
    """Mock Qdrant interface for testing."""
    mock = Mock()
    mock.search_similar = Mock(
        return_value=[
            {
                "content": "This is a sample document chunk about AI and machine learning.",
                "source": "sample_doc.pdf",
                "score": 0.9,
            }
        ]
    )
    mock.search_by_source = Mock(
        return_value=[
            {
                "content": "Document from specific source.",
                "source": "sample_doc.pdf",
                "score": 0.8,
            }
        ]
    )
    mock.get_collection_info = Mock(
        return_value={
            "points_count": 100,
            "name": "test_collection",
            "vectors_count": 100,
        }
    )
    return mock


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def weather_api_response():
    """Sample weather API response."""
    return {
        "coord": {"lon": -0.13, "lat": 51.51},
        "weather": [
            {
                "id": 300,
                "main": "Drizzle",
                "description": "light intensity drizzle",
                "icon": "09d",
            }
        ],
        "base": "stations",
        "main": {
            "temp": 15.5,
            "feels_like": 14.2,
            "temp_min": 14.0,
            "temp_max": 17.0,
            "pressure": 1013,
            "humidity": 65,
        },
        "visibility": 10000,
        "wind": {"speed": 12.3, "deg": 250},
        "clouds": {"all": 90},
        "dt": 1485789600,
        "sys": {
            "type": 1,
            "id": 5091,
            "message": 0.0103,
            "country": "GB",
            "sunrise": 1485762037,
            "sunset": 1485794878,
        },
        "timezone": 0,
        "id": 2643743,
        "name": "London",
        "cod": 200,
    }


@pytest.fixture
def geocoding_response():
    """Sample geocoding API response."""
    return {
        "features": [
            {
                "properties": {
                    "city": "London",
                    "country": "United Kingdom",
                    "lat": 51.5074,
                    "lon": -0.1278,
                }
            }
        ]
    }
