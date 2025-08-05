"""
Test cases for WeatherNode functionality.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from app.graph.nodes.weather import WeatherNode
from app.graph.state import AssistantState, QueryType, MessageRole, WeatherData


class TestWeatherNode:
    """Test cases for WeatherNode."""

    def test_weather_node_initialization(self):
        """Test WeatherNode initialization."""
        with patch("app.graph.nodes.weather.WeatherTool") as mock_weather_tool, patch(
            "app.graph.nodes.weather.LLMInterface"
        ) as mock_llm_interface:

            weather_node = WeatherNode()

            assert weather_node is not None
            assert hasattr(weather_node, "weather_tool")
            assert hasattr(weather_node, "llm_interface")

    @pytest.mark.asyncio
    async def test_process_weather_query_success(
        self, sample_assistant_state, mock_weather_tool, mock_llm_interface
    ):
        """Test successful weather query processing."""
        with patch(
            "app.graph.nodes.weather.WeatherTool", return_value=mock_weather_tool
        ), patch(
            "app.graph.nodes.weather.LLMInterface", return_value=mock_llm_interface
        ):

            weather_node = WeatherNode()

            # Mock weather tool response
            mock_weather_tool.get_weather_for_query.return_value = {
                "city": "London",
                "temperature": 15.5,
                "feels_like": 14.2,
                "description": "Partly cloudy",
                "humidity": 65,
                "wind_speed": 12.3,
                "pressure": 1013,
            }

            # Mock LLM response
            mock_llm_interface.summarize_weather_data.return_value = (
                "The weather in London is partly cloudy with a temperature of 15.5°C."
            )

            result = await weather_node.process_weather_query(sample_assistant_state)

            assert result.error_message is None
            assert result.weather_data is not None
            assert result.weather_data.city == "London"
            assert result.weather_data.temperature == 15.5
            assert result.weather_data.feels_like == 14.2
            assert result.weather_data.description == "Partly cloudy"
            assert result.weather_data.humidity == 65
            assert result.weather_data.wind_speed == 12.3
            assert result.weather_data.pressure == 1013
            assert result.processing_time is not None

            # Check that assistant message was added
            assistant_messages = [
                msg for msg in result.messages if msg.role == MessageRole.ASSISTANT
            ]
            assert len(assistant_messages) == 1
            assert (
                "The weather in London is partly cloudy"
                in assistant_messages[0].content
            )

    @pytest.mark.asyncio
    async def test_process_weather_query_no_query_provided(
        self, mock_weather_tool, mock_llm_interface
    ):
        """Test weather processing with no query."""
        with patch(
            "app.graph.nodes.weather.WeatherTool", return_value=mock_weather_tool
        ), patch(
            "app.graph.nodes.weather.LLMInterface", return_value=mock_llm_interface
        ):

            weather_node = WeatherNode()

            # Create state with no query
            state = AssistantState()

            result = await weather_node.process_weather_query(state)

            assert result.error_message == "No query provided for weather processing"
            assert result.weather_data is None

    @pytest.mark.asyncio
    async def test_process_weather_query_uses_last_user_message(
        self, mock_weather_tool, mock_llm_interface
    ):
        """Test weather processing uses last user message when no current query."""
        with patch(
            "app.graph.nodes.weather.WeatherTool", return_value=mock_weather_tool
        ), patch(
            "app.graph.nodes.weather.LLMInterface", return_value=mock_llm_interface
        ):

            weather_node = WeatherNode()

            # Create state with messages but no current query
            state = AssistantState()
            state.add_message(MessageRole.USER, "What's the weather in Paris?")

            mock_weather_tool.get_weather_for_query.return_value = {
                "city": "Paris",
                "temperature": 20.0,
                "feels_like": 19.0,
                "description": "Sunny",
                "humidity": 50,
                "wind_speed": 5.0,
                "pressure": 1015,
            }

            mock_llm_interface.summarize_weather_data.return_value = (
                "The weather in Paris is sunny with a temperature of 20°C."
            )

            result = await weather_node.process_weather_query(state)

            # Should use the last user message
            mock_weather_tool.get_weather_for_query.assert_called_with(
                "What's the weather in Paris?"
            )
            assert result.weather_data.city == "Paris"

    @pytest.mark.asyncio
    async def test_process_weather_query_no_city_extracted(
        self, sample_assistant_state, mock_weather_tool, mock_llm_interface
    ):
        """Test weather processing when no city can be extracted."""
        with patch(
            "app.graph.nodes.weather.WeatherTool", return_value=mock_weather_tool
        ), patch(
            "app.graph.nodes.weather.LLMInterface", return_value=mock_llm_interface
        ):

            weather_node = WeatherNode()

            # Mock weather tool to return None (no city extracted)
            mock_weather_tool.get_weather_for_query.return_value = None

            result = await weather_node.process_weather_query(sample_assistant_state)

            assert result.error_message == "Could not extract city name from query"
            assert result.weather_data is None

    @pytest.mark.asyncio
    async def test_process_weather_query_api_error(
        self, sample_assistant_state, mock_weather_tool, mock_llm_interface
    ):
        """Test weather processing when API returns an error."""
        with patch(
            "app.graph.nodes.weather.WeatherTool", return_value=mock_weather_tool
        ), patch(
            "app.graph.nodes.weather.LLMInterface", return_value=mock_llm_interface
        ):

            weather_node = WeatherNode()

            # Mock weather tool to return error
            mock_weather_tool.get_weather_for_query.return_value = {
                "error": "City not found"
            }

            result = await weather_node.process_weather_query(sample_assistant_state)

            assert result.error_message == "Weather API error: City not found"
            assert result.weather_data is None

    @pytest.mark.asyncio
    async def test_process_weather_query_processing_time(
        self, sample_assistant_state, mock_weather_tool, mock_llm_interface
    ):
        """Test that processing time is recorded."""
        with patch(
            "app.graph.nodes.weather.WeatherTool", return_value=mock_weather_tool
        ), patch(
            "app.graph.nodes.weather.LLMInterface", return_value=mock_llm_interface
        ):

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

            result = await weather_node.process_weather_query(sample_assistant_state)

            assert result.processing_time is not None
            assert isinstance(result.processing_time, float)
            assert result.processing_time >= 0

    @pytest.mark.asyncio
    async def test_process_weather_query_preserves_state_data(
        self, sample_assistant_state, mock_weather_tool, mock_llm_interface
    ):
        """Test that weather processing preserves existing state data."""
        with patch(
            "app.graph.nodes.weather.WeatherTool", return_value=mock_weather_tool
        ), patch(
            "app.graph.nodes.weather.LLMInterface", return_value=mock_llm_interface
        ):

            weather_node = WeatherNode()

            # Add some existing data to state
            sample_assistant_state.session_id = "test_session"
            sample_assistant_state.user_id = "test_user"
            sample_assistant_state.add_message(MessageRole.USER, "Previous message")

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

            result = await weather_node.process_weather_query(sample_assistant_state)

            # Check that existing data is preserved
            assert result.session_id == "test_session"
            assert result.user_id == "test_user"
            assert (
                len(result.messages) > 1
            )  # Should have previous message plus new assistant message

    def test_format_weather_response_no_weather_data(self, sample_assistant_state):
        """Test formatting weather response when no weather data exists."""
        with patch("app.graph.nodes.weather.WeatherTool"), patch(
            "app.graph.nodes.weather.LLMInterface"
        ):

            weather_node = WeatherNode()

            formatted_response = weather_node.format_weather_response(
                sample_assistant_state
            )

            assert isinstance(formatted_response, str)
            # Should handle the case gracefully

    @pytest.mark.asyncio
    async def test_process_weather_query_empty_string_query(
        self, mock_weather_tool, mock_llm_interface
    ):
        """Test weather processing with empty string query."""
        with patch(
            "app.graph.nodes.weather.WeatherTool", return_value=mock_weather_tool
        ), patch(
            "app.graph.nodes.weather.LLMInterface", return_value=mock_llm_interface
        ):

            weather_node = WeatherNode()

            state = AssistantState(current_query="")

            result = await weather_node.process_weather_query(state)

            assert result.error_message == "No query provided for weather processing"
            assert result.weather_data is None

    @pytest.mark.asyncio
    async def test_process_weather_query_assistant_message_format(
        self, sample_assistant_state, mock_weather_tool, mock_llm_interface
    ):
        """Test that assistant message is properly formatted."""
        with patch(
            "app.graph.nodes.weather.WeatherTool", return_value=mock_weather_tool
        ), patch(
            "app.graph.nodes.weather.LLMInterface", return_value=mock_llm_interface
        ):

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

            result = await weather_node.process_weather_query(sample_assistant_state)

            assistant_messages = [
                msg for msg in result.messages if msg.role == MessageRole.ASSISTANT
            ]
            assert len(assistant_messages) == 1

            assistant_message = assistant_messages[0]
            assert assistant_message.role == MessageRole.ASSISTANT
            assert "The weather in London is partly cloudy" in assistant_message.content
            assert assistant_message.timestamp is not None
