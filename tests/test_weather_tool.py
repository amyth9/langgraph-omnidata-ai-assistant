"""
Test cases for WeatherTool functionality.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, Tuple

from app.graph.tools.weather import WeatherTool


class TestWeatherTool:
    """Test cases for WeatherTool."""

    def test_weather_tool_initialization(self):
        """Test WeatherTool initialization."""
        with patch("app.graph.tools.weather.config") as mock_config:
            mock_config.weather.api_key = "test_api_key"
            mock_config.weather.base_url = "https://api.openweathermap.org/data/2.5"
            mock_config.weather.units = "metric"
            mock_config.geoapify.geocoding_url = (
                "https://api.geoapify.com/v1/geocode/search"
            )
            mock_config.geoapify.api_key = "test_geoapify_key"

            weather_tool = WeatherTool()

            assert weather_tool is not None
            assert weather_tool.api_key == "test_api_key"
            assert weather_tool.base_url == "https://api.openweathermap.org/data/2.5"
            assert weather_tool.units == "metric"

    def test_weather_tool_initialization_missing_api_key(self):
        """Test WeatherTool initialization with missing API key."""
        with patch("app.graph.tools.weather.config") as mock_config:
            mock_config.weather.api_key = None

            with pytest.raises(ValueError, match="OPENWEATHER_API_KEY is required"):
                WeatherTool()

    @pytest.mark.asyncio
    async def test_extract_city_from_query_success(self):
        """Test successful city extraction from query."""
        with patch("app.graph.tools.weather.config") as mock_config, patch(
            "app.graph.tools.weather.LLMInterface"
        ) as mock_llm_interface:

            mock_config.weather.api_key = "test_api_key"
            mock_config.weather.base_url = "https://api.openweathermap.org/data/2.5"
            mock_config.weather.units = "metric"
            mock_config.geoapify.geocoding_url = (
                "https://api.geoapify.com/v1/geocode/search"
            )
            mock_config.geoapify.api_key = "test_geoapify_key"

            weather_tool = WeatherTool()

            # Mock LLM response
            mock_llm_interface.return_value.generate_response.return_value = "london"

            result = await weather_tool.extract_city_from_query(
                "What's the weather like in London?"
            )

            assert result == "london"

    @pytest.mark.asyncio
    async def test_extract_city_from_query_llm_error_fallback(self):
        """Test city extraction falls back to regex when LLM fails."""
        with patch("app.graph.tools.weather.config") as mock_config, patch(
            "app.graph.tools.weather.LLMInterface"
        ) as mock_llm_interface:

            mock_config.weather.api_key = "test_api_key"
            mock_config.weather.base_url = "https://api.openweathermap.org/data/2.5"
            mock_config.weather.units = "metric"
            mock_config.geoapify.geocoding_url = (
                "https://api.geoapify.com/v1/geocode/search"
            )
            mock_config.geoapify.api_key = "test_geoapify_key"

            weather_tool = WeatherTool()

            # Mock LLM to raise an exception
            mock_llm_interface.return_value.generate_response.side_effect = Exception(
                "LLM error"
            )

            result = await weather_tool.extract_city_from_query(
                "What's the weather in Paris?"
            )

            # Should fall back to regex extraction
            assert result is not None

    def test_extract_location_regex_success(self):
        """Test regex-based location extraction."""
        with patch("app.graph.tools.weather.config") as mock_config:
            mock_config.weather.api_key = "test_api_key"
            mock_config.weather.base_url = "https://api.openweathermap.org/data/2.5"
            mock_config.weather.units = "metric"
            mock_config.geoapify.geocoding_url = (
                "https://api.geoapify.com/v1/geocode/search"
            )
            mock_config.geoapify.api_key = "test_geoapify_key"

            weather_tool = WeatherTool()

            result = weather_tool._extract_location_regex(
                "What's the weather in London?"
            )

            assert result == "london"

    @pytest.mark.asyncio
    async def test_get_latitude_longitude_no_results(self):
        """Test latitude/longitude retrieval when no results found."""
        with patch("app.graph.tools.weather.config") as mock_config, patch(
            "app.graph.tools.weather.requests.get"
        ) as mock_get:

            mock_config.weather.api_key = "test_api_key"
            mock_config.weather.base_url = "https://api.openweathermap.org/data/2.5"
            mock_config.weather.units = "metric"
            mock_config.geoapify.geocoding_url = (
                "https://api.geoapify.com/v1/geocode/search"
            )
            mock_config.geoapify.api_key = "test_geoapify_key"

            weather_tool = WeatherTool()

            # Mock empty geocoding response
            mock_response = Mock()
            mock_response.json.return_value = {"features": []}
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            lat, lon = weather_tool.get_latitude_longitude("NonexistentCity")

            # Should return fallback coordinates
            assert isinstance(lat, float)
            assert isinstance(lon, float)

    @pytest.mark.asyncio
    async def test_get_latitude_longitude_api_error(self):
        """Test latitude/longitude retrieval when API fails."""
        with patch("app.graph.tools.weather.config") as mock_config, patch(
            "app.graph.tools.weather.requests.get"
        ) as mock_get:

            mock_config.weather.api_key = "test_api_key"
            mock_config.weather.base_url = "https://api.openweathermap.org/data/2.5"
            mock_config.weather.units = "metric"
            mock_config.geoapify.geocoding_url = (
                "https://api.geoapify.com/v1/geocode/search"
            )
            mock_config.geoapify.api_key = "test_geoapify_key"

            weather_tool = WeatherTool()

            # Mock API error
            mock_get.side_effect = Exception("API error")

            lat, lon = weather_tool.get_latitude_longitude("London")

            # Should return fallback coordinates
            assert isinstance(lat, float)
            assert isinstance(lon, float)

    @pytest.mark.asyncio
    async def test_get_weather_data_success(self, weather_api_response):
        """Test successful weather data retrieval."""
        with patch("app.graph.tools.weather.config") as mock_config, patch(
            "app.graph.tools.weather.requests.get"
        ) as mock_get:

            mock_config.weather.api_key = "test_api_key"
            mock_config.weather.base_url = "https://api.openweathermap.org/data/2.5"
            mock_config.weather.units = "metric"
            mock_config.geoapify.geocoding_url = (
                "https://api.geoapify.com/v1/geocode/search"
            )
            mock_config.geoapify.api_key = "test_geoapify_key"

            weather_tool = WeatherTool()

            # Mock successful weather API response
            mock_response = Mock()
            mock_response.json.return_value = weather_api_response
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            result = weather_tool.get_weather_data("London", 51.5074, -0.1278)

            assert result["city"] == "London"
            assert result["temperature"] == 15.5
            assert result["feels_like"] == 14.2
            assert result["description"] == "light intensity drizzle"
            assert result["humidity"] == 65
            assert result["wind_speed"] == 12.3
            assert result["pressure"] == 1013

    @pytest.mark.asyncio
    async def test_get_weather_data_api_error(self):
        """Test weather data retrieval when API fails."""
        with patch("app.graph.tools.weather.config") as mock_config, patch(
            "app.graph.tools.weather.requests.get"
        ) as mock_get:

            mock_config.weather.api_key = "test_api_key"
            mock_config.weather.base_url = "https://api.openweathermap.org/data/2.5"
            mock_config.weather.units = "metric"
            mock_config.geoapify.geocoding_url = (
                "https://api.geoapify.com/v1/geocode/search"
            )
            mock_config.geoapify.api_key = "test_geoapify_key"

            weather_tool = WeatherTool()

            # Mock API error
            mock_get.side_effect = Exception("Weather API error")

            result = weather_tool.get_weather_data("London", 51.5074, -0.1278)

            assert "error" in result
            assert "Weather API error" in result["error"]

    @pytest.mark.asyncio
    async def test_get_weather_for_query_with_error_response(self):
        """Test weather retrieval when API returns an error."""
        with patch("app.graph.tools.weather.config") as mock_config, patch(
            "app.graph.tools.weather.requests.get"
        ) as mock_get, patch(
            "app.graph.tools.weather.LLMInterface"
        ) as mock_llm_interface:

            mock_config.weather.api_key = "test_api_key"
            mock_config.weather.base_url = "https://api.openweathermap.org/data/2.5"
            mock_config.weather.units = "metric"
            mock_config.geoapify.geocoding_url = (
                "https://api.geoapify.com/v1/geocode/search"
            )
            mock_config.geoapify.api_key = "test_geoapify_key"

            weather_tool = WeatherTool()

            # Mock LLM response
            mock_llm_interface.return_value.generate_response.return_value = (
                "nonexistentcity"
            )

            # Mock geocoding response with no results
            mock_geocoding_response = Mock()
            mock_geocoding_response.json.return_value = {"features": []}
            mock_geocoding_response.raise_for_status.return_value = None

            # Mock weather API error response
            mock_weather_response = Mock()
            mock_weather_response.json.return_value = {
                "cod": "404",
                "message": "city not found",
            }
            mock_weather_response.raise_for_status.return_value = None

            def mock_get_side_effect(url, *args, **kwargs):
                if "geoapify" in url:
                    return mock_geocoding_response
                else:
                    return mock_weather_response

            mock_get.side_effect = mock_get_side_effect

            result = await weather_tool.get_weather_for_query(
                "What's the weather in NonexistentCity?"
            )

            assert "error" in result
            assert "city not found" in result["error"]

    @pytest.mark.asyncio
    async def test_get_weather_for_query_geocoding_failure(self):
        """Test weather retrieval when geocoding fails."""
        with patch("app.graph.tools.weather.config") as mock_config, patch(
            "app.graph.tools.weather.requests.get"
        ) as mock_get, patch(
            "app.graph.tools.weather.LLMInterface"
        ) as mock_llm_interface:

            mock_config.weather.api_key = "test_api_key"
            mock_config.weather.base_url = "https://api.openweathermap.org/data/2.5"
            mock_config.weather.units = "metric"
            mock_config.geoapify.geocoding_url = (
                "https://api.geoapify.com/v1/geocode/search"
            )
            mock_config.geoapify.api_key = "test_geoapify_key"

            weather_tool = WeatherTool()

            # Mock LLM response
            mock_llm_interface.return_value.generate_response.return_value = "london"

            # Mock geocoding API error
            mock_get.side_effect = Exception("Geocoding API error")

            result = await weather_tool.get_weather_for_query(
                "What's the weather in London?"
            )

            # Should handle the error gracefully
            assert result is not None
            assert "error" in result or "city" in result
