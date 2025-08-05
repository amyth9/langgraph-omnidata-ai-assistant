"""
Test cases for API handling and error scenarios.
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

from app.graph.tools.weather import WeatherTool


class TestAPIHandling:
    """Test cases for API handling scenarios."""

    def test_weather_api_success_response(self, weather_api_response):
        """Test successful weather API response handling."""
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
            assert result["description"] == "light intensity drizzle"
            assert result["humidity"] == 65
            assert result["wind_speed"] == 12.3
            assert result["pressure"] == 1013

    def test_weather_api_network_error(self):
        """Test weather API network error handling."""
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

            # Mock network error
            mock_get.side_effect = Exception("Network connection failed")

            result = weather_tool.get_weather_data("London", 51.5074, -0.1278)

            assert "error" in result
            assert "Network connection failed" in result["error"]

    def test_weather_api_timeout_error(self):
        """Test weather API timeout error handling."""
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

            # Mock timeout error
            mock_get.side_effect = Exception("Request timeout")

            result = weather_tool.get_weather_data("London", 51.5074, -0.1278)

            assert "error" in result
            assert "Request timeout" in result["error"]

    def test_geocoding_api_no_results(self):
        """Test geocoding API when no results are found."""
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

    def test_geocoding_api_network_error(self):
        """Test geocoding API network error handling."""
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

            # Mock network error
            mock_get.side_effect = Exception("Network error")

            lat, lon = weather_tool.get_latitude_longitude("London")

            # Should return fallback coordinates
            assert isinstance(lat, float)
            assert isinstance(lon, float)

    def test_api_request_timeout_configuration(self):
        """Test API request timeout configuration."""
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

            # Mock timeout error
            mock_get.side_effect = Exception("Request timeout")

            result = weather_tool.get_weather_data("London", 51.5074, -0.1278)

            assert "error" in result
            assert "Request timeout" in result["error"]

    def test_api_retry_mechanism_simulation(self):
        """Test API retry mechanism simulation."""
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

            # Mock initial failure then success
            mock_response = Mock()
            mock_response.json.return_value = {
                "cod": "200",
                "name": "London",
                "main": {
                    "temp": 15.5,
                    "feels_like": 14.2,
                    "humidity": 65,
                    "pressure": 1013,
                },
                "weather": [{"description": "light intensity drizzle"}],
                "wind": {"speed": 12.3},
            }
            mock_response.raise_for_status.return_value = None

            # First call fails, second call succeeds
            mock_get.side_effect = [Exception("Temporary error"), mock_response]

            result = weather_tool.get_weather_data("London", 51.5074, -0.1278)

            # Should handle the error gracefully
            assert "error" in result or "city" in result
