import requests
import re
from typing import Dict, Any, Optional, Tuple
from requests.structures import CaseInsensitiveDict
from app.config import config
from app.interfaces.llm import LLMInterface


class WeatherTool:
    def __init__(self):
        self.api_key = config.weather.api_key
        self.base_url = config.weather.base_url
        self.geocoding_url = config.geoapify.geocoding_url
        self.geoapify_api_key = config.geoapify.api_key
        self.units = config.weather.units
        self.llm_interface = LLMInterface()

        if not self.api_key:
            raise ValueError("OPENWEATHER_API_KEY is required")

    async def extract_city_from_query(self, query: str) -> Optional[str]:
        try:
            system_prompt = """You are a location extractor. Extract ANY location information from weather queries, including landmarks, beaches, parks, neighborhoods, or any place name.

Return the complete location including any descriptive information. If no clear location is mentioned, return 'none'.

Examples:
- "what are conditions of manali now?" -> "manali"
- "weather in London" -> "london"
- "temperature in New York" -> "new york"
- "how's the weather in Tokyo?" -> "tokyo"
- "what's the temperature in Paris?" -> "paris"
- "is it raining in Mumbai?" -> "mumbai"
- "weather in downtown chicago, illinois" -> "downtown chicago, illinois"
- "temperature in bandra west, mumbai, maharashtra" -> "bandra west, mumbai, maharashtra"
- "how's weather in times square, new york city" -> "times square, new york city"
- "is it safe to travel to marina beach?" -> "marina beach"
- "weather at central park" -> "central park"
- "temperature in downtown area" -> "downtown"
- "how's the weather at the beach?" -> "beach"
- "conditions at the mall" -> "mall"
- "weather near the airport" -> "airport"
- "temperature at the stadium" -> "stadium"
- "weather in the park" -> "park"
- "conditions at the shopping center" -> "shopping center"

Extract ANY location reference, not just city names."""

            messages = [
                {
                    "role": "user",
                    "content": f"Extract the complete location from this weather query: {query}",
                }
            ]

            response = await self.llm_interface.generate_response(
                messages=messages, system_prompt=system_prompt
            )

            location = response.strip().lower()
            if location in ["none", "no location", "unknown", ""]:
                return None

            return location

        except Exception as e:
            return self._extract_location_regex(query)

    def _extract_location_regex(self, query: str) -> Optional[str]:
        weather_keywords = [
            "weather",
            "temperature",
            "forecast",
            "climate",
            "hot",
            "cold",
            "rain",
            "snow",
            "sunny",
            "cloudy",
            "windy",
            "humidity",
            "conditions",
            "safe",
            "travel",
        ]

        query_lower = query.lower()

        for keyword in weather_keywords:
            query_lower = query_lower.replace(keyword, "")

        query_lower = re.sub(r"\s+", " ", query_lower).strip()

        if not query_lower or query_lower in ["?", "!", ".", ",", ";", ":", '"', "'"]:
            return None

        # Look for location patterns
        location_patterns = [
            r"to\s+([^?]+)",  # "travel to marina beach"
            r"at\s+([^?]+)",  # "weather at central park"
            r"in\s+([^?]+)",  # "temperature in downtown"
            r"near\s+([^?]+)",  # "weather near airport"
            r"around\s+([^?]+)",  # "conditions around mall"
            r"([^?]+)\s+weather",  # "marina beach weather"
            r"([^?]+)\s+temperature",  # "central park temperature"
            r"([^?]+)\s+conditions",  # "beach conditions"
        ]

        for pattern in location_patterns:
            match = re.search(pattern, query_lower)
            if match:
                location = match.group(1).strip()
                if location and len(location) > 2:
                    return location

        # If no pattern matches, try to extract any capitalized words
        words = query.split()
        potential_location_parts = []

        for i, word in enumerate(words):
            # Skip common words and weather terms
            if (
                word.lower() not in weather_keywords
                and len(word) > 2
                and word[0].isupper()
                and not word.endswith("?")
            ):

                # Collect this word and any following words that might be part of the location
                location_part = word
                j = i + 1
                while (
                    j < len(words)
                    and words[j].lower() not in weather_keywords
                    and not words[j].endswith("?")
                ):
                    location_part += " " + words[j]
                    j += 1

                if location_part and len(location_part) > 2:
                    potential_location_parts.append(location_part)

        if potential_location_parts:
            # Return the longest potential location
            return max(potential_location_parts, key=len)

        return query_lower

    def get_latitude_longitude(self, city: str) -> Tuple[float, float]:
        try:
            if not self.geoapify_api_key:
                return self._get_coordinates_fallback(city)

            params = {
                "text": city,
                "apiKey": self.geoapify_api_key,
                "limit": 1,
            }

            response = requests.get(self.geocoding_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if not data.get("features"):
                return self._get_coordinates_fallback(city)

            feature = data["features"][0]
            coordinates = feature["geometry"]["coordinates"]

            return coordinates[1], coordinates[0]

        except Exception as e:
            return self._get_coordinates_fallback(city)

    def _get_coordinates_fallback(self, city: str) -> Tuple[float, float]:
        fallback_coordinates = {
            "mumbai": (19.0760, 72.8777),
            "delhi": (28.7041, 77.1025),
            "bangalore": (12.9716, 77.5946),
            "hyderabad": (17.3850, 78.4867),
            "chennai": (13.0827, 80.2707),
            "kolkata": (22.5726, 88.3639),
            "pune": (18.5204, 73.8567),
            "ahmedabad": (23.0225, 72.5714),
            "jaipur": (26.9124, 75.7873),
            "london": (51.5074, -0.0799),
            "new york": (40.7128, -74.0060),
            "tokyo": (35.6812, 139.7671),
            "paris": (48.8566, 2.3522),
            "beijing": (39.9087, 116.3975),
            "sydney": (-33.8688, 151.2153),
        }

        city_lower = city.lower().strip()
        return fallback_coordinates.get(
            city_lower, (19.0760, 72.8777)
        )  # Default to Mumbai

    def get_weather_data(
        self, city: str, latitude: float, longitude: float
    ) -> Dict[str, Any]:
        try:
            params = {
                "lat": latitude,
                "lon": longitude,
                "appid": self.api_key,
                "units": self.units,
            }

            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if data.get("cod") != 200:
                return {
                    "error": f"Weather API error: {data.get('message', 'Unknown error')}"
                }

            main_data = data.get("main", {})
            weather_data = data.get("weather", [{}])[0]

            return {
                "city": city,
                "temperature": main_data.get("temp", 0),
                "feels_like": main_data.get("feels_like", 0),
                "humidity": main_data.get("humidity", 0),
                "pressure": main_data.get("pressure", 0),
                "description": weather_data.get("description", ""),
                "main": weather_data.get("main", ""),
                "icon": weather_data.get("icon", ""),
                "wind_speed": data.get("wind", {}).get("speed", 0),
                "wind_deg": data.get("wind", {}).get("deg", 0),
                "visibility": data.get("visibility", 0),
                "clouds": data.get("clouds", {}).get("all", 0),
                "sunrise": data.get("sys", {}).get("sunrise", 0),
                "sunset": data.get("sys", {}).get("sunset", 0),
                "country": data.get("sys", {}).get("country", ""),
                "timezone": data.get("timezone", 0),
                "raw_data": data,
            }

        except requests.exceptions.RequestException as e:
            return {"error": f"Network error: {str(e)}"}
        except Exception as e:
            return {"error": f"Weather API error: {str(e)}"}

    async def get_weather_for_query(self, query: str) -> Optional[Dict[str, Any]]:
        try:
            city = await self.extract_city_from_query(query)

            if not city:
                return None

            latitude, longitude = self.get_latitude_longitude(city)
            weather_data = self.get_weather_data(city, latitude, longitude)

            return weather_data

        except Exception as e:
            return {"error": f"Error processing weather query: {str(e)}"}

    def format_weather_response(self, weather_data: Dict[str, Any]) -> str:
        if "error" in weather_data:
            return f"❌ Error: {weather_data['error']}"

        response = f"🌤️ Weather Information:\n"
        response += f"• Temperature: {weather_data.get('temperature', 0)}°C\n"
        response += f"• Feels like: {weather_data.get('feels_like', 0)}°C\n"
        response += f"• Humidity: {weather_data.get('humidity', 0)}%\n"
        response += f"• Wind Speed: {weather_data.get('wind_speed', 0)} m/s\n"
        response += f"• Conditions: {weather_data.get('description', '').capitalize()}"

        return response
