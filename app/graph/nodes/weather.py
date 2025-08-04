import time
from typing import Dict, Any
from app.graph.state import AssistantState, WeatherData, MessageRole
from app.graph.tools.weather import WeatherTool
from app.interfaces.llm import LLMInterface
from app.graph.utils import format_weather_response


class WeatherNode:
    def __init__(self):
        self.weather_tool = WeatherTool()
        self.llm_interface = LLMInterface()

    async def process_weather_query(self, state: AssistantState) -> AssistantState:
        try:
            start_time = time.time()

            query = state.current_query
            if not query:
                query = state.get_last_user_message()

            if not query:
                state.error_message = "No query provided for weather processing"
                return state

            weather_data = await self.weather_tool.get_weather_for_query(query)

            if weather_data is None:
                state.error_message = "Could not extract city name from query"
                return state

            if "error" in weather_data:
                state.error_message = f"Weather API error: {weather_data['error']}"
                return state

            weather_info = WeatherData(
                city=weather_data.get("city", "Unknown"),
                temperature=weather_data.get("temperature", 0),
                description=weather_data.get("description", ""),
                humidity=weather_data.get("humidity", 0),
                wind_speed=weather_data.get("wind_speed", 0),
                pressure=weather_data.get("pressure", 0),
                raw_data=weather_data,
            )

            state.weather_data = weather_info

            llm_response = await self.llm_interface.summarize_weather_data(
                weather_data=weather_data, query=query
            )

            state.add_message(MessageRole.ASSISTANT, llm_response)

            state.processing_time = time.time() - start_time

            return state

        except Exception as e:
            state.error_message = f"Error in weather node: {str(e)}"
            return state
