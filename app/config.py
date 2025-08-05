import os
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class WeatherConfig(BaseModel):
    api_key: str = Field(default_factory=lambda: os.getenv("OPENWEATHER_API_KEY", ""))
    base_url: str = "https://api.openweathermap.org/data/2.5/weather"
    units: str = "metric"


class GeoapifyConfig(BaseModel):
    geocoding_url: str = "https://api.geoapify.com/v1/geocode/search"
    api_key: str = Field(default_factory=lambda: os.getenv("GEOAPIFY_API_KEY", ""))


class QdrantConfig(BaseModel):
    endpoint: str = Field(default_factory=lambda: os.getenv("QDRANT_ENDPOINT", ""))
    api_key: str = Field(default_factory=lambda: os.getenv("QDRANT_API_KEY", ""))
    collection_name: str = "ai_assistant_docs"


class GoogleAIConfig(BaseModel):
    api_key: str = Field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    model: str = "gemini-2.0-flash"
    temperature: float = 0.5
    max_tokens: int = 1000


class LangSmithConfig(BaseModel):
    api_key: str = Field(default_factory=lambda: os.getenv("LANGSMITH_API_KEY", ""))
    project: str = Field(
        default_factory=lambda: os.getenv("LANGSMITH_PROJECT", "AI-Assistant")
    )
    endpoint: str = Field(
        default_factory=lambda: os.getenv(
            "LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"
        )
    )
    tracing: bool = Field(
        default_factory=lambda: os.getenv("LANGSMITH_TRACING", "true").lower() == "true"
    )


class AppConfig(BaseModel):
    weather: WeatherConfig = Field(default_factory=WeatherConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    google_ai: GoogleAIConfig = Field(default_factory=GoogleAIConfig)
    langsmith: LangSmithConfig = Field(default_factory=LangSmithConfig)
    geoapify: GeoapifyConfig = Field(default_factory=GeoapifyConfig)
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_retries: int = 3

    def validate_config(self) -> None:
        if not self.google_ai.api_key:
            raise ValueError("GOOGLE_API_KEY is required")
        if not self.qdrant.endpoint or not self.qdrant.api_key:
            raise ValueError("QDRANT_ENDPOINT and QDRANT_API_KEY are required")
        if not self.weather.api_key:
            raise ValueError("OPENWEATHER_API_KEY is required")


config = AppConfig()
