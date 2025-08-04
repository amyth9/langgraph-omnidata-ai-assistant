from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class QueryType(str, Enum):
    WEATHER = "weather"
    RAG = "rag"
    UNKNOWN = "unknown"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    role: MessageRole
    content: str
    timestamp: Optional[str] = None


class WeatherData(BaseModel):
    city: str
    temperature: float
    description: str
    humidity: int
    wind_speed: float
    pressure: int
    raw_data: Dict[str, Any]


class RAGResult(BaseModel):
    query: str
    relevant_chunks: List[str]
    summary: str
    sources: List[str] = Field(default_factory=list)
