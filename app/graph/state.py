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


class AssistantState(BaseModel):
    messages: List[Message] = Field(default_factory=list)
    current_query: str = ""
    query_type: QueryType = QueryType.UNKNOWN
    weather_data: Optional[WeatherData] = None
    rag_result: Optional[RAGResult] = None
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None

    def add_message(self, role: MessageRole, content: str) -> None:
        from datetime import datetime

        message = Message(
            role=role, content=content, timestamp=datetime.now().isoformat()
        )
        self.messages.append(message)

    def get_last_user_message(self) -> Optional[str]:
        for message in reversed(self.messages):
            if message.role == MessageRole.USER:
                return message.content
        return None

    def get_conversation_history(self) -> str:
        history = []
        for message in self.messages[-10:]:
            history.append(f"{message.role.value}: {message.content}")
        return "\n".join(history)

    def clear_processing_data(self) -> None:
        self.weather_data = None
        self.rag_result = None
        self.error_message = None
        self.processing_time = None
