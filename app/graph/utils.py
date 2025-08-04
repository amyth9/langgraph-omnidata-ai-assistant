import asyncio
from typing import Dict, Any, Optional
from app.graph.state import AssistantState, QueryType, MessageRole


def should_continue_to_weather(state: AssistantState) -> bool:
    return state.query_type == QueryType.WEATHER and not state.error_message


def should_continue_to_rag(state: AssistantState) -> bool:
    return state.query_type == QueryType.RAG and not state.error_message


def should_continue_to_final_response(state: AssistantState) -> bool:
    return (
        state.weather_data is not None or state.rag_result is not None
    ) and not state.error_message


def get_next_node(state: AssistantState) -> str:
    if state.error_message:
        return "final_response"

    if state.query_type == QueryType.WEATHER:
        return "weather_node"
    elif state.query_type == QueryType.RAG:
        return "rag_node"
    else:
        return "final_response"


def format_weather_response(state: AssistantState) -> str:
    if state.error_message:
        return f"❌ Error: {state.error_message}"

    if not state.weather_data:
        return "❌ No weather data available"

    weather = state.weather_data

    response = f"🌤️ Weather in {weather.city}:\n"
    response += f"• Temperature: {weather.temperature}°C\n"
    response += f"• Conditions: {weather.description.capitalize()}\n"

    if state.processing_time:
        response += f"\n\n⏱️ Processing time: {state.processing_time:.2f}s"

    return response


def format_rag_response(state: AssistantState) -> str:
    if state.error_message:
        return f"❌ Error: {state.error_message}"

    if not state.rag_result:
        return "❌ No RAG results available"

    rag = state.rag_result

    response = f"📚 Response based on {len(rag.sources)} source(s):\n\n"
    response += rag.summary

    if rag.sources:
        response += f"\n\n📖 Sources: {', '.join(rag.sources)}"

    if state.processing_time:
        response += f"\n\n⏱️ Processing time: {state.processing_time:.2f}s"

    return response


def get_response_text(state: AssistantState) -> str:
    for message in reversed(state.messages):
        if message.role == MessageRole.ASSISTANT:
            return message.content

    return "No response generated"


def get_conversation_summary(state: AssistantState) -> Dict[str, Any]:
    user_messages = [msg for msg in state.messages if msg.role == MessageRole.USER]
    assistant_messages = [
        msg for msg in state.messages if msg.role == MessageRole.ASSISTANT
    ]

    return {
        "total_messages": len(state.messages),
        "user_messages": len(user_messages),
        "assistant_messages": len(assistant_messages),
        "query_type": state.query_type.value if state.query_type else "unknown",
        "has_weather_data": state.weather_data is not None,
        "has_rag_result": state.rag_result is not None,
        "has_error": state.error_message is not None,
    }
