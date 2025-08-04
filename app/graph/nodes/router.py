import time
from typing import Dict, Any
from app.graph.state import AssistantState, QueryType, MessageRole
from app.interfaces.llm import LLMInterface
from app.graph.utils import get_next_node


class RouterNode:
    def __init__(self):
        self.llm_interface = LLMInterface()

    async def route_query(self, state: AssistantState) -> AssistantState:
        try:
            start_time = time.time()

            query = state.current_query
            if not query:
                query = state.get_last_user_message()

            if not query:
                state.error_message = "No query provided"
                return state

            classification = await self.llm_interface.classify_query(query)

            if "weather" in classification:
                state.query_type = QueryType.WEATHER
            elif "rag" in classification:
                state.query_type = QueryType.RAG
            else:
                state.query_type = QueryType.RAG

            classification_msg = f"Query classified as: {state.query_type.value}"
            state.add_message(MessageRole.SYSTEM, classification_msg)

            state.processing_time = time.time() - start_time

            return state

        except Exception as e:
            state.error_message = f"Error in router node: {str(e)}"
            state.query_type = QueryType.UNKNOWN
            return state

    def get_next_node(self, state: AssistantState) -> str:
        return get_next_node(state)
