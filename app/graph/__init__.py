import asyncio
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
from app.graph.state import AssistantState, MessageRole
from app.graph.nodes.router import RouterNode
from app.graph.nodes.weather import WeatherNode
from app.graph.nodes.rag import RAGNode
from app.interfaces.llm import LLMInterface
from app.graph.state import QueryType


class AssistantGraph:
    def __init__(self):
        self.router_node = RouterNode()
        self.weather_node = WeatherNode()
        self.rag_node = RAGNode()
        self.llm_interface = LLMInterface()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AssistantState)

        workflow.add_node("router", self.router_node.route_query)
        workflow.add_node("weather_node", self.weather_node.process_weather_query)
        workflow.add_node("rag_node", self.rag_node.process_rag_query)
        workflow.add_node("final_response", self._generate_final_response)

        workflow.set_entry_point("router")

        workflow.add_conditional_edges(
            "router",
            self._route_to_next_node,
            {
                "weather_node": "weather_node",
                "rag_node": "rag_node",
                "final_response": "final_response",
            },
        )

        workflow.add_edge("weather_node", "final_response")
        workflow.add_edge("rag_node", "final_response")
        workflow.add_edge("final_response", END)

        return workflow.compile()

    def _route_to_next_node(self, state: AssistantState) -> str:
        if state.error_message:
            return "final_response"

        if state.query_type == QueryType.WEATHER:
            return "weather_node"
        elif state.query_type == QueryType.RAG:
            return "rag_node"
        else:
            return "final_response"

    async def _generate_final_response(self, state: AssistantState) -> AssistantState:
        try:
            if state.error_message:
                error_response = f"❌ Error: {state.error_message}"
                state.add_message(MessageRole.ASSISTANT, error_response)
                return state

            last_assistant_message = None
            for message in reversed(state.messages):
                if message.role == MessageRole.ASSISTANT:
                    last_assistant_message = message.content
                    break

            if last_assistant_message:
                return state
            else:
                generic_response = "I'm sorry, I couldn't process your request. Please try asking about weather or upload a document to ask questions about it."
                state.add_message(MessageRole.ASSISTANT, generic_response)
                return state

        except Exception as e:
            error_response = f"❌ Error generating final response: {str(e)}"
            state.add_message(MessageRole.ASSISTANT, error_response)
            return state

    async def process_query(
        self,
        query: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> AssistantState:
        try:
            initial_state = AssistantState(
                current_query=query, session_id=session_id, user_id=user_id
            )

            initial_state.add_message(MessageRole.USER, query)

            try:
                result = await self.graph.ainvoke(initial_state)

                if isinstance(result, dict):
                    final_state = AssistantState(**result)
                else:
                    final_state = result

                return final_state
            except Exception as graph_error:
                error_state = AssistantState(
                    current_query=query,
                    session_id=session_id,
                    user_id=user_id,
                    error_message=f"Graph execution error: {str(graph_error)}",
                )
                error_state.add_message(MessageRole.USER, query)
                error_state.add_message(
                    MessageRole.ASSISTANT, f"❌ Error: {str(graph_error)}"
                )
                return error_state

        except Exception as e:
            error_state = AssistantState(
                current_query=query,
                session_id=session_id,
                user_id=user_id,
                error_message=f"Graph execution error: {str(e)}",
            )
            error_state.add_message(MessageRole.USER, query)
            error_state.add_message(MessageRole.ASSISTANT, f"❌ Error: {str(e)}")

            return error_state

    def get_response_text(self, state: AssistantState) -> str:
        for message in reversed(state.messages):
            if message.role == MessageRole.ASSISTANT:
                return message.content

        return "No response generated"

    def get_processing_metadata(self, state: AssistantState) -> Dict[str, Any]:
        metadata = {
            "query_type": state.query_type.value if state.query_type else "unknown",
            "processing_time": state.processing_time or 0.0,
            "error_message": state.error_message,
            "total_messages": len(state.messages),
        }

        if state.weather_data:
            metadata["weather_data"] = {
                "city": state.weather_data.city,
                "temperature": state.weather_data.temperature,
                "description": state.weather_data.description,
            }

        if state.rag_result:
            metadata["rag_data"] = {
                "sources": state.rag_result.sources,
                "chunks_retrieved": len(state.rag_result.relevant_chunks),
            }

        return metadata
