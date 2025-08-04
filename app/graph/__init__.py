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
