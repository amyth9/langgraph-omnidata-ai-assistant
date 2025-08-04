import time
from typing import Dict, Any
from app.graph.state import AssistantState, RAGResult, MessageRole
from app.graph.tools.retriever_tool import RetrieverTool
from app.interfaces.llm_interface import LLMInterface
from app.graph.utils import format_rag_response


class RAGNode:
    def __init__(self):
        self.retriever_tool = RetrieverTool()
        self.llm_interface = LLMInterface()

    async def process_rag_query(self, state: AssistantState) -> AssistantState:
        try:
            start_time = time.time()

            query = state.current_query
            if not query:
                query = state.get_last_user_message()

            if not query:
                state.error_message = "No query provided for RAG processing"
                return state

            rag_context = await self.retriever_tool.get_rag_context(query)

            if not rag_context["relevant_chunks"]:
                state.error_message = "No relevant documents found for your query"
                return state

            rag_result = RAGResult(
                query=query,
                relevant_chunks=rag_context["relevant_chunks"],
                summary="",
                sources=rag_context["sources"],
            )

            state.rag_result = rag_result

            llm_response = await self.llm_interface.summarize_rag_results(
                query=query,
                relevant_chunks=rag_context["relevant_chunks"],
                sources=rag_context["sources"],
            )

            state.rag_result.summary = llm_response

            state.add_message(MessageRole.ASSISTANT, llm_response)

            state.processing_time = time.time() - start_time

            return state

        except Exception as e:
            state.error_message = f"Error in RAG node: {str(e)}"
            return state
