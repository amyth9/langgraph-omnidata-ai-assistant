import os
from typing import List, Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.callbacks import LangChainTracer
from langsmith import Client
from app.config import config


class LLMInterface:
    def __init__(self):
        self.client = ChatGoogleGenerativeAI(
            model=config.google_ai.model,
            temperature=config.google_ai.temperature,
            max_tokens=config.google_ai.max_tokens,
            google_api_key=config.google_ai.api_key,
        )

        if config.langsmith.tracing:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_ENDPOINT"] = config.langsmith.endpoint
            os.environ["LANGCHAIN_API_KEY"] = config.langsmith.api_key
            os.environ["LANGCHAIN_PROJECT"] = config.langsmith.project

            self.tracer = LangChainTracer()
            self.langsmith_client = Client(
                api_url=config.langsmith.endpoint, api_key=config.langsmith.api_key
            )
        else:
            self.tracer = None
            self.langsmith_client = None

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        try:
            langchain_messages = []

            if system_prompt:
                langchain_messages.append(SystemMessage(content=system_prompt))

            for message in messages:
                if message["role"] == "user":
                    langchain_messages.append(HumanMessage(content=message["content"]))
                elif message["role"] == "assistant":
                    langchain_messages.append(AIMessage(content=message["content"]))

            response = await self.client.ainvoke(
                langchain_messages, config={"metadata": metadata or {}}
            )

            return response.content

        except Exception as e:
            raise Exception(f"Error generating LLM response: {str(e)}")

    async def summarize_weather_data(
        self, weather_data: Dict[str, Any], query: str
    ) -> str:
        system_prompt = """You are a helpful weather assistant. First, present the weather information in a clear, structured format with each detail on a new line, then provide a conversational answer to the user's query.

Weather Information Format:
Temperature: XX°C
Feels like: XX°C  
Humidity: XX%
Wind: X.X m/s

Then answer the user's question conversationally. For example:
- "Is it cold?" → "No, it's quite warm and comfortable"
- "How's the weather?" → "It's nice and sunny, perfect for outdoor activities"
- "Should I bring a jacket?" → "You probably won't need one, it's quite warm"

Be helpful, conversational, and provide practical advice based on the weather conditions. Make sure each weather detail is on its own line."""

        messages = [
            {
                "role": "user",
                "content": f"Query: {query}\nWeather Data: {weather_data}\n\nPlease provide the weather information in the structured format with each detail on a new line, then answer the user's question conversationally.",
            }
        ]

        return await self.generate_response(
            messages=messages,
            system_prompt=system_prompt,
            metadata={"task": "weather_summary", "query": query},
        )

    async def summarize_rag_results(
        self, query: str, relevant_chunks: List[str], sources: List[str]
    ) -> str:
        system_prompt = """You are a helpful assistant that answers questions based on provided document information. 
        Use only the information provided in the relevant chunks to answer the question. 
        If the information is not sufficient, say so clearly. Be accurate and helpful."""

        context = "\n\n".join(relevant_chunks)
        messages = [
            {
                "role": "user",
                "content": f"Question: {query}\n\nRelevant Information:\n{context}\n\nPlease answer the question based on the provided information.",
            }
        ]

        return await self.generate_response(
            messages=messages,
            system_prompt=system_prompt,
            metadata={"task": "rag_summary", "query": query, "sources": sources},
        )

    async def classify_query(self, query: str) -> str:
        system_prompt = """You are a query classifier. Determine if the user's question is about:
        1. Weather (current weather, temperature, weather forecast, etc.) - respond with "weather"
        2. Document content (questions about uploaded PDFs, documents, etc.) - respond with "rag"
        
        Only respond with "weather" or "rag"."""

        messages = [{"role": "user", "content": f"Classify this query: {query}"}]

        response = await self.generate_response(
            messages=messages,
            system_prompt=system_prompt,
            metadata={"task": "query_classification", "query": query},
        )

        return response.strip().lower()
