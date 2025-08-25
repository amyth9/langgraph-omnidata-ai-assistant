# Omnidata AI Assistant

A sophisticated AI assistant built with **LangGraph**, **LangChain**, and **LangSmith** that provides real-time weather information and intelligent document Q&A using RAG (Retrieval-Augmented Generation).

## Features

- **Real-time Weather Data**: Fetch current weather information for any city using OpenWeatherMap API
- **Document Q&A**: Upload PDF documents and ask questions using RAG
- **Intelligent Routing**: Automatic classification of queries (weather vs. document questions)
- **Vector Search**: Advanced document retrieval using Qdrant vector database
- **LangSmith Integration**: Comprehensive LLM response evaluation and tracing
- **Streamlit UI**: Beautiful, user-friendly chat interface
- **Async Processing**: High-performance async/await architecture
- **Comprehensive Testing**: Full test coverage with pytest


## Tech Stack

| Component           | Technology                    |
| ------------------- | ----------------------------- |
| **Orchestration**   | LangGraph                     |
| **LLM**             | Google Generative AI (Gemini) |
| **Vector Database** | Qdrant                        |
| **Embeddings**      | Google AI embedding-001       |
| **Evaluation**      | LangSmith                     |
| **Frontend**        | Streamlit                     |
| **PDF Processing**  | PDFPlumberParser              |
| **Testing**         | pytest                        |

## Prerequisites

- Python 3.10+
- Google AI API key
- OpenWeatherMap API key
- Qdrant Cloud account
- LangSmith account

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd AI-Assistant
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Setup

Create a `.env` file in the root directory with your API keys and configurations. Replace the placeholders with your actual keys:

```bash
# Copy the example file
cp .env.example .env

# Edit the .env file

# QDRANT
QDRANT_ENDPOINT=your_qdrant_endpoint_here
QDRANT_API_KEY=your_qdrant_api_key_here

# GOOGLE GENERATIVE AI
GOOGLE_API_KEY=your_google_api_key_here

# LANGSMITH
LANGSMITH_TRACING="true"
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT="AI-Assistant"

# OPEN WEATHER API
OPENWEATHER_API_KEY=your_openweather_api_key_here 

# GEOAPIFY API
GEOAPIFY_API_KEY=your_geoapify_api_key_here
```

### 4. Run the Application

```bash
cd ui
streamlit run app.py
```

OR

```bash
python run_app.py
```

## Usage

### Weather Queries

Ask about weather for any city:
- "What's the weather in London?"
- "Temperature in New York"
- "Weather forecast for Tokyo"

### Document Queries

Upload PDF documents and ask questions:
- "What are the main topics?"
- "Summarize the key points"
- "What does the document say about...?"

## Project Structure

```
├── app/
│   ├── config.py              # Configuration management
│   ├── graph/                 # LangGraph workflow
│   │   ├── __init__.py       # Main workflow
│   │   ├── state.py          # State management
│   │   ├── utils.py          # Utility functions
│   │   ├── nodes/            # Graph nodes
│   │   └── tools/            # Processing tools
│   ├── interfaces/           # External service interfaces
│   └── utils/               # Utility functions
|       └── ingest.py         # Ingestion script
├── ui/
│   └── app.py               # Streamlit UI
├── tests/                   # Test files
├── requirements.txt         # Dependencies
├── run_app.py              # Application launcher
└── README.md               # Documentation
```

## Testing

Run the test suite:

```bash
# Using the test runner script (recommended)
python3 run_tests.py

# Using pytest directly
pytest tests/
```

## API Keys Required

1. **Google AI**: For LLM and embeddings
2. **OpenWeatherMap**: For weather data
3. **Qdrant**: For vector database
4. **LangSmith**: For evaluation and tracing

## License

This project is licensed under the MIT License.
