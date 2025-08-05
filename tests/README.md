# Tests

This directory contains the test suite for the AI Pipeline project.

## Structure

```
tests/
├── conftest.py              # Pytest configuration and fixtures
├── test_state.py            # State management tests
├── test_router_node.py      # Router node tests
├── test_weather_node.py     # Weather node tests
├── test_rag_node.py         # RAG node tests
├── test_weather_tool.py     # Weather tool tests
├── test_retriever_tool.py   # Retriever tool tests
├── test_api_handling.py     # API handling tests
└── test_integration.py      # Integration tests
```

## Running Tests

### Quick Start
```bash
# Run all tests
python3 run_tests.py

# Run with pytest directly
pytest tests/
```

### Test Categories
```bash
# Unit tests
python3 run_tests.py --unit

# Integration tests
python3 run_tests.py --integration

# API tests
python3 run_tests.py --api

# All tests with coverage
python3 run_tests.py --coverage
```

### Coverage Report
```bash
# Generate HTML coverage report
python3 run_tests.py --coverage --html
```

## Test Results

- **93 passing tests**
- **0 failing tests**
- **Comprehensive coverage** of all components

## Dependencies

All testing dependencies are included in `requirements.txt`:
- pytest
- pytest-asyncio
- pytest-cov
- pytest-mock 