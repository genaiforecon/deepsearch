# Minimal Deep Research Agent

This is a minimal implementation of the Deep Research Agent that demonstrates the core functionality without requiring the full OpenAI Agents Python SDK.

## Structure

- `agents/` - Core agent implementation
  - `__init__.py` - Exports the main components
  - `agent.py` - Agent class with minimal implementation
  - `model_settings.py` - Model settings class
  - `run.py` - Runner class for executing agents
  - `tool.py` - Tool implementations

- `main.py` - Agent definitions and research manager implementation
- `run_agent.py` - CLI entry point

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python run_agent.py
```

You will be prompted to enter a research query. The agent will:

1. Plan searches related to your query
2. Execute those searches (simulated)
3. Generate a research report based on the results
4. Verify the report quality and coverage

## Requirements

- Python 3.8+
- OpenAI API key (you will be prompted to enter it if not set as an environment variable)

## Dependencies

- openai>=1.0.0
- pydantic>=2.0.0
- rich>=13.0.0
- typing-extensions>=4.0.0