# Document Q&A System with LangChain, LangGraph and Lambda Labs

This system allows you to ingest documents and ask questions about them using LangChain, LangGraph, and Lambda Labs API.

## Enterprise Office Utilization and ESG Simulation

This project includes a comprehensive simulation of corporate office behaviors across multiple U.S. cities for an insurance company. The simulation focuses on occupancy trends, energy consumption, ESG performance, and strategic space optimization.

### Cities Included:
- Philadelphia
- Los Angeles
- Miami
- Chicago
- New York City

### Data Categories Simulated:
- Employee Directory
- Badge Swipe Data
- Meeting Room Utilization
- Lease and Market Data
- Energy Consumption
- ESG Metrics

## Setup

1. Install the required packages:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Make sure your documents are placed in the `documents/` folder.

3. Set up environment variables for API keys and configuration:
   - Create a `.env` file in the project root
   - Add your API keys and configuration settings
   - See the `.env.example` file for required variables

## Usage

You can ask questions about your documents in two ways:

### Using the ask_question.py script

```bash
./ask_question.py "Your question about the documents?"
```

For example:
```bash
./ask_question.py "What was the average meeting room utilization in NYC?"
```

Note: The elp_ai_agent will automatically ingest documents when needed.

### Using the elp_ai_agent.py directly

Edit the question in `elp_ai_agent.py` and run:
```bash
python elp_ai_agent.py
```

### Interactive Chat (v3)

For a more interactive experience, use the v3 interactive chat:
```bash
python interactive_chat_v3.py
```

## System Components

- `ingest_documents.py`: Optional utility to pre-process documents and create a vector database
- `elp_ai_agent.py`: Main system that handles both document ingestion and question answering
- `ask_question.py`: Command-line interface for asking questions
- `config.py`: Configuration management for API keys and settings
- `interactive_chat_v3.py`: Interactive chat interface for asking questions