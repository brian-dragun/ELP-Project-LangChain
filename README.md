# Document Q&A System with LangChain, LangGraph and Lambda Labs

This system allows you to ingest documents and ask questions about them using LangChain, LangGraph, and Lambda Labs API.

## Setup

1. Install the required packages:
   python3 -m venv .venv && source .venv/bin/activate
   ```
   pip install -r requirements.txt
   ```

2. Make sure your documents are placed in the `documents/` folder.

3. Run the document ingestion process:
   ```
   python ingest_documents.py
   ```

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

### Using the qa_system.py directly

Edit the question in `qa_system.py` and run:
```bash
python qa_system.py
```

## System Components

- `ingest_documents.py`: Processes documents and creates a vector database
- `qa_system.py`: Main system that handles retrieval and question answering
- `ask_question.py`: Command-line interface for asking questions

## Sample Questions

- "How many employees work in the Philadelphia office?"
- "What is the lease term for the Los Angeles office?"
- "What are the ESG metrics for the Miami office?"
- "Compare meeting room utilization rates across all cities."
- "Which office has the highest energy consumption?"