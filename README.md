# ELP Document Intelligence System

## Overview

ELP Document Intelligence is an advanced RAG (Retrieval Augmented Generation) system built with LangChain that enables intelligent analysis of corporate real estate documents. The system provides powerful document processing capabilities with a conversational interface, making it easy to extract insights from complex datasets across multiple office locations.


---

## 📘 Related Documentation

- 📄 [AI Document Intelligence Summary](./ELP_AI_Document_Intelligence_Summary.md) – Full system architecture, prompt design, and capability breakdown.

---


## 🌐 Enterprise Office Utilization & ESG Simulation

A realistic simulation of workplace behaviors across major U.S. cities — designed for strategic decision-making, space optimization, and ESG monitoring in an insurance enterprise.

### 🏙️ Cities Simulated:
- 🏛️ Philadelphia
- 🌴 Los Angeles
- 🌞 Miami
- 🌆 Chicago
- 🗽 New York City

### 📂 Data Categories Simulated:
- 👥 Employee Directory
- 🚪 Badge Swipe Logs
- 🧠 Meeting Room Utilization
- 🏢 Lease & Market Data
- ⚡ Energy Consumption
- 🌱 ESG Metrics

## Key Features

- **Document-Aware Chat System**: Natural language interface to query documents across multiple office locations
- **Advanced Reasoning**: Step-by-step reasoning with confidence scoring
- **Self-Questioning**: AI-generated clarifying questions to improve answer accuracy
- **Fact Extraction & Verification**: Automatically extracts and verifies key facts from documents
- **Pattern Detection**: Extract structured data like dates, currencies, and percentages
- **Data Visualization**: Generate charts and graphs from document data
- **Cross-Document Analysis**: Compare information across multiple documents
- **Auto-Summarization**: Generate concise summaries of documents
- **Incremental Document Processing**: Efficiently update the vector database when documents change
- **LangSmith Integration**: Request tracking and feedback for continuous improvement

## Architecture

The system is built with a modular architecture:

1. **Document Ingestion**: Process documents into a vector database (ChromaDB)
2. **AI Agent**: Core LLM with reasoning capabilities
3. **Reasoning Service**: Handle complex multi-step reasoning processes
4. **Interactive Chat**: User-friendly interface with chat history
5. **Document Intelligence**: Data extraction, visualization, and insight generation
6. **Fact Memory**: Store and retrieve extracted facts

## Installation

### Prerequisites

- Python 3.9+ 
- Pipenv or Virtualenv (recommended)
- API key for an LLM provider like Lambda Labs

### Setup

1. Clone the repository
```bash
git clone https://github.com/yourusername/ELP-Project-LangChain.git
cd ELP-Project-LangChain
```

2. Create a virtual environment and install dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:
```
LAMBDA_API_KEY=your_lambda_api_key_here
LANGCHAIN_API_KEY=your_langchain_api_key_here
```

4. Ingest documents into the vector database:
```bash
python ingest_documents.py
```

## Usage

### Starting the Chat Interface

```bash
python ai_agent_interactive_chat.py
```

## 🧠 Sample Questions

### 🔎 Occupancy & Planning
- "Which office had the highest average occupancy in April 2025?"
- "Predict which offices will exceed capacity thresholds next month."

### ⚡ Energy Cost Analysis
- "Which city has the highest average energy cost?"
- "Predict total energy cost in NYC for the next 6 months."

### 🌿 ESG Metrics
- "Which office performs best environmentally?"
- "Where would occupancy policies cut energy usage most?"

### 🗺️ Strategic Decisions
- "Which city should consolidate based on utilization?"
- "What is the impact of relocating 25% of NYC staff to Miami?"

### 🛂 Badge & Employee Data
- "How many employees are assigned to LA?"
- "Compare badge utilization between Chicago and NYC."

### 🏢 Facilities & Leasing
- "What is the lease cost per square foot in Philadelphia?"
- "What is our total leased square footage in LA?"

## ✨ Ready-to-Use AI Prompts

### 📈 Predict Headcount Needs
> "Based on current occupancy rates... estimate where we need more seats."

### 🏢 Recommend Office Consolidation
> "Identify underutilized offices below 50% and suggest closures."

### 🔮 Forecast Occupancy
> "Predict May 2025 occupancy from April swipe patterns."

### 💸 Recommend Cost Savings
> "Suggest 3 strategies based on lease, occupancy, and energy cost data."

## Available Commands

When using the interactive chat mode, the following commands are available:

### Basic Commands

| Command | Description |
|---------|-------------|
| `help` | Show all available commands and sample questions |
| `exit` or `quit` | End the session and save history |
| `clear` | Clear conversation history |

### Document Management

| Command | Description |
|---------|-------------|
| `refresh` | Manually refresh the document database |
| `delete chroma` | Delete the ChromaDB vector database |
| `rebuild database` | Delete and completely rebuild the database |

### Information Display

| Command | Description |
|---------|-------------|
| `facts` | Show facts the system has learned |
| `runs` | View links to recent LangSmith runs |
| `insights` | Show generated insights |

### Feature Toggles

| Command | Description |
|---------|-------------|
| `reasoning on` | Enable display of reasoning steps |
| `reasoning off` | Disable display of reasoning steps |
| `insights on` | Enable automatic insight generation |
| `insights off` | Disable automatic insight generation |

### Document Intelligence

| Command | Description |
|---------|-------------|
| `visualize [data]` | Create visualization from data |
| `summarize [document]` | Generate summary of document |
| `extract [pattern] from [text]` | Extract patterns (dates, numbers, etc.) |
| `compare [docs]` | Compare multiple documents |

### Pattern Types 

When using the `extract` command, the following pattern types are available:
- `currency`: Find monetary values (e.g., $100, 50 dollars)
- `percentage`: Find percentage values (e.g., 50%)
- `date`: Find dates in various formats
- `email`: Find email addresses
- `phone`: Find phone numbers
- `time`: Find time values
- `numeric`: Find any numeric values

## Advanced Features

### Document Processing

The system handles various document types including:
- CSV files
- PDFs
- Text files
- HTML documents
- Markdown files
- Excel spreadsheets
- Word documents

### Incremental Updates

The system uses a hash-based approach to detect document changes and update the vector database incrementally, avoiding full reprocesses when only some files have changed.

### Cross-Document Analysis

Special processing is applied to documents like employee data to create cross-document insights that make it easy to compare data across offices.

## 🧠 Prompt Design Philosophy

Prompts are structured instructions that control what the AI does. In our system, they ensure:

- Answers are **grounded in ingested documents**
- Facts are **extracted with sources + confidence**
- Claims are **verified** before being presented
- Responses show **step-by-step reasoning**
- AI finds **insights and summaries** automatically

## Customization

The system can be customized by modifying:

- `config.py` - System configuration options
- `ai_agent.py` - Core LLM integration 
- `reasoning.py` - Reasoning service
- Document loaders in `ingest_documents.py` for new document types

## Future Improvements

See `z_improvements.md` for a list of planned enhancements.

## License

This project is proprietary and confidential.

## Contact

For questions or support, please contact [your.email@example.com](mailto:your.email@example.com).

---

🔥 Built to help Facilities and Management teams make better decisions — faster, smarter, and with confidence.