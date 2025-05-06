# ELP Document Intelligence System

## Overview

ELP Document Intelligence is an advanced RAG (Retrieval Augmented Generation) system built with LangChain that enables intelligent analysis of corporate real estate documents. The system provides powerful document processing capabilities with a conversational interface, making it easy to extract insights from complex datasets across multiple office locations.

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

## 📘 Related Documentation

- 📄 [AI Document Intelligence Summary](./ELP_AI_Document_Intelligence_Summary.md) – Full system architecture, prompt design, and capability breakdown.

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
python -m venv .venv && source .venv/bin/activate  # On Windows: .venv\Scripts\activate
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

### Using the API

The Document Intelligence System now provides a REST API built with FastAPI, allowing programmatic access to all system capabilities.

#### Starting the API Server

```bash
python api.py
```

This will start a FastAPI server on `http://localhost:8000` by default.

#### API Documentation

Interactive API documentation is automatically generated and available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

#### Available Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | Process document queries with optional reasoning |
| `/documents/refresh` | POST | Refresh the document database (incremental or full rebuild) |
| `/documents/status` | GET | Get document database stats and status |
| `/facts` | GET | Retrieve all extracted facts |
| `/facts` | POST | Add a new fact to the fact memory |
| `/extract-patterns` | POST | Extract patterns like dates, emails, etc. from text |
| `/summarize` | POST | Generate a summary of a document |
| `/visualize` | POST | Create data visualizations |
| `/compare-documents` | POST | Compare multiple documents |
| `/insights` | POST | Generate insights from documents |
| `/health` | GET | Check API health status |

#### Example API Usage

**Querying Documents**

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Which office had the highest occupancy in April?",
    "reasoning_mode": true,
    "detail_level": "detailed"
  }'
```

**Refreshing the Document Database**

```bash
curl -X POST "http://localhost:8000/documents/refresh" \
  -H "Content-Type: application/json" \
  -d '{
    "documents_path": "./documents",
    "full_rebuild": false
  }'
```

**Extracting Patterns from Text**

```bash
curl -X POST "http://localhost:8000/extract-patterns" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Meeting scheduled for January 15, 2025 at 3:30 PM. Contact sales@example.com for details.",
    "pattern_type": "date"
  }'
```

**Generating Data Visualizations**

```bash
curl -X POST "http://localhost:8000/visualize" \
  -H "Content-Type: application/json" \
  -d '{
    "data": "City,Occupancy\nNYC,78%\nLA,65%\nChicago,71%\nMiami,82%\nPhiladelphia,68%",
    "chart_type": "bar",
    "title": "Office Occupancy by City"
  }'
```

#### API Request/Response Types

**Query Request**
```json
{
  "query": "Which office had the highest average occupancy?",
  "reasoning_mode": true,
  "detail_level": "standard"
}
```

**Document Refresh Request**
```json
{
  "documents_path": "./documents",
  "full_rebuild": false
}
```

**Pattern Extraction Request**
```json
{
  "text": "Contract expires on 12/31/2025. Budget: $1,500,000.",
  "pattern_type": "currency"
}
```

**Visualization Request**
```json
{
  "data": "Month,Energy,Cost\nJan,1200,3600\nFeb,1100,3300\nMar,1300,3900",
  "chart_type": "line",
  "title": "Energy Consumption Trend"
}
```

**Document Comparison Request**
```json
{
  "documents": ["la_employees.csv", "nyc_employees.csv"],
  "aspects": ["headcount", "departments", "roles"]
}
```

#### Integration with Other Systems

The API can be integrated with:

- **BI Dashboards**: Connect Tableau, Power BI or similar tools to the API
- **Custom Applications**: Build web or mobile apps that leverage document intelligence
- **Automation Tools**: Include in workflows with tools like Zapier or n8n
- **Scheduled Jobs**: Set up cron jobs to refresh data and generate reports

#### API Security Considerations

The current API implementation is intended for internal use. For production deployment, consider implementing:

- API key authentication
- HTTPS encryption
- Rate limiting
- Access control based on roles
- Request logging and monitoring

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

| Command | Description | Usage and Expectations |
|---------|-------------|------------------------|
| `help` | Show all available commands and sample questions | **Usage**: Type `help` at any time<br>**Expects**: Displays a comprehensive list of all available commands and sample queries you can ask the system |
| `exit` or `quit` | End the session and save history | **Usage**: Type `exit` or `quit` when finished<br>**Expects**: Conversation history will be saved to `chat_history.json` and the program will terminate |
| `clear` | Clear conversation history | **Usage**: Type `clear` to start fresh<br>**Expects**: Removes all previous conversation turns while maintaining learned facts |

### Document Management

| Command | Description | Usage and Expectations |
|---------|-------------|------------------------|
| `refresh` | Manually refresh the document database | **Usage**: Type `refresh` after adding or modifying documents<br>**Expects**: System will incrementally update the vector database with any document changes (takes 1-2 minutes depending on changes) |
| `delete chroma` | Delete the ChromaDB vector database | **Usage**: Type `delete chroma` when you want to reset the database<br>**Expects**: Removes the entire database folder; requires a `rebuild database` afterward |
| `rebuild database` | Delete and completely rebuild the database | **Usage**: Type `rebuild database` for a fresh start<br>**Expects**: Full reprocessing of all documents (takes 3-5 minutes depending on document count) |

### Information Display

| Command | Description | Usage and Expectations |
|---------|-------------|------------------------|
| `facts` | Show facts the system has learned | **Usage**: Type `facts` to see extracted knowledge<br>**Expects**: Displays a table of all facts the system has extracted, along with their sources and confidence levels |
| `runs` | View links to recent LangSmith runs | **Usage**: Type `runs` to see debugging information<br>**Expects**: Shows URLs to recent LangSmith trace visualizations (requires LANGCHAIN_API_KEY) |
| `insights` | Show generated insights | **Usage**: Type `insights` to see automated analysis<br>**Expects**: Displays a list of insights the system has automatically generated from your documents |

### Feature Toggles

| Command | Description | Usage and Expectations |
|---------|-------------|------------------------|
| `reasoning on` | Enable display of reasoning steps | **Usage**: Type `reasoning on` for transparent thinking<br>**Expects**: Future responses will include detailed step-by-step reasoning, including confidence scores and verification steps |
| `reasoning off` | Disable display of reasoning steps | **Usage**: Type `reasoning off` for concise answers<br>**Expects**: Future responses will show only the final answer without detailed reasoning |
| `insights on` | Enable automatic insight generation | **Usage**: Type `insights on` for proactive analysis<br>**Expects**: System will automatically generate insights when answering relevant queries |
| `insights off` | Disable automatic insight generation | **Usage**: Type `insights off` for faster responses<br>**Expects**: System will not automatically generate insights, potentially increasing response speed |

### Document Intelligence

| Command | Description | Usage and Expectations |
|---------|-------------|------------------------|
| `visualize [data]` | Create visualization from data | **Usage**: Type `visualize` followed by CSV data or a reference to data (e.g., `visualize occupancy rates by city`)<br>**Expects**: Generates and displays a chart based on the provided data; saves the image to the `visualizations` folder |
| `summarize [document]` | Generate summary of document | **Usage**: Type `summarize` followed by document name (e.g., `summarize nyc_lease_market_data.csv`)<br>**Expects**: Produces a concise summary highlighting key points from the document |
| `extract [pattern] from [text]` | Extract patterns (dates, numbers, etc.) | **Usage**: Type `extract` followed by pattern type and text (e.g., `extract dates from The meeting is on May 15, 2025`)<br>**Expects**: Returns all instances of the specified pattern found in the text |
| `compare [docs]` | Compare multiple documents | **Usage**: Type `compare` followed by document names (e.g., `compare nyc_employees.csv la_employees.csv`)<br>**Expects**: Produces a comparison analysis highlighting similarities and differences between the documents |

### Advanced Usage Examples

**Visualizing Data**
```
> visualize City,Occupancy,Energy Usage (kWh)
> NYC,78%,45000
> LA,65%,38000
> Chicago,71%,42000
> Miami,82%,51000
> Philadelphia,68%,36000
```
*Expects*: Creates and displays a bar chart showing occupancy and energy usage by city

**Extracting Multiple Pattern Types**
```
> extract currency from The budget for Q2 is $350,000 with an additional €25,000 for contingency
```
*Expects*: Returns ["$350,000", "€25,000"]

**Comparing Documents with Focus Areas**
```
> compare nyc_employees.csv la_employees.csv with focus on department distribution and salary ranges
```
*Expects*: Produces an analysis comparing the department distribution and salary ranges between NYC and LA offices

**Creating Insights with Specific Parameters**
```
> insights on employee headcount and space utilization across all offices
```
*Expects*: Generates and displays insights specifically about headcount and utilization patterns

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