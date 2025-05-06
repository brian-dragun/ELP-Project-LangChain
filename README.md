# ELP Document Intelligence System

## Overview

ELP Document Intelligence is an advanced RAG (Retrieval Augmented Generation) system built with LangChain that enables intelligent analysis of corporate real estate and lease documents as well as security badging data. The system provides powerful document processing capabilities with a conversational interface, making it easy to extract insights from complex datasets across multiple office locations.

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

## Web Interface

The Document Intelligence System includes a web-based interface for easier team collaboration and testing. This temporary solution provides access to all system capabilities through an intuitive browser interface until migration to Azure AI Foundry.

### Starting the Web Interface

```bash
# Install required dependencies
pip install streamlit pandas matplotlib seaborn

# Start the web interface
streamlit run web_interface.py
```

By default, the interface runs on port 8080 and can be accessed at:
- http://localhost:8080 (on the local machine)
- http://[server-ip-address]:8080 (from other machines on the network)

### Navigation Options

The web interface includes several navigation sections, each providing access to different system capabilities:

#### 1. Chat Interface

The main interactive component that allows users to query the document intelligence system through a chat-like interface.

**Features:**
- Natural language queries about your documents
- Chat history preservation during your session
- Toggle for displaying detailed reasoning steps
- Source attribution for answers

**Example Interactions:**
- "Which office had the highest average occupancy in April 2025?"
- "Compare energy usage between NYC and Chicago offices"
- "What are the key ESG metrics across all locations?"

**Expected Output:**
- Direct answers to your queries
- Optional reasoning steps showing how the system arrived at the answer
- Document sources used to generate the response

#### 2. Document Management

Administrative interface for managing the document database and vector store.

**Features:**
- Document database statistics and visualization 
- Options to refresh or rebuild the document database
- Visual breakdown of document types in the system

**Expected Output:**
- Document count and type distribution charts
- Success/failure messages for database operations
- Estimated processing time for operations

#### 3. Facts & Insights

Display and management of extracted facts and generated insights from your documents.

**Features:**
- Table view of all facts extracted from documents
- Confidence scores for each extracted fact
- Source attribution for facts
- Manual fact extraction option

**Expected Output:**
- Structured table of facts with metadata
- Success messages when extracting new facts
- Filterable/sortable fact database

#### 4. System Status

System health monitoring and configuration information.

**Features:**
- System configuration overview
- Memory usage statistics
- API usage tracking
- Health checks for system components

**Expected Output:**
- Vector database size metrics
- System health indicators (success/failure)
- Model information and configuration details
- API usage statistics

### Web Interface Access Control

The web interface is designed for internal team use and does not implement authentication by default. When deploying in a shared environment, consider:

1. Limiting access through network controls
2. Using a reverse proxy with authentication
3. Implementing a VPN requirement for access

### Customizing the Web Interface

The web interface can be customized by modifying the `web_interface.py` file. Common customizations include:

```python
# Change page title and icon
st.set_page_config(
    page_title="Custom Title", 
    page_icon="🔍",
    layout="wide"
)

# Add custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)
```

## 🧠 Sample Questions & System Capabilities

Each question below is designed to test specific capabilities of the Document Intelligence System based on the available data.

### 🔎 Occupancy & Planning
- "Which office had the highest average occupancy in April 2025?" 
  *(Tests temporal data analysis across badge_data.csv files)*
- "Predict which offices will exceed capacity thresholds next month."
  *(Tests predictive modeling using historical badge data trends)*
- "What was the week-over-week change in occupancy across all offices for the past month?"
  *(Tests time-series analysis and cross-document aggregation)*
- "Which meeting rooms in Chicago have less than 50% utilization?"
  *(Tests granular data filtering in chi_meeting_room_utilization.csv)*
- "What's the correlation between office occupancy and day of the week across our locations?"
  *(Tests pattern recognition and multi-variable correlation analysis)*

### ⚡ Energy Cost Analysis
- "Which city has the highest average energy cost?"
  *(Tests basic aggregation across esg_metrics.csv files)*
- "Predict total energy cost in NYC for the next 6 months."
  *(Tests forecasting using time-series energy consumption data)*
- "How does energy usage per employee compare across all offices?"
  *(Tests cross-document normalization between employees.csv and esg_metrics.csv)*
- "What's the correlation between occupancy rates and energy consumption in Miami?"
  *(Tests multi-document correlation between badge_data.csv and esg_metrics.csv)*
- "Which office has shown the most improvement in energy efficiency over the last quarter?"
  *(Tests trend analysis and percentage change calculations)*

### 🌿 ESG Metrics
- "Which office performs best environmentally?"
  *(Tests multi-variable ranking across all esg_metrics.csv files)*
- "Where would occupancy policies cut energy usage most?"
  *(Tests hypothetical scenario modeling with badge data and energy patterns)*
- "Calculate the carbon footprint per employee across all offices."
  *(Tests data normalization between employees.csv and esg_metrics.csv files)*
- "How do our ESG metrics compare to industry benchmarks?"
  *(Tests external context integration with internal metrics)*
- "What would be the environmental impact of implementing a 4-day work week in our LA office?"
  *(Tests complex scenario modeling using badge, employee, and energy data)*

### 🗺️ Strategic Decisions
- "Which city should consolidate based on utilization?"
  *(Tests decision optimization based on multiple datasets)*
- "What is the impact of relocating 25% of NYC staff to Miami?"
  *(Tests scenario modeling across multiple offices and datasets)*
- "If we need to reduce total leased space by 15%, which offices should be prioritized and why?"
  *(Tests multi-criteria decision analysis using lease_market_data.csv and badge_data.csv)*
- "Based on current trends, where should we invest in expanding office space next year?"
  *(Tests trend extrapolation and predictive analysis across all locations)*
- "What would be the financial impact of converting 30% of NYC office space to hot-desking?"
  *(Tests financial modeling and workspace optimization analysis)*

### 🛂 Badge & Employee Data
- "How many employees are assigned to LA?"
  *(Tests basic count aggregation in la_employees.csv)*
- "Compare badge utilization between Chicago and NYC."
  *(Tests cross-document comparison between badge_data.csv files)*
- "Which departments have the highest and lowest office attendance rates?"
  *(Tests joined analysis between employees.csv and badge_data.csv files)*
- "Is there a correlation between seniority level and office attendance frequency?"
  *(Tests attribute correlation across employee directories and badge logs)*
- "What percentage of employees use the office less than once per week?"
  *(Tests frequency analysis and percentage calculations across datasets)*

### 🏢 Facilities & Leasing
- "What is the lease cost per square foot in Philadelphia?"
  *(Tests specific data extraction from phl_lease_market_data.csv)*
- "What is our total leased square footage in LA?"
  *(Tests summation calculation from la_lease_market_data.csv)*
- "Which office has the best and worst cost-per-employee ratio?"
  *(Tests derived metric calculation across lease data and employee headcounts)*
- "How do our lease costs compare to current market rates in each city?"
  *(Tests comparative analysis between internal and market data points)*
- "When are our lease renewal dates and what are the projected market rates at those times?"
  *(Tests date extraction and future projection from lease_market_data.csv files)*

### 📊 Data Visualization Requests
- "Create a chart showing occupancy rates by day of week for all offices."
  *(Tests data visualization capabilities and temporal pattern recognition)*
- "Visualize the correlation between meeting room bookings and total office attendance."
  *(Tests multi-document visualization between badge_data.csv and meeting_room_utilization.csv)*
- "Generate a heat map of energy usage by hour of day for each location."
  *(Tests complex visualization of time-series energy consumption data)*
- "Show me a comparison chart of space utilization vs. lease cost across offices."
  *(Tests multi-variable comparative visualization across document types)*
- "Create a dashboard visualization of key ESG metrics by location."
  *(Tests multi-document dashboard visualization capabilities)*

## ✨ Ready-to-Use AI Prompts

### 📈 Predict Headcount Needs
> "Based on current occupancy rates and growth projections, estimate where we need more seats in Q3 2025."

### 🏢 Recommend Office Consolidation
> "Identify underutilized offices below 50% capacity and suggest a phased consolidation plan with projected savings."

### 🔮 Forecast Occupancy
> "Predict May 2025 occupancy from April swipe patterns considering seasonal trends and upcoming holidays."

### 💸 Recommend Cost Savings
> "Suggest 3 data-driven strategies based on lease terms, occupancy patterns, and energy consumption to reduce our real estate costs by 20% while maintaining employee satisfaction."

### 🌱 ESG Improvement Plan
> "Analyze our current ESG metrics and recommend 5 actionable initiatives to improve our environmental performance while optimizing space utilization."

### 📋 Executive Summary Creation
> "Create an executive summary of our real estate portfolio performance with key metrics, trends, and recommendations for the board meeting."

### 🔍 Anomaly Detection
> "Identify any unusual patterns or anomalies in our office utilization data across all locations and suggest possible explanations."

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

---

🔥 Built to help Facilities and Management teams make better decisions — faster, smarter, and with confidence.