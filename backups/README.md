# 🏢📊 Document Q&A System with LangChain, LangGraph & Lambda Labs

This project enables powerful AI-driven Q&A and analytics over enterprise documents using **LangChain**, **LangGraph**, and a **Lambda Labs-hosted LLM**.

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

---

## 📘 Related Documentation

- 📄 [AI Document Intelligence Summary](./ELP_AI_Document_Intelligence_Summary.md) – Full system architecture, prompt design, and capability breakdown.

---

## ⚙️ AI Agent Setup

### 🧼 Clean Environment (if needed)
```bash
rm -rf .venv
```

### 📦 Step 1: Install Dependencies
```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip setuptools
pip install -r requirements.txt
```

### 📁 Step 2: Add Documents
Place files in the `documents/` folder.

### 🔐 Step 3: Set Configuration
Create a `.env` file in the root directory. See `.env.example` for required keys.

---

## 🚀 Usage

### 🧠 Run Basic Q&A Agent
```bash
python ai_agent.py
```

### 💬 Launch Interactive Chat
```bash
python ai_agent_interactive_chat.py
```

---

## 🧩 System Components

| File                          | Description                                                |
|-------------------------------|------------------------------------------------------------|
| `ai_agent.py`                 | Core LangGraph agent for document retrieval + answering    |
| `ai_agent_interactive_chat.py`| Interactive chat interface with tools + memory             |
| `ingest_documents.py`         | Utility to pre-process and embed documents (optional)      |
| `ask_question.py`             | CLI tool to ask a single question                          |
| `config.py`                   | Stores API keys and configuration                          |

---

## 🔬 Simulation Logic

### 🛠️ Assumptions:
- Badge rates, entry/exit ratios, and door prioritization
- Energy tied to badge activity per seat
- Over 90% occupancy triggers alerts
- City-based cost models and energy prices
- Realistic meeting room booking and ESG scoring

### 📈 Analysis Scenarios:
- Threshold Monitoring & Warnings
- Daily Energy Forecasts
- Future Occupancy Modeling
- Hybrid Work Cost Simulations
- Space Consolidation Risk Detection

### 🗂️ Output Files:
- `*_employees.csv`, `*_badge_data.csv`, `*_meeting_room_utilization.csv`
- `*_lease_market_data.csv`, `*_esg_metrics.csv`
- `occupancy_energy_analysis_v2.csv`

---

## 🤖 AI Agent Capabilities

- 📊 Real-Time Reporting
- 🔮 Forecasting Occupancy & Energy
- 🧩 ESG Strategy Recommendations
- 🧠 Strategic Planning Assistance
- 📉 Cost Optimization

---

## 🧠 Sample Questions

### 🔎 Occupancy & Planning
- “Which office had the highest average occupancy in April 2025?”
- “Predict which offices will exceed capacity thresholds next month.”

### ⚡ Energy Cost Analysis
- “Which city has the highest average energy cost?”
- “Predict total energy cost in NYC for the next 6 months.”

### 🌿 ESG Metrics
- “Which office performs best environmentally?”
- “Where would occupancy policies cut energy usage most?”

### 🗺️ Strategic Decisions
- “Which city should consolidate based on utilization?”
- “What is the impact of relocating 25% of NYC staff to Miami?”

### 🛂 Badge & Employee Data
- “How many employees are assigned to LA?”
- “Compare badge utilization between Chicago and NYC.”

### 🏢 Facilities & Leasing
- “What is the lease cost per square foot in Philadelphia?”
- “What is our total leased square footage in LA?”

---

## ✨ Ready-to-Use AI Prompts

### 📈 Predict Headcount Needs
> “Based on current occupancy rates... estimate where we need more seats.”

### 🏢 Recommend Office Consolidation
> “Identify underutilized offices below 50% and suggest closures.”

### 🔮 Forecast Occupancy
> “Predict May 2025 occupancy from April swipe patterns.”

### 💸 Recommend Cost Savings
> “Suggest 3 strategies based on lease, occupancy, and energy cost data.”

---

## 🧠 Complex Questions You Can Ask

- “Which office best balances sustainability and financials?”
- “Which space is most efficiently used?”
- “What is the correlation between employee satisfaction and ESG scores?”
- “Which lease needs renegotiation first?”
- “Which office should be the model for future expansions?”

---

## 💡 Tip:
Use `ai_agent_interactive_chat.py` for the best experience — including:
- Reasoning steps
- Confidence scoring
- Fact checking
- Insight generation
- Auto-visualization

---

## 🧠 Prompt Design Philosophy

Prompts are structured instructions to control what the AI does. In our system, they ensure:

- Answers are **grounded in ingested documents**
- Facts are **extracted with sources + confidence**
- Claims are **verified** before being presented
- Responses show **step-by-step reasoning**
- AI finds **insights and summaries** automatically

---

## 🎯 Prompt Types

| Prompt Type         | What It Does                                                        | Why It Matters for ELP                                      |
|---------------------|---------------------------------------------------------------------|--------------------------------------------------------------|
| Question Answering  | Answers using retrieved docs only                                   | Trustworthy, document-based answers                          |
| Fact Extraction     | Pulls key facts + sources                                           | Helps teams digest data fast                                 |
| Fact Verification   | Checks if a claim is true                                           | Prevents misinformation in business decisions                |
| Self-Questioning    | AI asks itself clarifying questions                                 | Improves precision on vague queries                          |
| Reasoning Steps     | Explains the steps it took to reach an answer                       | Transparent, auditable logic                                 |
| Confidence Analysis | Scores each part of the response                                    | Tells us how much to trust the answer                        |
| Insight Generation  | Surfaces patterns or outliers                                       | Suggests strategic actions                                   |
| Document Summary    | Generates concise summaries of documents                           | Saves time on long ESG or facility files                     |
| Cross-Compare       | Finds similarities and differences across documents                 | Enables deeper comparisons between cities/offices            |

---

# Document Intelligence System

An AI-powered system for analyzing documents and answering questions about office data across multiple locations.

## Features

- **Document Analysis**: Extract insights, patterns, and data from various document formats
- **Natural Language Interface**: Ask questions in plain language about your documents
- **Interactive Chat**: Have continuous conversations with context and memory
- **Data Visualization**: Auto-generate charts and graphs from your data
- **Fact Tracking**: Automatically extract and remember facts from documents
- **Cross-Document Intelligence**: Compare and analyze data across multiple sources
- **LangSmith Integration**: Performance tracking and evaluation

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

## Sample Questions

1. "How many employees work in the Los Angeles office?"
2. "What is the badge utilization rate in Chicago compared to NYC?"
3. "Which city has the highest meeting room utilization on weekends?"
4. "What's the trend in energy usage across all offices over time?"
5. "Compare the sustainability metrics of Chicago and Miami offices"
6. "Extract all dates from the NYC lease document"
7. "Summarize the ESG metrics across all locations"

---

🔥 Built to help Facilities and Management teams make better decisions — faster, smarter, and with confidence.