# ELP AI Document Intelligence System - Technical Documentation

## System Architecture

The ELP Document Intelligence System is built on a flexible, modular architecture that combines multiple AI components:

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│                  │     │                  │     │                  │
│  Document        │────▶│  Vector Store    │────▶│  AI Agent        │
│  Processing      │     │  (ChromaDB)      │     │  System          │
│                  │     │                  │     │                  │
└──────────────────┘     └──────────────────┘     └──────────────────┘
         │                                                 │
         │                                                 │
         ▼                                                 ▼
┌──────────────────┐                             ┌──────────────────┐
│                  │                             │                  │
│  Document        │                             │  Interactive     │
│  Intelligence    │◀───────────────────────────▶│  Chat Interface  │
│  Modules         │                             │                  │
└──────────────────┘                             └──────────────────┘
```

### Core Components

1. **Document Processing Pipeline**
   - `ingest_documents.py`: Processes various document types, extracts content, and creates enhanced documents with metadata
   - Handles CSV, PDF, text, HTML, Markdown, Excel, and Word documents
   - Performs special processing for CSV files with location-based data
   - Creates derived documents for cross-document analysis

2. **Vector Database**
   - ChromaDB persistent storage for document embeddings
   - HuggingFace embeddings (all-MiniLM-L6-v2)
   - Efficient document retrieval with similarity search
   - Hash-based change detection for incremental updates

3. **AI Agent System**
   - `ai_agent.py`: Core LLM interface with LambdaLabs API integration
   - Fallback mechanisms to handle API failures
   - LangSmith integration for run tracking and feedback

4. **Reasoning Service**
   - `reasoning.py`: Handles complex multi-step reasoning
   - Context retrieval with adaptive search
   - Confidence scoring and uncertainty detection

5. **Interactive Chat**
   - `ai_agent_interactive_chat.py`: User interface with natural language processing
   - Memory systems: Fact Memory, Conversation Memory, Summary Memory
   - Special command handling and session management

6. **Document Intelligence Modules**
   - **FactMemory**: Structured storage for extracted facts
   - **PatternExtractor**: Extracts structured data using regex
   - **DocumentSummarizer**: Generates concise summaries
   - **DataVisualizer**: Creates charts and visualizations
   - **InsightGenerator**: Automatically identifies key insights
   - **CrossDocumentAnalyzer**: Finds connections across documents
   - **SelfQuestioningTool**: Generates clarifying questions
   - **FactExtractionTool**: Extracts key facts from documents
   - **FactVerificationTool**: Verifies statements against sources

## 🧠 What Are Prompts?

Prompts are the structured instructions we give to the AI model to guide its response — like giving a detailed question or a script to follow. They define how the AI understands the user's request, what information it uses, and how it presents the output.

## 🎯 Purpose of the Prompts in Our System

In our document-aware AI agent, we use custom prompts to ensure the AI:

- Answers questions based only on the retrieved documents — not general internet knowledge.
- Extracts facts in a structured format, with confidence scores and sources.
- Verifies statements against the documents for accuracy.
- Thinks step by step before responding.
- Generates helpful insights and summaries from multiple documents.

These prompt templates allow us to enforce consistency, avoid hallucination, and align responses with real business data.

## 🧩 Types of Prompts We Use

| Prompt Type            | What It Does                                                               | Why It Matters for ELP                                           |
|------------------------|----------------------------------------------------------------------------|------------------------------------------------------------------|
| Question Answering     | Tells the AI to answer based on a specific context of documents.           | Ensures leasing or ESG decisions are made using real data only. |
| Fact Extraction        | Instructs the AI to find 5–7 key facts from documents, with sources.       | Helps quickly summarize critical details (e.g. headcount, cost).|
| Fact Verification      | Asks the AI to confirm if a statement is supported by the documents.       | Prevents false assumptions in reporting or recommendations.     |
| Self-Questioning       | Forces the AI to ask clarifying questions to better understand the user.   | Improves accuracy when user inputs are vague or incomplete.     |
| Reasoning Steps        | Prompts the AI to walk through its thought process step by step.           | Makes the AI's decision-making transparent and auditable.       |
| Confidence Analysis    | Gets the AI to assign scores to each statement based on how well-supported.| Helps us trust or challenge AI-generated insights.              |
| Insight Generation     | Guides the AI to find actionable insights from office data.                | Automatically surfaces opportunities for optimization or savings.|
| Document Summary       | Prompts the AI to summarize a document concisely.                          | Speeds up review of long ESG or leasing docs.                   |
| Cross-Document Compare | Instructs the AI to find similarities, differences, and contradictions.    | Enables informed comparisons across office locations.           |

## Data Flow

1. **Document Ingestion Flow**
   - Documents loaded from files → Text extraction → Chunking → Embedding → Vector storage
   - Special documents created for cross-document analysis
   - Document hashes saved for incremental updates

2. **Query Processing Flow**
   - User query → Retrieval → Context preparation → Reasoning → Response generation
   - Optional: Self-questioning, fact extraction, visualization

3. **Memory Management**
   - Fact memory persists across sessions
   - Conversation memory maintains context during a session
   - Document changes automatically detected and processed

## File Descriptions

### Core Files

- **ai_agent_interactive_chat.py**: Main chat interface and document intelligence features
- **ai_agent.py**: Core LLM integration with LambdaLabs and fallback mechanisms
- **config.py**: Configuration settings, API keys, and global parameters
- **reasoning.py**: Reasoning service for query processing and context retrieval
- **ingest_documents.py**: Document processing pipeline for vector database creation

### Supporting Files

- **chat_history.json**: Stores conversation history
- **fact_memory.json**: Persists extracted facts
- **documents_hash.txt**: Tracks document changes for incremental updates

### Document Folders

- **documents/**: Source documents for processing
  - Contains employee data, badge data, ESG metrics, lease data, and meeting room utilization for multiple office locations
- **chroma_db/**: Vector database storage
- **visualizations/**: Generated charts and graphs

## Technical Implementation Details

### Document Processing

The document processing system uses a sophisticated pipeline with special handling for different file types:

1. **CSV Processing**
   - Groups data by city/location
   - Creates comprehensive city documents
   - Generates summary statistics
   - Extracts employee counts for comparative analysis

2. **Text Splitting**
   - RecursiveCharacterTextSplitter with 1000 token chunks and 100 token overlap
   - Semantic splitting on paragraph and sentence boundaries

3. **Cross-Document Enhancement**
   - Creates "derived" documents that combine information
   - Generates comparative documents between cities
   - Builds an "all offices" document with combined metrics

4. **Incremental Updates**
   - Uses combined hash of modification time and content
   - Detects added, modified, and deleted files
   - Updates only changed documents in the vector store
   - Regenerates derived documents when source documents change

### Reasoning System

The reasoning service employs a simplified but powerful approach:

1. **Adaptive Retrieval**
   - Retrieves most relevant documents (k=15)
   - Performs specialized queries for certain types of questions
   - Combines results from multiple retrieval strategies

2. **Context Preparation**
   - Formats retrieved documents for LLM processing
   - Extracts metadata for source tracking

3. **Response Generation**
   - Single unified prompt for both reasoning and answer generation
   - Configurable detail level (basic, standard, detailed)

4. **Confidence Analysis**
   - Detects uncertainty phrases in responses
   - Calculates term overlap between response and source documents
   - Provides confidence scores for generated answers

### Interactive Features

The system includes several advanced interactive features:

1. **Self-Questioning**
   - Generates clarifying questions about user queries
   - Helps refine ambiguous questions

2. **Fact Extraction and Verification**
   - Extracts key facts from documents with confidence scores
   - Verifies statements against source material
   - Stores facts in persistent memory

3. **Data Visualization**
   - Extracts tabular data from text
   - Generates appropriate chart types based on data
   - Creates and saves visualizations as PNG files

4. **Cross-Document Analysis**
   - Compares information across multiple documents
   - Identifies connections and relationships
   - Generates insights based on multiple sources

5. **Document Monitoring**
   - Background watchdog thread monitors document changes
   - Automatic database refresh when documents change
   - Polling fallback if watchdog is not available

## LLM Integration

The system uses a custom LambdaLabsLLM implementation with:

1. **API Integration**
   - HTTP API calls to Lambda Labs endpoints
   - JSON parsing and error handling

2. **Fallback Mechanisms**
   - Primary model → Fallback model → Last resort models
   - Exponential backoff retry logic
   - Timeout and error handling

3. **LangSmith Integration**
   - Run tracking and history
   - Performance monitoring
   - Error analysis

## 🏁 Why This Matters for ELP

- Enables **data-driven decisions** for office planning and ESG strategy.
- Reduces manual review by surfacing **key facts and patterns** automatically.
- Adds explainability through **reasoning and verification**.
- Functions as a **virtual analyst** for internal workplace intelligence.

## Future Development Areas

1. **Advanced Reasoning**
   - Graph-based reasoning for complex queries
   - Multi-step planning for complex analysis

2. **Data Processing**
   - Support for more document formats
   - Improved data extraction from semi-structured documents

3. **User Experience**
   - Web interface with interactive visualizations
   - User feedback integration for continuous improvement

4. **Performance Optimization**
   - Caching frequently accessed documents
   - Parallel processing for large document sets

5. **Extended Intelligence**
   - Time-series analysis for trend detection
   - Predictive analytics for future forecasting

---

🔙 [Back to Project README](./README.md)