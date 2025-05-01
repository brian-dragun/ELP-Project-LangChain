# 🧠 AI-Powered Document Intelligence System

This system powers an interactive AI assistant that helps workplace teams analyze and understand internal business documents such as office occupancy data, ESG reports, and facility metrics. It combines language model reasoning, document search, and data intelligence into one user-friendly experience.

---

## 🔧 System Overview

It consists of two main components:

### 1. `ai_agent.py` – Core Retrieval and Response Engine
- Sets up a custom LLM (Lambda Labs) and vector store (ChromaDB).
- Uses HuggingFace embeddings for document retrieval.
- Defines a LangGraph pipeline with two key steps:
  - **`retrieve`**: Fetches relevant documents based on a user query.
  - **`generate_answer`**: Uses a structured prompt to generate an answer using retrieved content only.

### 2. `ai_agent_interactive_chat.py` – Interactive Chat System
- Adds a conversation interface with memory and advanced document intelligence.
- Auto-monitors the `/documents` folder for updates and refreshes the vector store as needed.
- Integrates specialized tools for:
  - Fact extraction and verification
  - Confidence analysis
  - Insight generation
  - Data visualization
  - Document summarization
  - Cross-document comparison
  - Regex-based pattern extraction

---

## 🧠 What Are Prompts?

Prompts are the structured instructions we give to the AI model to guide its response — like giving a detailed question or a script to follow. They define how the AI understands the user's request, what information it uses, and how it presents the output.

---

## 🎯 Purpose of the Prompts in Our System

In our document-aware AI agent, we use custom prompts to ensure the AI:

- Answers questions based only on the retrieved documents — not general internet knowledge.
- Extracts facts in a structured format, with confidence scores and sources.
- Verifies statements against the documents for accuracy.
- Thinks step by step before responding.
- Generates helpful insights and summaries from multiple documents.

These prompt templates allow us to enforce consistency, avoid hallucination, and align responses with real business data.

---

## 🧩 Types of Prompts We Use

| Prompt Type            | What It Does                                                               | Why It Matters for ELP                                           |
|------------------------|----------------------------------------------------------------------------|------------------------------------------------------------------|
| Question Answering     | Tells the AI to answer based on a specific context of documents.           | Ensures leasing or ESG decisions are made using real data only. |
| Fact Extraction        | Instructs the AI to find 5–7 key facts from documents, with sources.       | Helps quickly summarize critical details (e.g. headcount, cost).|
| Fact Verification      | Asks the AI to confirm if a statement is supported by the documents.       | Prevents false assumptions in reporting or recommendations.     |
| Self-Questioning       | Forces the AI to ask clarifying questions to better understand the user.   | Improves accuracy when user inputs are vague or incomplete.     |
| Reasoning Steps        | Prompts the AI to walk through its thought process step by step.           | Makes the AI’s decision-making transparent and auditable.       |
| Confidence Analysis    | Gets the AI to assign scores to each statement based on how well-supported.| Helps us trust or challenge AI-generated insights.              |
| Insight Generation     | Guides the AI to find actionable insights from office data.                | Automatically surfaces opportunities for optimization or savings.|
| Document Summary       | Prompts the AI to summarize a document concisely.                          | Speeds up review of long ESG or leasing docs.                   |
| Cross-Document Compare | Instructs the AI to find similarities, differences, and contradictions.    | Enables informed comparisons across office locations.           |

---

## 🗣️ How It’s Used

Users interact via a command-line interface. Example queries:

- “What is the energy usage trend in LA vs NYC?”
- “Summarize the ESG report for Chicago.”
- “Compare lease agreements between two cities.”

AI responses are:
- Grounded in actual documents.
- Verified for accuracy.
- Supplemented with reasoning, confidence scoring, and visuals.

---

## 🏁 Why This Matters for ELP/WLP

- Enables **data-driven decisions** for office planning and ESG strategy.
- Reduces manual review by surfacing **key facts and patterns** automatically.
- Adds explainability through **reasoning and verification**.
- Functions as a **virtual analyst** for internal workplace intelligence.