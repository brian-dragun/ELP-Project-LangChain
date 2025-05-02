#!/usr/bin/env python3
from config import Config
from ai_agent import llm, logger
from pydantic import Field, field_validator, BaseModel
from langchain.smith import RunEvalConfig, run_on_dataset
from langsmith.evaluation import StringEvaluator
from langsmith.run_trees import RunTree
from langsmith import Client
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.agents import initialize_agent, AgentType
from langchain.tools import BaseTool
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationSummaryMemory
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import List, Dict, Any, Optional, ClassVar, Type, Union, Pattern, Tuple
from datetime import datetime
import chromadb
import plotly.io as pio
import plotly.express as px
import matplotlib.pyplot as plt
import os
import sys
import json
import glob
import hashlib
import time
import threading
import re
import base64
import io
import uuid
import csv
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Langchain imports

# LangSmith imports

# pydantic imports

# Local imports

# Initialize LangSmith client
langsmith_client = Client(
    api_key=Config.LANGCHAIN_API_KEY,
    api_url=Config.LANGSMITH_ENDPOINT
)

# Define colors for terminal output


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class FactMemory:
    """A structured memory system that stores extracted facts from documents"""

    def __init__(self, memory_file="fact_memory.json"):
        self.memory_file = memory_file
        self.facts = self._load_facts()
        self.session_facts = []

    def _load_facts(self):
        """Load facts from file if it exists"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []

    def save_facts(self):
        """Save facts to file"""
        all_facts = self.facts + self.session_facts
        with open(self.memory_file, 'w') as f:
            json.dump(all_facts, f, indent=2)

    def add_fact(self, fact, source_document, confidence=0.0):
        """Add a new fact to memory"""
        fact_entry = {
            "fact": fact,
            "source": source_document,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "verified": False
        }
        self.session_facts.append(fact_entry)
        return fact_entry

    def get_relevant_facts(self, query, top_k=5):
        """Retrieve facts relevant to the query"""
        # Simple keyword matching for now - could be enhanced with embeddings
        relevant_facts = []
        all_facts = self.facts + self.session_facts

        query_terms = set(query.lower().split())
        for fact in all_facts:
            fact_text = fact["fact"].lower()
            overlap = sum(1 for term in query_terms if term in fact_text)
            if overlap > 0:
                relevant_facts.append((fact, overlap))

        # Sort by relevance and get top k
        relevant_facts.sort(key=lambda x: x[1], reverse=True)
        return [fact for fact, _ in relevant_facts[:top_k]]


class SelfQuestioningTool(BaseTool):
    """A tool for the agent to ask itself questions to clarify understanding"""

    name: ClassVar[str] = "self_questioning"
    description: ClassVar[str] = "Ask yourself clarifying questions to better understand the user's query"

    def _run(self, query: str) -> str:
        """Generate self-questions about the query"""
        prompt = f"""Based on the user's question: "{query}", 
        generate 2-3 clarifying questions that would help you better understand what they're asking.
        Focus on ambiguities, implied assumptions, or missing context that would help you give a better answer.
        Return only the questions, one per line."""

        response = llm.invoke(prompt)
        return response

    async def _arun(self, query: str) -> str:
        """Async implementation of self-questioning"""
        return self._run(query)


class FactExtractionTool(BaseTool):
    """A tool for extracting facts from retrieved documents"""

    name: ClassVar[str] = "fact_extraction"
    description: ClassVar[str] = "Extract key facts from retrieved documents"
    retriever: Any = Field(default=None)

    def __init__(self, retriever: Any = None, **kwargs):
        super().__init__(**kwargs)
        if retriever is not None:
            self.retriever = retriever

    def _run(self, query: str) -> str:
        """Extract facts relevant to the query from documents"""
        if self.retriever is None:
            return "Error: Retriever is not configured."

        docs = self.retriever.invoke(query)
        docs_text = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""Extract 5-7 key facts from the following documents that are relevant to: "{query}"
        
        Documents:
        {docs_text}
        
        For each fact, include:
        - The specific fact
        - The source document or data point it comes from
        - A confidence score (1-100%) indicating how certain you are about this fact
        
        Format each fact as: "FACT: [fact] | SOURCE: [source] | CONFIDENCE: [score]"
        """

        response = llm.invoke(prompt)
        return response

    async def _arun(self, query: str) -> str:
        """Async implementation of fact extraction"""
        return self._run(query)


class FactVerificationTool(BaseTool):
    """A tool for verifying facts against source documents"""

    name: ClassVar[str] = "fact_verification"
    description: ClassVar[str] = "Verify a statement against the retrieved documents"
    retriever: Any = Field(default=None)

    def __init__(self, retriever: Any = None, **kwargs):
        super().__init__(**kwargs)
        if retriever is not None:
            self.retriever = retriever

    def _run(self, statement: str) -> str:
        """Verify if a statement is supported by the documents"""
        if self.retriever is None:
            return "Error: Retriever is not configured."

        # Get relevant documents
        docs = self.retriever.invoke(statement)
        docs_text = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""Verify if the following statement is supported by the provided documents.
        
        Statement: "{statement}"
        
        Documents:
        {docs_text}
        
        Respond with:
        - SUPPORTED: If the statement is directly supported by the documents
        - PARTIALLY SUPPORTED: If parts of the statement are supported but not all
        - UNSUPPORTED: If the statement contradicts the documents or has no supporting evidence
        - UNCLEAR: If there isn't enough information to verify
        
        Also include:
        - The relevant text from the documents that supports or contradicts the statement
        - A confidence score (1-100%) for your verification
        """

        response = llm.invoke(prompt)
        return response

    async def _arun(self, statement: str) -> str:
        """Async implementation of fact verification"""
        return self._run(statement)


class ConfidenceAnalyzer:
    """Analyzes responses to determine confidence levels for different parts"""

    def analyze_response(self, response, query, context):
        """Analyze a response for confidence levels"""

        prompt = f"""Analyze the following answer to a user query and assign confidence levels to different parts of the response.

User Query: "{query}"

Context Used:
{context}

Response to Analyze:
{response}

For each key statement or claim in the response, assign a confidence score (0-100%) based on:
1. How directly it's supported by the context
2. Whether it includes speculation or assumptions
3. The completeness of the information provided

Format your analysis as:
STATEMENT: [statement text]
CONFIDENCE: [score]%
REASONING: [brief explanation for the confidence score]

Focus on the most important 3-5 statements in the response.
"""

        confidence_analysis = llm.invoke(prompt)
        return confidence_analysis

    def extract_confidence_scores(self, confidence_analysis):
        """Extract the confidence scores from the analysis"""
        pattern = r"STATEMENT:.*?CONFIDENCE: (\d+)%"
        matches = re.findall(pattern, confidence_analysis, re.DOTALL)
        if matches:
            scores = [int(score) for score in matches]
            return {
                "overall": sum(scores) / len(scores),
                "min": min(scores),
                "max": max(scores),
                "scores": scores,
                "analysis": confidence_analysis
            }
        return {"overall": 0, "min": 0, "max": 0, "scores": [], "analysis": confidence_analysis}


class PatternExtractor:
    """Extract structured data using regular expressions"""

    def __init__(self):
        # Common pattern matchers for business data
        self.patterns = {
            "currency": r"\$\s*\d+(?:\.\d{2})?|\d+\s*(?:dollars|USD)",
            "percentage": r"\d+(?:\.\d+)?\s*%",
            "date": r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}",
            "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "phone": r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
            "time": r"\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?",
            "numeric": r"[-+]?\d*\.?\d+"
        }

    def add_pattern(self, name: str, pattern: str):
        """Add a new pattern to the extractor"""
        try:
            re.compile(pattern)  # Check if pattern is valid
            self.patterns[name] = pattern
            return True
        except re.error:
            return False

    def extract_all_patterns(self, text: str) -> Dict[str, List[str]]:
        """Extract all known patterns from text"""
        results = {}
        for pattern_name, pattern in self.patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                results[pattern_name] = matches
        return results

    def extract_pattern(self, text: str, pattern_name: str) -> List[str]:
        """Extract a specific pattern from text"""
        if pattern_name not in self.patterns:
            return []
        return re.findall(self.patterns[pattern_name], text)

    def extract_custom_pattern(self, text: str, pattern: str) -> List[str]:
        """Extract using a custom pattern"""
        try:
            return re.findall(pattern, text)
        except re.error:
            return []


class DocumentSummarizer:
    """Generate concise summaries of documents or collections"""

    def summarize_document(self, document_text: str, max_length: int = 200) -> str:
        """Summarize a single document"""
        prompt = f"""Summarize the following document in a concise way (maximum {max_length} words):

{document_text[:10000]}  # Limit input to prevent token overflow

Focus on the key information, important facts, and main conclusions.
"""
        summary = llm.invoke(prompt)
        return summary

    def summarize_collection(self, documents: List[str], max_length: int = 300) -> str:
        """Summarize a collection of documents"""
        # First, summarize each document briefly
        doc_summaries = []
        for i, doc in enumerate(documents[:10]):  # Limit to 10 docs
            short_summary = self.summarize_document(doc, max_length=50)
            doc_summaries.append(f"Document {i+1}: {short_summary}")

        # Then, create an overall summary
        all_summaries = "\n\n".join(doc_summaries)
        prompt = f"""Create a comprehensive summary of the following document collection (maximum {max_length} words):

{all_summaries}

Your summary should:
1. Identify the main themes across documents
2. Highlight key findings or data points
3. Note any contradictions or inconsistencies
4. Provide an overall assessment of the information
"""
        collection_summary = llm.invoke(prompt)
        return collection_summary


class DataVisualizer:
    """Generate visualizations from document data"""

    def __init__(self, output_dir="visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.visualization_history = []

    def _extract_table_data(self, text: str) -> pd.DataFrame:
        """Extract tabular data from text"""
        # Try to detect CSV format first
        if ',' in text and '\n' in text:
            try:
                # Use StringIO to convert string to file-like object for pandas
                data = pd.read_csv(io.StringIO(text))
                return data
            except Exception:
                pass

        # If CSV extraction fails, ask LLM to extract structured data
        prompt = f"""Extract the tabular data from the following text and format it as CSV:

{text}

Format:
column1,column2,column3
value1,value2,value3
...

Only return the CSV data, nothing else.
"""
        csv_result = llm.invoke(prompt)

        # Try to parse the extracted CSV
        try:
            data = pd.read_csv(io.StringIO(csv_result))
            return data
        except Exception:
            # Return empty DataFrame if all attempts fail
            return pd.DataFrame()

    def _plot_to_base64(self, plt_figure) -> str:
        """Convert matplotlib figure to base64 string for display"""
        buf = io.BytesIO()
        plt_figure.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        return img_str

    def _save_visualization(self, plt_figure, title: str) -> str:
        """Save visualization to file and return the path"""
        filename = f"{title.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt_figure.savefig(filepath, bbox_inches='tight')
        plt.close(plt_figure)
        return filepath

    def generate_visualization(self, data_text: str, chart_type: str = None, title: str = None) -> Tuple[str, str]:
        """Generate visualization from text data"""
        # Extract tabular data
        df = self._extract_table_data(data_text)

        if df.empty or len(df) < 2:
            return None, "Could not extract sufficient data for visualization"

        # If chart type not specified, determine a suitable type
        if not chart_type:
            num_numerical = sum(pd.api.types.is_numeric_dtype(
                df[col]) for col in df.columns)
            num_categorical = sum(not pd.api.types.is_numeric_dtype(
                df[col]) for col in df.columns)

            if num_numerical >= 2:
                chart_type = "line" if len(df) > 5 else "bar"
            elif num_numerical == 1 and num_categorical >= 1:
                chart_type = "bar"
            else:
                chart_type = "pie" if len(df) <= 10 else "bar"

        # Create appropriate visualization
        fig, ax = plt.subplots(figsize=(10, 6))

        try:
            if chart_type == "line":
                numeric_cols = [
                    col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
                if len(numeric_cols) >= 2:
                    x_col, y_col = numeric_cols[0], numeric_cols[1]
                    df.plot(x=x_col, y=y_col, kind='line', ax=ax)
                else:
                    df.plot(kind='line', ax=ax)

            elif chart_type == "bar":
                if len(df.columns) >= 2:
                    x_col = df.columns[0]
                    y_col = [
                        col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
                    if y_col:
                        df.plot(x=x_col, y=y_col[0], kind='bar', ax=ax)
                    else:
                        df.plot(kind='bar', ax=ax)
                else:
                    df.plot(kind='bar', ax=ax)

            elif chart_type == "pie":
                # Use first numeric column for values, first non-numeric for labels
                value_col = next((col for col in df.columns if pd.api.types.is_numeric_dtype(
                    df[col])), df.columns[-1])
                label_col = next((col for col in df.columns if not pd.api.types.is_numeric_dtype(
                    df[col])), df.columns[0])
                df.plot(kind='pie', y=value_col,
                        labels=df[label_col], autopct='%1.1f%%', ax=ax)

            elif chart_type == "scatter":
                numeric_cols = [
                    col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
                if len(numeric_cols) >= 2:
                    df.plot(
                        x=numeric_cols[0], y=numeric_cols[1], kind='scatter', ax=ax)
                else:
                    return None, "Not enough numeric columns for scatter plot"
            else:
                df.plot(kind='bar', ax=ax)
        except Exception as e:
            return None, f"Error creating visualization: {str(e)}"

        # Set title
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"{chart_type.capitalize()} Chart of Data")

        # Save and convert to base64
        viz_path = self._save_visualization(fig, title or chart_type)
        base64_img = self._plot_to_base64(fig)
        plt.close(fig)

        # Store visualization in history
        viz_entry = {
            "title": title or f"{chart_type.capitalize()} Chart",
            "chart_type": chart_type,
            "path": viz_path,
            "timestamp": datetime.now().isoformat()
        }
        self.visualization_history.append(viz_entry)

        return base64_img, viz_path


class InsightGenerator:
    """Automatically generate insights from document data"""

    def generate_insights(self, context_docs: List[Any], top_k: int = 3) -> List[Dict[str, Any]]:
        """Generate key insights from the provided documents"""
        # Combine document content
        combined_text = "\n\n".join([doc.page_content for doc in context_docs])

        prompt = f"""Analyze the following document data and generate {top_k} key insights.
        Each insight should reveal something interesting, unexpected, or actionable from the data.

Document Data:
{combined_text}

For each insight:
1. Describe the insight in a clear, concise sentence
2. Provide supporting evidence or specific data points
3. Explain why this insight is notable or valuable
4. Assign a confidence score (1-100%)
5. Suggest one possible action based on this insight

Format each insight as:
INSIGHT: [description]
EVIDENCE: [supporting data]
IMPORTANCE: [why it matters]
CONFIDENCE: [score]%
ACTION: [suggested action]

Only include insights that are directly supported by the data.
"""
        insights_text = llm.invoke(prompt)

        # Parse the insights
        insights = []
        insight_pattern = r"INSIGHT: (.*?)EVIDENCE: (.*?)IMPORTANCE: (.*?)CONFIDENCE: (\d+)%\s*ACTION: (.*?)(?=INSIGHT:|$)"
        matches = re.findall(insight_pattern, insights_text, re.DOTALL)

        for match in matches:
            if len(match) == 5:
                insight = {
                    "description": match[0].strip(),
                    "evidence": match[1].strip(),
                    "importance": match[2].strip(),
                    "confidence": int(match[3]),
                    "action": match[4].strip(),
                    "timestamp": datetime.now().isoformat()
                }
                insights.append(insight)

        return insights


class CrossDocumentAnalyzer:
    """Analyze information across multiple documents"""

    def compare_documents(self, docs: List[Any], comparison_aspect: str = None) -> Dict[str, Any]:
        """Compare multiple documents based on a specific aspect"""
        # Get document contents and metadata
        doc_contents = []
        for i, doc in enumerate(docs):
            content = doc.page_content
            source = doc.metadata.get(
                'source', f"Document {i+1}") if hasattr(doc, 'metadata') else f"Document {i+1}"
            doc_contents.append((source, content))

        # If comparison aspect is not provided, try to determine it
        aspect_prompt = ""
        if comparison_aspect:
            aspect_prompt = f"Focus specifically on comparing the documents in terms of: {comparison_aspect}."

        # Format document contents for the prompt
        formatted_docs = ""
        for source, content in doc_contents:
            formatted_docs += f"\n\n--- {source} ---\n{content}"

        prompt = f"""Analyze and compare the following documents:
{formatted_docs}

{aspect_prompt}

Provide:
1. A concise summary of each document's key information
2. Major similarities across the documents
3. Key differences or contradictions between documents
4. An overall assessment of reliability and consistency
5. Any data trends or patterns visible across documents

Format your response with clear headings for each section.
"""
        comparison = llm.invoke(prompt)

        # Create structured output
        result = {
            "comparison": comparison,
            "documents": [source for source, _ in doc_contents],
            "timestamp": datetime.now().isoformat()
        }

        return result

    def find_connections(self, docs: List[Any], query: str = None) -> List[Dict[str, Any]]:
        """Find connections and relationships between documents"""
        # Extract document contents and metadata
        doc_contents = []
        for i, doc in enumerate(docs):
            content = doc.page_content
            source = doc.metadata.get(
                'source', f"Document {i+1}") if hasattr(doc, 'metadata') else f"Document {i+1}"
            doc_contents.append((source, content))

        # Create prompt
        connections_prompt = f"Specifically look for connections related to: {query}" if query else ""

        # Format document contents
        formatted_docs = ""
        for source, content in doc_contents:
            # Limit content length
            formatted_docs += f"\n\n--- {source} ---\n{content[:1000]}..."

        prompt = f"""Analyze the following documents and identify meaningful connections between them:
{formatted_docs}

{connections_prompt}

For each significant connection you find:
1. Describe the connection in a clear, concise sentence
2. Specify which documents are connected
3. Provide the relevant excerpts from each document
4. Explain the significance of this connection
5. Assign a strength score (1-100%) to indicate how strong/reliable this connection is

Focus on unexpected or non-obvious connections that provide additional insight.
"""
        connections_text = llm.invoke(prompt)

        # Parse the connections
        connections = []

        # Simple parsing - could be enhanced with more robust parsing
        sections = connections_text.split("\n\n")
        current_connection = {}

        for section in sections:
            if section.strip().startswith("Connection"):
                if current_connection and "description" in current_connection:
                    connections.append(current_connection)
                current_connection = {"description": section.strip()}
            elif "Documents:" in section or "Document:" in section:
                current_connection["documents"] = section.strip()
            elif "Excerpts:" in section or "Excerpt:" in section:
                current_connection["excerpts"] = section.strip()
            elif "Significance:" in section:
                current_connection["significance"] = section.strip()
            elif "Strength:" in section or "% strength" in section.lower() or "% confidence" in section.lower():
                # Extract percentage
                match = re.search(r"(\d+)%", section)
                if match:
                    current_connection["strength"] = int(match.group(1))
                else:
                    current_connection["strength"] = 0

        # Add the last connection if there is one
        if current_connection and "description" in current_connection:
            connections.append(current_connection)

        return connections


class DocumentChatV2:
    def __init__(self, auto_refresh=True):
        # Basic setup
        self.history = []
        self.documents_path = os.path.join(
            os.path.dirname(__file__), 'documents')
        self.chroma_path = os.path.join(os.path.dirname(__file__), 'chroma_db')
        self.hash_file = os.path.join(
            os.path.dirname(__file__), 'documents_hash.txt')
        self.client = chromadb.PersistentClient(path=self.chroma_path)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.monitor_running = False
        self.last_check_time = time.time()

        # Advanced memory systems
        self.fact_memory = FactMemory()
        self.conversation_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.summary_memory = ConversationSummaryMemory(
            llm=llm,
            memory_key="conversation_summary"
        )

        # Check if documents have changed and refresh if needed
        if auto_refresh and self._documents_changed():
            self._refresh_documents()

        # Set up vector store and retriever
        self.vectorstore = Chroma(
            client=self.client,
            collection_name="document_collection",
            embedding_function=self.embeddings
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

        # Set up advanced components
        self.confidence_analyzer = ConfidenceAnalyzer()
        self.reasoning_mode = True

        # Set up tools for agent
        self.tools = [
            SelfQuestioningTool(),
            FactExtractionTool(retriever=self.retriever),
            FactVerificationTool(retriever=self.retriever)
        ]

        # Start background document monitoring thread
        if auto_refresh:
            self._start_document_monitor()

    def _start_document_monitor(self):
        """Start a background thread to monitor document changes using watchdog"""
        self.monitor_running = True

        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler

            class DocumentChangeHandler(FileSystemEventHandler):
                def __init__(self, chat_instance):
                    self.chat_instance = chat_instance
                    self.last_refresh_time = time.time()
                    self.refresh_cooldown = 60  # Minimum seconds between refreshes
                    self.silent_mode = True  # Only show messages for actual changes

                def on_any_event(self, event):
                    if event.src_path.endswith(('.csv', '.txt', '.pdf', '.html', '.md', '.xlsx', '.docx')):
                        current_time = time.time()
                        # Only refresh if it's been at least 60 seconds since last refresh
                        if current_time - self.last_refresh_time > self.refresh_cooldown:
                            # Don't print anything yet - we'll check if changes were actually made
                            silent_result = self.chat_instance._refresh_documents(silent=self.silent_mode)
                            
                            # Only print message if there were actual changes
                            if not self.silent_mode or (isinstance(silent_result, dict) and sum(silent_result.values()) > 0):
                                print(
                                    f"\n{Colors.YELLOW}File change detected: {event.src_path}! Database updated.{Colors.ENDC}")
                                print("\n👤 You: ", end="")
                                sys.stdout.flush()
                                
                            self.last_refresh_time = current_time

            # Create and start the watchdog observer
            self.event_handler = DocumentChangeHandler(self)
            self.observer = Observer()
            self.observer.schedule(
                self.event_handler, self.documents_path, recursive=True)
            self.observer.daemon = True
            self.observer.start()
            logger.info(f"Started watchdog monitor on {self.documents_path}")

        except ImportError:
            # Fall back to polling method if watchdog is not installed
            logger.warning(
                "Watchdog library not found, falling back to polling method")
            self.monitor_thread = threading.Thread(
                target=self._monitor_documents, daemon=True)
            self.monitor_thread.start()

    def _monitor_documents(self):
        """Legacy monitor method using polling (only used if watchdog is not available)"""
        check_interval = 30  # Check every 30 seconds (increased from 10 to reduce polling frequency)

        while self.monitor_running:
            time.sleep(check_interval)

            # Check if documents have changed
            if self._documents_changed():
                current_time = time.time()
                # Only refresh if it's been at least 60 seconds since last refresh
                if current_time - self.last_check_time > 60:
                    print(
                        f"\n{Colors.YELLOW}Document changes detected via polling! Refreshing database...{Colors.ENDC}")
                    self._refresh_documents()
                    self.last_check_time = current_time
                    print("\n👤 You: ", end="")
                    sys.stdout.flush()

    def _get_documents_hash(self):
        """Generate a hash of all documents in the documents folder"""
        files = sorted(glob.glob(f"{self.documents_path}/*.*"))
        hash_content = ""

        for file_path in files:
            hash_content += file_path
            if os.path.exists(file_path):
                hash_content += str(os.path.getmtime(file_path))

        return hashlib.md5(hash_content.encode()).hexdigest()

    def _documents_changed(self):
        """Check if documents have been added, modified, or removed"""
        current_hash = self._get_documents_hash()

        # If hash file exists, compare with stored hash
        if os.path.exists(self.hash_file):
            with open(self.hash_file, 'r') as f:
                stored_hash = f.read().strip()
                return current_hash != stored_hash

        return True  # If hash file doesn't exist, consider it changed

    def _save_documents_hash(self):
        """Save the current hash of documents"""
        current_hash = self._get_documents_hash()
        with open(self.hash_file, 'w') as f:
            f.write(current_hash)

    def _refresh_documents(self, silent=False):
        """Re-ingest documents only if needed using the hash-based incremental system"""
        if not silent:
            print("Checking document database...")

        try:
            # Import the ingest_documents module
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from ingest_documents import update_database_incrementally, ingest_documents

            # Check if vector store exists and perform incremental update
            change_counts = update_database_incrementally(
                documents_dir=self.documents_path,
                chroma_path=self.chroma_path,
                hash_file=self.hash_file,
                show_progress=not silent,
                verbose=not silent
            )

            # Check if changes were made or if there was a full rebuild
            if not silent:
                if change_counts.get('full_rebuild', 0) > 0:
                    print(
                        f"{Colors.GREEN}Database fully rebuilt with all documents!{Colors.ENDC}")
                elif sum(change_counts.values()) == 0:
                    print(
                        f"{Colors.GREEN}Database is up to date. No changes needed.{Colors.ENDC}")
                else:
                    print(f"{Colors.GREEN}Database updated incrementally!{Colors.ENDC}")
                    print(
                        f"Changes: {change_counts['added']} added, {change_counts['modified']} modified, {change_counts['deleted']} deleted")

            # Reinitialize vector store and retriever
            self.client = chromadb.PersistentClient(path=self.chroma_path)
            self.vectorstore = Chroma(
                client=self.client,
                collection_name="document_collection",
                embedding_function=self.embeddings
            )
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": 5})

            # Ensure self.tools exists before trying to update it
            if not hasattr(self, 'tools'):
                # Initialize tools if they don't exist
                self.tools = [
                    SelfQuestioningTool(),
                    FactExtractionTool(retriever=self.retriever),
                    FactVerificationTool(retriever=self.retriever)
                ]
            else:
                # Update tools with the new retriever
                for tool in self.tools:
                    if hasattr(tool, 'retriever'):
                        tool.retriever = self.retriever

            # Update last check time
            self.last_check_time = time.time()

            # Return change counts for the caller to determine if changes happened
            return change_counts

        except ImportError:
            if not silent:
                print(
                    f"{Colors.RED}Could not import ingest_documents module. Please ensure it exists.{Colors.ENDC}")
            return False

        except Exception as e:
            if not silent:
                print(f"{Colors.RED}Error refreshing documents: {str(e)}{Colors.ENDC}")
                logger.exception("Error refreshing documents")
            return False

    def get_relevant_context(self, query):
        """Retrieve relevant document snippets for the query"""
        docs = self.retriever.invoke(query)
        return docs

    def extract_facts_from_context(self, query, context_docs):
        """Extract facts from retrieved document context"""
        context_text = "\n\n".join([doc.page_content for doc in context_docs])

        # Use the fact extraction tool
        fact_tool = next(
            (tool for tool in self.tools if tool.name == "fact_extraction"), None)
        if fact_tool:
            facts_text = fact_tool._run(query)

            # Parse facts
            fact_pattern = r"FACT: (.*?) \| SOURCE: (.*?) \| CONFIDENCE: (\d+)%"
            facts = re.findall(fact_pattern, facts_text)

            # Store facts in memory
            extracted_facts = []
            for fact_text, source, confidence_str in facts:
                try:
                    confidence = float(confidence_str) / 100.0
                except:
                    confidence = 0.5

                fact = self.fact_memory.add_fact(
                    fact_text.strip(), source.strip(), confidence)
                extracted_facts.append(fact)

            return extracted_facts

        return []

    def format_prompt_with_history(self, query, context_docs, self_questions=None, facts=None):
        """Format the prompt with chat history, relevant facts, and document context"""
        # Extract document context
        context_str = "\n\n".join([doc.page_content for doc in context_docs])

        # Get relevant facts from memory
        memory_facts = self.fact_memory.get_relevant_facts(query)

        # Format facts section
        facts_str = ""
        if facts or memory_facts:
            facts_str = "Relevant Facts:\n"

            # Include newly extracted facts
            if facts:
                for fact in facts:
                    facts_str += f"- {fact['fact']} (Confidence: {fact['confidence']:.0%})\n"

            # Include facts from memory
            if memory_facts:
                for fact in memory_facts:
                    facts_str += f"- {fact['fact']} (From memory, Source: {fact['source']})\n"

            facts_str += "\n"

        # Format self-questions section
        questions_str = ""
        if self_questions:
            questions_str = "Clarifying Questions:\n" + self_questions + "\n\n"

        # Get conversation summary if available
        conversation_summary = ""
        try:
            if hasattr(self.summary_memory, 'moving_summary_buffer') and self.summary_memory.moving_summary_buffer:
                conversation_summary = f"Conversation Summary:\n{self.summary_memory.moving_summary_buffer}\n\n"
            elif hasattr(self.summary_memory, 'buffer') and self.summary_memory.buffer:
                conversation_summary = f"Conversation Summary:\n{self.summary_memory.buffer}\n\n"
            elif hasattr(self.summary_memory, 'buffer_as_str') and callable(getattr(self.summary_memory, 'buffer_as_str')):
                summary = self.summary_memory.buffer_as_str()
                if summary:
                    conversation_summary = f"Conversation Summary:\n{summary}\n\n"
        except Exception as e:
            # If we can't get the summary, just continue without it
            pass

        # Format the full prompt
        full_prompt = f"""Answer the following question based on the provided context, relevant facts, and our previous conversation.

Current date: {datetime.now().strftime('%B %d, %Y')}

{conversation_summary}{facts_str}{questions_str}
Document Context:
{context_str}

Question: {query}

Before providing your final answer:
1. Think step by step about what information is needed
2. Consider what assumptions you're making
3. Verify claims against the document context
4. Assign confidence levels to different parts of your answer

Provide a clear and detailed answer using only the information from the context and known facts.
If the answer cannot be determined from the available information, say "I don't have enough information to answer this question."

Answer:"""
        return full_prompt

    def verify_response(self, response, context_docs):
        """Verify the response against the source documents"""
        # Use the fact verification tool
        verify_tool = next(
            (tool for tool in self.tools if tool.name == "fact_verification"), None)
        if verify_tool:
            verification = verify_tool._run(response)
            return verification

        return "Unable to verify response."

    def add_reasoning_steps(self, query, context_docs):
        """Generate explicit reasoning steps for answering the query"""
        context_str = "\n\n".join(
            # Limit context for reasoning
            [doc.page_content for doc in context_docs[:2]])

        prompt = f"""Given this question: "{query}" and the following context:

{context_str}

Walk through your reasoning process step by step:
1. What are the key elements of this question?
2. What relevant information do we have in the context?
3. What calculations or comparisons need to be made?
4. What assumptions or limitations should be considered?
5. What is the logical path to the answer?

Format your response with clear STEP 1, STEP 2, etc. headings."""

        reasoning = llm.invoke(prompt)
        return reasoning

    def process_query(self, query):
        """Process a user query and return an answer using the unified reasoning system"""
        try:
            # Import the reasoning service
            from reasoning import ReasoningService
            from ai_agent import llm

            # Initialize the reasoning service if needed
            if not hasattr(self, 'reasoning_service'):
                self.reasoning_service = ReasoningService(
                    llm=llm,
                    chroma_path=self.chroma_path
                )

            # Process the query using the unified reasoning system
            detail_level = 'detailed' if self.reasoning_mode else 'standard'
            result = self.reasoning_service.process_query(
                query, detail_level=detail_level)

            # Update conversation memory with the new exchange
            if "answer" in result:
                self.conversation_memory.save_context(
                    {"input": query}, {"output": result["answer"]})
                self.summary_memory.save_context(
                    {"input": query}, {"output": result["answer"]})
                self.history.append((query, result["answer"]))

            return result

        except Exception as e:
            logger.exception("Error processing query")
            return {
                "answer": f"Error processing your question: {str(e)}",
                "error": str(e)
            }


class DocumentChatV3(DocumentChatV2):
    """Enhanced document chat system with advanced Document Intelligence capabilities"""

    def __init__(self, auto_refresh=True):
        super().__init__(auto_refresh=auto_refresh)

        # Initialize Document Intelligence components
        self.pattern_extractor = PatternExtractor()
        self.document_summarizer = DocumentSummarizer()
        self.data_visualizer = DataVisualizer(output_dir="visualizations")
        self.insight_generator = InsightGenerator()
        self.cross_doc_analyzer = CrossDocumentAnalyzer()

        # Initialize LangSmith components
        self.langsmith_enabled = Config.LANGSMITH_TRACING_ENABLED
        self.langsmith_client = langsmith_client if self.langsmith_enabled else None
        self.run_ids = []  # Store run IDs for tracking

        # Track insights and visualizations
        self.insights = []
        self.auto_insights_mode = True  # Automatically suggest insights
        self.visualization_mode = True  # Generate visualizations when appropriate
        self.charts_generated = []

        # Replace welcome message
        print(f"\n{Colors.HEADER}" + "="*60 + Colors.ENDC)
        print(
            f"{Colors.BOLD}🤖 Document-Aware Chat System v3 with Document Intelligence{Colors.ENDC}")
        print(f"{Colors.HEADER}" + "="*60 + Colors.ENDC)
        print("Ask questions about your office data across Chicago, LA, Miami, NYC, and Philadelphia.")
        print(f"{Colors.CYAN}Enhanced with: Memory, Reasoning, Self-Questioning, Fact Checking, and Confidence Scoring{Colors.ENDC}")
        print(f"{Colors.GREEN}NEW Document Intelligence Features:{Colors.ENDC}")
        print(f"  {Colors.YELLOW}✓{Colors.ENDC} Automatic Data Insights")
        print(f"  {Colors.YELLOW}✓{Colors.ENDC} Data Visualization Generation")
        print(f"  {Colors.YELLOW}✓{Colors.ENDC} Cross-Document Analysis")
        print(f"  {Colors.YELLOW}✓{Colors.ENDC} Auto-Summarization")
        print(
            f"  {Colors.YELLOW}✓{Colors.ENDC} Regular Expression Pattern Extraction")
        print(f"  {Colors.YELLOW}✓{Colors.ENDC} LangSmith Tracking & Feedback")
        print(f"{Colors.YELLOW}Commands:{Colors.ENDC}")
        print("  exit/quit - End the session")
        print("  help - Show sample questions")
        print("  clear - Clear chat history")
        print("  refresh - Manually refresh document database")
        print("  facts - Show learned facts")
        print("  runs - View LangSmith run links")
        print("  insights - Show generated insights")
        print("  insights on/off - Toggle automatic insights")
        print("  visualize [data] - Create visualization from data")
        print("  summarize [document] - Summarize a document")
        print("  compare [docs] - Compare multiple documents")
        print("  extract [pattern] - Extract patterns from text")
        print("  reasoning on/off - Toggle reasoning steps visibility")
        print(f"{Colors.HEADER}" + "="*60 + Colors.ENDC + "\n")

    def display_response(self, result):
        """Display the response with formatting for different components"""
        if "error" in result:
            print(f"{Colors.RED}{result['answer']}{Colors.ENDC}")
            return
        
        # Extract any final answer pattern from the original answer
        answer_text = result['answer']
        final_answer = None
        
        # Check for final answer pattern and extract it
        final_answer_match = re.search(
            r'(\*\*Final Answer\*\*:|\*\*Final Answer:\*\*|Final Answer:)(.*?)(\.|$)',
            answer_text,
            re.IGNORECASE | re.DOTALL
        )
        
        if final_answer_match:
            final_answer = final_answer_match.group(0)
            # Remove the final answer from the original text
            answer_text = answer_text.replace(final_answer, "").strip()
        
        # Apply green color to boxed answers in the remaining answer text
        if "$\\boxed{" in answer_text:
            # Find all boxed answers and color them green
            answer_text = re.sub(
                r'\$\\boxed\{(.*?)\}\$',
                f"{Colors.GREEN}$\\\\boxed{{\\1}}${Colors.ENDC}",
                answer_text
            )
        
        # Display the main answer without the final answer part
        print(f"{answer_text}")
        
        # Show verification if available
        if "verification" in result and result["verification"]:
            print(f"{Colors.YELLOW}Verification:{Colors.ENDC}")
            # Handle both string and dictionary verification formats
            if isinstance(result["verification"], str):
                verification_lines = result["verification"].split("\n")
                # Limit to first 5 lines for brevity
                for line in verification_lines[:5]:
                    print(f"{Colors.YELLOW}{line}{Colors.ENDC}")
            else:
                # Handle dictionary format
                verification_dict = result["verification"]
                if "status" in verification_dict:
                    print(
                        f"{Colors.YELLOW}Status: {verification_dict['status']}{Colors.ENDC}")
                if "confidence" in verification_dict:
                    # Convert confidence to percentage if it's a decimal
                    confidence_value = verification_dict["confidence"]
                    if isinstance(confidence_value, (int, float)) and confidence_value <= 1.0:
                        confidence_value = confidence_value * 100
                    print(
                        f"{Colors.YELLOW}Confidence: {confidence_value:.0f}%{Colors.ENDC}")
            print()
        
        # Show confidence scores if available
        if "confidence" in result and result["confidence"] and "overall" in result["confidence"]:
            confidence = result["confidence"]["overall"]
            # Convert confidence to percentage if it's a decimal
            if isinstance(confidence, (int, float)) and confidence <= 1.0:
                confidence = confidence * 100
            confidence_color = Colors.RED if confidence < 60 else Colors.YELLOW if confidence < 80 else Colors.GREEN
            print(f"{confidence_color}Overall Confidence: {confidence:.0f}%{Colors.ENDC}")
        
        # Show document sources if available
        if "context_sources" in result and result["context_sources"]:
            unique_sources = set(
                source for source in result["context_sources"] if source != 'unknown')
            if unique_sources:
                print(f"{Colors.CYAN}Sources: {', '.join(unique_sources)}{Colors.ENDC}")
        
        # Add insight if available
        if "insight" in result:
            insight_text = result["insight"]
            if insight_text and isinstance(insight_text, str) and insight_text.strip():
                print(
                    f"\n{Colors.PURPLE}New Insight: {insight_text.replace(Colors.GREEN, '').replace(Colors.ENDC, '')}{Colors.ENDC}")
        
        # Add visualization note if available
        if "visualization_note" in result:
            print(result["visualization_note"])
        
        # If reasoning is available, show it automatically without prompting
        if self.reasoning_mode and "reasoning" in result and result["reasoning"]:
            print(f"\n{Colors.BLUE}Reasoning Process:{Colors.ENDC}")
            print(result["reasoning"])
        
        # Display the final answer at the bottom in green if it was found
        if final_answer:
            print(f"\n{Colors.GREEN}{final_answer}{Colors.ENDC}")

    def show_help(self):
        """Enhanced help display with Document Intelligence capabilities"""
        help_text = f"""
{Colors.GREEN}Sample questions you can ask:{Colors.ENDC}
{Colors.YELLOW}-----------------------------{Colors.ENDC}
1. "How many employees work in the Los Angeles office?"
2. "What is the badge utilization rate in Chicago compared to NYC?"
3. "Which city has the highest meeting room utilization on weekends?"
4. "What's the trend in energy usage across all offices over time?"
5. "Create a visualization of meeting room utilization by city"
6. "Compare the sustainability metrics of Chicago and Miami offices"
7. "Extract all dates from the NYC lease document"
8. "Summarize the ESG metrics across all locations"

{Colors.CYAN}Advanced Commands:{Colors.ENDC}
- "visualize [data]": Create visualization from data
- "summarize [document]": Generate summary of document
- "extract [pattern] from [text]": Extract patterns (dates, numbers, etc.)
- "compare [docs]": Compare multiple documents
- "insights": Show generated insights
- "insights on/off": Toggle automatic insights
- "reasoning on/off": Toggle display of reasoning steps
- "facts": Show facts the system has learned
- "refresh": Manually refresh document database
- "delete chroma": Delete the ChromaDB database
- "rebuild database": Delete and rebuild the entire database
- "clear": Clear conversation history
- "help": Show this help message
- "exit" or "quit": End the session

{Colors.CYAN}LangSmith Commands:{Colors.ENDC}
- "runs": View links to recent LangSmith runs
- You can rate responses 1-5 when prompted for feedback

{Colors.YELLOW}Pattern Types:{Colors.ENDC} currency, percentage, date, email, phone, time, numeric

For more sample questions, check SAMPLE_QUESTIONS.md
"""
        print(help_text)

    def show_facts(self):
        """Show facts that have been learned"""
        all_facts = self.fact_memory.facts + self.fact_memory.session_facts
        if not all_facts:
            print(f"{Colors.YELLOW}No facts have been learned yet.{Colors.ENDC}")
            return

        print(f"\n{Colors.GREEN}Learned Facts:{Colors.ENDC}")
        print(f"{Colors.YELLOW}-------------{Colors.ENDC}")

        # Limit to 20 facts for readability
        for i, fact in enumerate(all_facts[:20]):
            confidence = fact.get("confidence", 0) * 100
            confidence_color = Colors.RED if confidence < 60 else Colors.YELLOW if confidence < 80 else Colors.GREEN
            print(f"{i+1}. {fact['fact']}")
            print(f"   {Colors.CYAN}Source:{Colors.ENDC} {fact['source']}")
            print(
                f"   {confidence_color}Confidence: {confidence:.0f}%{Colors.ENDC}")

        if len(all_facts) > 20:
            print(f"\n... and {len(all_facts) - 20} more facts.")

    def save_history(self):
        """Save chat history and facts to files"""
        # Save chat history
        history_file = "chat_history.json"
        try:
            with open(history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
            print(f"Chat history saved to {history_file}")
        except Exception as e:
            print(f"Error saving chat history: {e}")

        # Save facts
        self.fact_memory.save_facts()
        print(f"Facts saved to {self.fact_memory.memory_file}")

    def run_session(self):
        """Run an interactive chat session with advanced features"""
        try:
            while True:
                # Get user input
                user_input = input(f"\n{Colors.CYAN}👤 You:{Colors.ENDC} ")
                user_input = user_input.strip()

                # Check for special commands
                if not user_input:
                    continue

                # Exit commands
                if user_input.lower() in ['exit', 'quit', 'q']:
                    self.monitor_running = False  # Stop the monitoring thread
                    self.save_history()
                    print(
                        f"{Colors.GREEN}Goodbye! Chat history and facts have been saved.{Colors.ENDC}")
                    break

                # Help command
                elif user_input.lower() == 'help':
                    self.show_help()
                    continue

                # Clear history command
                elif user_input.lower() == 'clear':
                    self.history = []
                    self.conversation_memory.clear()
                    print(f"{Colors.GREEN}Chat history cleared.{Colors.ENDC}")
                    continue

                # Facts command
                elif user_input.lower() == 'facts':
                    self.show_facts()
                    continue

                # Refresh documents command
                elif user_input.lower() == 'refresh':
                    print(
                        f"{Colors.YELLOW}Refreshing document database...{Colors.ENDC}")
                    success = self._refresh_documents()
                    if success:
                        print(
                            f"{Colors.GREEN}Document database refreshed successfully!{Colors.ENDC}")
                    else:
                        print(
                            f"{Colors.RED}Failed to refresh document database.{Colors.ENDC}")
                    continue

                # Delete Chroma database command
                elif user_input.lower() == 'delete chroma':
                    print(
                        f"{Colors.RED}WARNING: This will delete the entire ChromaDB database. Are you sure? (y/n){Colors.ENDC}")
                    confirm = input().strip().lower()
                    if confirm == 'y' or confirm == 'yes':
                        print(
                            f"{Colors.YELLOW}Deleting ChromaDB database...{Colors.ENDC}")
                        import shutil
                        try:
                            # Close any open connections
                            self.client = None
                            self.vectorstore = None
                            self.retriever = None

                            # Delete the directory
                            if os.path.exists(self.chroma_path) and os.path.isdir(self.chroma_path):
                                shutil.rmtree(self.chroma_path)
                                print(
                                    f"{Colors.GREEN}ChromaDB database deleted successfully!{Colors.ENDC}")
                            else:
                                print(
                                    f"{Colors.YELLOW}ChromaDB directory not found.{Colors.ENDC}")
                        except Exception as e:
                            print(
                                f"{Colors.RED}Error deleting ChromaDB: {e}{Colors.ENDC}")
                    else:
                        print(
                            f"{Colors.GREEN}Database deletion cancelled.{Colors.ENDC}")
                    continue

                # Rebuild database command
                elif user_input.lower() == 'rebuild database':
                    print(
                        f"{Colors.YELLOW}This will delete and rebuild the entire database. Continue? (y/n){Colors.ENDC}")
                    confirm = input().strip().lower()
                    if confirm == 'y' or confirm == 'yes':
                        print(
                            f"{Colors.YELLOW}Rebuilding database from scratch...{Colors.ENDC}")

                        # Delete ChromaDB
                        import shutil
                        try:
                            # Close any open connections
                            self.client = None
                            self.vectorstore = None
                            self.retriever = None

                            # Delete the directory if it exists
                            if os.path.exists(self.chroma_path) and os.path.isdir(self.chroma_path):
                                shutil.rmtree(self.chroma_path)
                                print(
                                    f"{Colors.GREEN}Old database deleted.{Colors.ENDC}")
                        except Exception as e:
                            print(
                                f"{Colors.RED}Error deleting database: {e}{Colors.ENDC}")
                            continue

                        # Reinitialize client
                        self.client = chromadb.PersistentClient(
                            path=self.chroma_path)

                        # Rebuild database
                        success = self._refresh_documents()
                        if success:
                            print(
                                f"{Colors.GREEN}Database rebuilt successfully!{Colors.ENDC}")
                        else:
                            print(
                                f"{Colors.RED}Failed to rebuild database.{Colors.ENDC}")
                    else:
                        print(
                            f"{Colors.GREEN}Database rebuild cancelled.{Colors.ENDC}")
                    continue

                # Toggle reasoning mode
                elif user_input.lower() == 'reasoning on':
                    self.reasoning_mode = True
                    print(f"{Colors.GREEN}Reasoning steps enabled.{Colors.ENDC}")
                    continue

                elif user_input.lower() == 'reasoning off':
                    self.reasoning_mode = False
                    print(f"{Colors.GREEN}Reasoning steps disabled.{Colors.ENDC}")
                    continue

                elif user_input.lower() == 'reasoning':
                    status = "enabled" if self.reasoning_mode else "disabled"
                    print(
                        f"{Colors.GREEN}Reasoning steps are currently {status}.{Colors.ENDC}")
                    continue

                # Process regular queries
                print(f"\n{Colors.GREEN}🤖 Assistant:{Colors.ENDC} ", end="")
                result = self.process_query(user_input)
                self.display_response(result)

                # Silent LangSmith feedback tracking (no user prompts)
                if self.langsmith_enabled and "answer" in result and not "error" in result:
                    # Track the interaction in LangSmith without asking for user feedback
                    try:
                        self.track_run(user_input, result)
                    except:
                        pass

                print(f"\n{Colors.YELLOW}" + "-"*50 + Colors.ENDC)

        except KeyboardInterrupt:
            self.monitor_running = False  # Stop the monitoring thread
            print(f"\n{Colors.YELLOW}Session ended by user.{Colors.ENDC}")
            self.save_history()
        except Exception as e:
            self.monitor_running = False  # Stop the monitoring thread
            print(f"{Colors.RED}An error occurred: {e}{Colors.ENDC}")
            logger.exception("Error in chat session")
            self.save_history()

    # Track a run in LangSmith
    def track_run(self, query, result):
        """Track a query and response in LangSmith"""
        if not self.langsmith_enabled or not self.langsmith_client:
            return None


if __name__ == "__main__":
    # Check for command line arguments
    refresh_flag = any(arg.lower() in ['--refresh', '-r']
                       for arg in sys.argv[1:])

    chat = DocumentChatV3(auto_refresh=True)
    chat.run_session()
