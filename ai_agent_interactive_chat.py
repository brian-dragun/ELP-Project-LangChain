#!/usr/bin/env python3
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
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import chromadb
from datetime import datetime
from typing import List, Dict, Any, Optional, ClassVar, Type, Union, Pattern, Tuple

# Langchain imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationSummaryMemory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.tools import BaseTool
from langchain.agents import initialize_agent, AgentType
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# LangSmith imports
from langsmith import Client
from langsmith.run_trees import RunTree
from langsmith.evaluation import StringEvaluator
from langchain.smith import RunEvalConfig, run_on_dataset

# pydantic imports
from pydantic import Field, field_validator, BaseModel

# Local imports
from ai_agent import llm, logger
from config import Config

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
            num_numerical = sum(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns)
            num_categorical = sum(not pd.api.types.is_numeric_dtype(df[col]) for col in df.columns)
            
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
                numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
                if len(numeric_cols) >= 2:
                    x_col, y_col = numeric_cols[0], numeric_cols[1]
                    df.plot(x=x_col, y=y_col, kind='line', ax=ax)
                else:
                    df.plot(kind='line', ax=ax)
                    
            elif chart_type == "bar":
                if len(df.columns) >= 2:
                    x_col = df.columns[0]
                    y_col = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
                    if y_col:
                        df.plot(x=x_col, y=y_col[0], kind='bar', ax=ax)
                    else:
                        df.plot(kind='bar', ax=ax)
                else:
                    df.plot(kind='bar', ax=ax)
                    
            elif chart_type == "pie":
                # Use first numeric column for values, first non-numeric for labels
                value_col = next((col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])), df.columns[-1])
                label_col = next((col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])), df.columns[0])
                df.plot(kind='pie', y=value_col, labels=df[label_col], autopct='%1.1f%%', ax=ax)
                
            elif chart_type == "scatter":
                numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
                if len(numeric_cols) >= 2:
                    df.plot(x=numeric_cols[0], y=numeric_cols[1], kind='scatter', ax=ax)
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
            source = doc.metadata.get('source', f"Document {i+1}") if hasattr(doc, 'metadata') else f"Document {i+1}"
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
            source = doc.metadata.get('source', f"Document {i+1}") if hasattr(doc, 'metadata') else f"Document {i+1}"
            doc_contents.append((source, content))
        
        # Create prompt
        connections_prompt = f"Specifically look for connections related to: {query}" if query else ""
        
        # Format document contents
        formatted_docs = ""
        for source, content in doc_contents:
            formatted_docs += f"\n\n--- {source} ---\n{content[:1000]}..."  # Limit content length
            
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
        self.documents_path = os.path.join(os.path.dirname(__file__), 'documents')
        self.chroma_path = os.path.join(os.path.dirname(__file__), 'chroma_db')
        self.hash_file = os.path.join(os.path.dirname(__file__), 'documents_hash.txt')
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

    # [File monitoring methods remain the same]
    def _start_document_monitor(self):
        """Start a background thread to monitor document changes"""
        self.monitor_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_documents, daemon=True)
        self.monitor_thread.start()
        
    def _monitor_documents(self):
        """Monitor the documents folder for changes in background"""
        check_interval = 10  # Check every 10 seconds
        
        while self.monitor_running:
            time.sleep(check_interval)
            
            # Check if documents have changed
            if self._documents_changed():
                current_time = time.time()
                # Only refresh if it's been at least 60 seconds since last refresh
                if current_time - self.last_check_time > 60:
                    print(f"\n{Colors.YELLOW}New documents detected! Refreshing database...{Colors.ENDC}")
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

    def _refresh_documents(self):
        """Re-ingest documents only if needed"""
        print("Checking document database...")
        
        try:
            # First, check if the vector store already exists and contains documents
            if os.path.exists(self.chroma_path) and os.path.isdir(self.chroma_path):
                try:
                    # Check if we have a valid collection with documents
                    collection = self.client.get_collection("document_collection")
                    count = collection.count()
                    
                    # If we have a good number of documents and the hash hasn't changed
                    if count > 0 and not self._documents_changed():
                        print(f"{Colors.GREEN}Document database is up to date with {count} documents.{Colors.ENDC}")
                        
                        # Update class variables without reloading
                        self.vectorstore = Chroma(
                            client=self.client,
                            collection_name="document_collection",
                            embedding_function=self.embeddings
                        )
                        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
                        
                        # Make sure the document hash is saved
                        self._save_documents_hash()
                        
                        # Update tools with existing retriever
                        for tool in self.tools:
                            if hasattr(tool, 'retriever'):
                                tool.retriever = self.retriever
                        
                        return True
                except Exception as e:
                    # If there's an issue checking the collection, we'll recreate it
                    print(f"{Colors.YELLOW}Could not verify existing database: {e}{Colors.ENDC}")
            
            # If we get here, we need to refresh the database
            print(f"{Colors.YELLOW}Refreshing document database...{Colors.ENDC}")
            
            # Use our improved ingest_documents function from the module
            try:
                # Import the ingest function dynamically
                sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                from ingest_documents import ingest_documents
                
                # Call the ingest function with our documents path
                num_docs = ingest_documents(self.documents_path)
                print(f"Loaded {num_docs} documents")
                
                # Update vector store and retriever references
                self.vectorstore = Chroma(
                    client=self.client,
                    collection_name="document_collection",
                    embedding_function=self.embeddings
                )
                self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
                
                # Update tools with new retriever
                for tool in self.tools:
                    if hasattr(tool, 'retriever'):
                        tool.retriever = self.retriever
                
                # Save the new document hash
                self._save_documents_hash()
                
                print(f"{Colors.GREEN}Document refresh complete!{Colors.ENDC}")
                return True
                
            except ImportError:
                print(f"{Colors.YELLOW}Could not import ingest_documents module, falling back to built-in ingestion{Colors.ENDC}")
                # Fall back to original document loading if import fails
                
                # Configure document loading (with more file types)
                loader = DirectoryLoader(
                    self.documents_path,
                    glob="**/*.*",  # Load all file types
                    silent_errors=True
                )
                
                try:
                    documents = loader.load()
                    print(f"Loaded {len(documents)} documents")
                    
                    # Split documents into chunks
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                    splits = text_splitter.split_documents(documents)
                    print(f"Split into {len(splits)} chunks")
                    
                    # Special processing for employee data in CSV files
                    enhanced_splits = []
                    for doc in splits:
                        enhanced_splits.append(doc)
                        
                        # Check if the document contains employee count data
                        content = doc.page_content.lower()
                        if ("numberofemployees" in content or "employee_count" in content or 
                            "employees" in content or "headcount" in content):
                            
                            # Try to extract the city and employee count
                            city_match = re.search(r'city:?\s*([a-zA-Z\s]+)', content, re.IGNORECASE)
                            employee_match = re.search(r'(?:numberofemployees|employee_count|employees|headcount):?\s*(\d+)', content, re.IGNORECASE)
                            
                            if city_match and employee_match:
                                city = city_match.group(1).strip()
                                employee_count = employee_match.group(1).strip()
                                
                                # Create a specialized document for this employee count
                                emp_content = f"The {city} office has {employee_count} employees."
                                emp_doc = Document(
                                    page_content=emp_content,
                                    metadata={
                                        "source": doc.metadata.get("source", "unknown"),
                                        "file_type": "csv",
                                        "content_type": "employee_count",
                                        "city": city,
                                        "employee_count": employee_count
                                    }
                                )
                                enhanced_splits.append(emp_doc)
                                print(f"Created specific employee count document for {city}: {employee_count} employees")
                    
                    # Remove existing collection if it exists
                    try:
                        self.client.delete_collection(name="document_collection")
                    except:
                        pass
                    
                    # Create vector store with new documents
                    vectorstore = Chroma.from_documents(
                        documents=enhanced_splits,
                        embedding=self.embeddings,
                        client=self.client,
                        collection_name="document_collection"
                    )
                    
                    # Update vector store and retriever references
                    self.vectorstore = vectorstore
                    self.retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                    
                    # Update tools with new retriever
                    for tool in self.tools:
                        if hasattr(tool, 'retriever'):
                            tool.retriever = self.retriever
                    
                    # Save the new document hash
                    self._save_documents_hash()
                    
                    print(f"{Colors.GREEN}Document refresh complete!{Colors.ENDC}")
                    return True
                except Exception as e:
                    print(f"{Colors.RED}Error loading documents: {e}{Colors.ENDC}")
                    raise
            
        except Exception as e:
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
        fact_tool = next((tool for tool in self.tools if tool.name == "fact_extraction"), None)
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
                
                fact = self.fact_memory.add_fact(fact_text.strip(), source.strip(), confidence)
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
        verify_tool = next((tool for tool in self.tools if tool.name == "fact_verification"), None)
        if verify_tool:
            verification = verify_tool._run(response)
            return verification
        
        return "Unable to verify response."

    def add_reasoning_steps(self, query, context_docs):
        """Generate explicit reasoning steps for answering the query"""
        context_str = "\n\n".join([doc.page_content for doc in context_docs[:2]])  # Limit context for reasoning
        
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
        """Process a user query and return an answer with reasoning, verification, and confidence"""
        try:
            # Step 1: Generate self-questions for clarification
            self_questioning_tool = next((tool for tool in self.tools if tool.name == "self_questioning"), None)
            self_questions = self_questioning_tool._run(query) if self_questioning_tool else ""
            
            # Step 2: Retrieve relevant documents
            context_docs = self.get_relevant_context(query)
            
            # Step 3: Extract and store facts from context
            facts = self.extract_facts_from_context(query, context_docs)
            
            # Step 4: Generate reasoning steps (optional)
            reasoning = self.add_reasoning_steps(query, context_docs) if self.reasoning_mode else ""
            
            # Step 5: Format prompt with all components
            formatted_prompt = self.format_prompt_with_history(query, context_docs, self_questions, facts)
            
            # Step 6: Generate response
            response = llm.invoke(formatted_prompt)
            
            # Step 7: Verify response against documents
            verification = self.verify_response(response, context_docs)
            
            # Step 8: Analyze confidence levels
            context_str = "\n\n".join([doc.page_content for doc in context_docs])
            confidence_analysis = self.confidence_analyzer.analyze_response(response, query, context_str)
            confidence_scores = self.confidence_analyzer.extract_confidence_scores(confidence_analysis)
            
            # Step 9: Update conversation memory
            self.conversation_memory.save_context({"input": query}, {"output": response})
            self.summary_memory.save_context({"input": query}, {"output": response})
            self.history.append((query, response))
            
            # Prepare the final result with all components
            result = {
                "answer": response,
                "self_questions": self_questions if self_questions else None,
                "reasoning": reasoning if self.reasoning_mode else None,
                "verification": verification,
                "confidence": confidence_scores,
                "context_sources": [doc.metadata.get('source', 'unknown') if hasattr(doc, 'metadata') else 'unknown' for doc in context_docs]
            }
            
            return result
            
        except Exception as e:
            logger.exception("Error processing query")
            return {
                "answer": f"Error processing your question: {str(e)}",
                "error": str(e)
            }

    def display_response(self, result):
        """Display the response with formatting for different components"""
        if "error" in result:
            print(f"{Colors.RED}{result['answer']}{Colors.ENDC}")
            return
        
        # Display the main answer
        print(f"{result['answer']}\n")
        
        # Show verification if available
        if "verification" in result and result["verification"]:
            print(f"{Colors.YELLOW}Verification:{Colors.ENDC}")
            verification_lines = result["verification"].split("\n")
            for line in verification_lines[:5]:  # Limit to first 5 lines for brevity
                print(f"{Colors.YELLOW}{line}{Colors.ENDC}")
            print()
        
        # Show confidence scores if available
        if "confidence" in result and result["confidence"] and "overall" in result["confidence"]:
            confidence = result["confidence"]["overall"]
            confidence_color = Colors.RED if confidence < 60 else Colors.YELLOW if confidence < 80 else Colors.GREEN
            print(f"{confidence_color}Overall Confidence: {confidence:.0f}%{Colors.ENDC}")
            
        # Show document sources if available
        if "context_sources" in result and result["context_sources"]:
            unique_sources = set(source for source in result["context_sources"] if source != 'unknown')
            if unique_sources:
                print(f"{Colors.CYAN}Sources: {', '.join(unique_sources)}{Colors.ENDC}")
        
        # Show reasoning steps if enabled and available
        if self.reasoning_mode and "reasoning" in result and result["reasoning"]:
            print(f"\n{Colors.BLUE}Reasoning Steps (hidden by default, 'reasoning on' to show){Colors.ENDC}")

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
        
        for i, fact in enumerate(all_facts[:20]):  # Limit to 20 facts for readability
            confidence = fact.get("confidence", 0) * 100
            confidence_color = Colors.RED if confidence < 60 else Colors.YELLOW if confidence < 80 else Colors.GREEN
            print(f"{i+1}. {fact['fact']}")
            print(f"   {Colors.CYAN}Source:{Colors.ENDC} {fact['source']}")
            print(f"   {confidence_color}Confidence: {confidence:.0f}%{Colors.ENDC}")
        
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
                    print(f"{Colors.GREEN}Goodbye! Chat history and facts have been saved.{Colors.ENDC}")
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
                    print(f"{Colors.YELLOW}Refreshing document database...{Colors.ENDC}")
                    success = self._refresh_documents()
                    if success:
                        print(f"{Colors.GREEN}Document database refreshed successfully!{Colors.ENDC}")
                    else:
                        print(f"{Colors.RED}Failed to refresh document database.{Colors.ENDC}")
                    continue
                
                # Delete Chroma database command
                elif user_input.lower() == 'delete chroma':
                    print(f"{Colors.RED}WARNING: This will delete the entire ChromaDB database. Are you sure? (y/n){Colors.ENDC}")
                    confirm = input().strip().lower()
                    if confirm == 'y' or confirm == 'yes':
                        print(f"{Colors.YELLOW}Deleting ChromaDB database...{Colors.ENDC}")
                        import shutil
                        try:
                            # Close any open connections
                            self.client = None
                            self.vectorstore = None
                            self.retriever = None
                            
                            # Delete the directory
                            if os.path.exists(self.chroma_path) and os.path.isdir(self.chroma_path):
                                shutil.rmtree(self.chroma_path)
                                print(f"{Colors.GREEN}ChromaDB database deleted successfully!{Colors.ENDC}")
                            else:
                                print(f"{Colors.YELLOW}ChromaDB directory not found.{Colors.ENDC}")
                        except Exception as e:
                            print(f"{Colors.RED}Error deleting ChromaDB: {e}{Colors.ENDC}")
                    else:
                        print(f"{Colors.GREEN}Database deletion cancelled.{Colors.ENDC}")
                    continue
                
                # Rebuild database command
                elif user_input.lower() == 'rebuild database':
                    print(f"{Colors.YELLOW}This will delete and rebuild the entire database. Continue? (y/n){Colors.ENDC}")
                    confirm = input().strip().lower()
                    if confirm == 'y' or confirm == 'yes':
                        print(f"{Colors.YELLOW}Rebuilding database from scratch...{Colors.ENDC}")
                        
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
                                print(f"{Colors.GREEN}Old database deleted.{Colors.ENDC}")
                        except Exception as e:
                            print(f"{Colors.RED}Error deleting database: {e}{Colors.ENDC}")
                            continue
                        
                        # Reinitialize client
                        self.client = chromadb.PersistentClient(path=self.chroma_path)
                        
                        # Rebuild database
                        success = self._refresh_documents()
                        if success:
                            print(f"{Colors.GREEN}Database rebuilt successfully!{Colors.ENDC}")
                        else:
                            print(f"{Colors.RED}Failed to rebuild database.{Colors.ENDC}")
                    else:
                        print(f"{Colors.GREEN}Database rebuild cancelled.{Colors.ENDC}")
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
                    print(f"{Colors.GREEN}Reasoning steps are currently {status}.{Colors.ENDC}")
                    continue
                
                # Process regular queries
                print(f"\n{Colors.GREEN}🤖 Assistant:{Colors.ENDC} ", end="")
                result = self.process_query(user_input)
                self.display_response(result)
                
                # Add feedback collection feature
                if self.langsmith_enabled:
                    if "answer" in result and not "error" in result:
                        try:
                            rating = input(f"\n{Colors.CYAN}Rate this response (1-5, or skip): {Colors.ENDC}")
                            if rating.isdigit() and 1 <= int(rating) <= 5:
                                self.get_feedback(user_input, result["answer"], int(rating))
                                print(f"{Colors.GREEN}Thank you for your feedback!{Colors.ENDC}")
                        except:
                            pass
                
                # Show reasoning if enabled and available
                if self.reasoning_mode and "reasoning" in result and result["reasoning"]:
                    should_show = input(f"\n{Colors.CYAN}Show reasoning steps? (y/n): {Colors.ENDC}")
                    if should_show.lower() == "y":
                        print(f"\n{Colors.YELLOW}Reasoning Process:{Colors.ENDC}")
                        print(result["reasoning"])
                
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
        print(f"{Colors.BOLD}🤖 Document-Aware Chat System v3 with Document Intelligence{Colors.ENDC}")
        print(f"{Colors.HEADER}" + "="*60 + Colors.ENDC)
        print("Ask questions about your office data across Chicago, LA, Miami, NYC, and Philadelphia.")
        print(f"{Colors.CYAN}Enhanced with: Memory, Reasoning, Self-Questioning, Fact Checking, and Confidence Scoring{Colors.ENDC}")
        print(f"{Colors.GREEN}NEW Document Intelligence Features:{Colors.ENDC}")
        print(f"  {Colors.YELLOW}✓{Colors.ENDC} Automatic Data Insights")
        print(f"  {Colors.YELLOW}✓{Colors.ENDC} Data Visualization Generation")
        print(f"  {Colors.YELLOW}✓{Colors.ENDC} Cross-Document Analysis")
        print(f"  {Colors.YELLOW}✓{Colors.ENDC} Auto-Summarization")
        print(f"  {Colors.YELLOW}✓{Colors.ENDC} Regular Expression Pattern Extraction")
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

    # Track a run in LangSmith
    def track_run(self, query, result):
        """Track a query and response in LangSmith"""
        if not self.langsmith_enabled or not self.langsmith_client:
            return None
        
        try:
            # Create a run tree to track the conversation
            run_tree = RunTree(
                name="document_chat_interaction",
                serialized={
                    "name": "document_chat_interaction",
                    "run_type": "chain",
                    "inputs": {"query": query},
                    "outputs": {"result": result}
                },
                client=self.langsmith_client,
                project_name=Config.LANGSMITH_PROJECT
            )
            
            # Execute the run
            run_id = run_tree.end()
            self.run_ids.append(run_id)
            return run_id
        except Exception as e:
            logger.exception(f"Error tracking run in LangSmith: {str(e)}")
            return None
    
    # Method to get conversation feedback
    def get_feedback(self, user_query, assistant_response, user_rating):
        """Record user feedback on a conversation turn"""
        if not self.langsmith_enabled or not self.langsmith_client:
            return
        
        try:
            # Try to find the most recent run ID
            if self.run_ids:
                run_id = self.run_ids[-1]
                self.langsmith_client.create_feedback(
                    run_id=run_id,
                    key="user_rating",
                    value=user_rating,  # 1-5 scale or thumbs up/down
                    comment=f"Query: {user_query}"
                )
                logger.info(f"Added feedback for run {run_id}: {user_rating}")
        except Exception as e:
            logger.exception(f"Error adding feedback: {str(e)}")
    
    # Override process_query from parent class to handle special commands
    def process_query(self, query):
        """Enhanced query processing with document intelligence and LangSmith tracking"""
        try:
            # Process special commands
            if query.lower() == 'runs':
                if self.langsmith_enabled and self.run_ids:
                    runs_info = {"answer": f"\nRecent LangSmith Runs:\n"}
                    
                    run_links = []
                    for i, run_id in enumerate(self.run_ids[-5:]):  # Show last 5 runs
                        run_links.append(f"{i+1}. {Config.LANGSMITH_ENDPOINT}/runs/{run_id}")
                    
                    runs_info["answer"] += "\n".join(run_links)
                    return runs_info
                else:
                    return {"answer": "No LangSmith runs available or LangSmith is disabled."}
                    
            elif query.lower().startswith("insights") and query.lower() != "insights on" and query.lower() != "insights off":
                # Handle the insights command - show generated insights
                if not self.insights:
                    return {"answer": "No insights have been generated yet. Ask some questions about the data first!"}
                
                insights_response = {"answer": f"\n{Colors.GREEN}Generated Insights:{Colors.ENDC}\n"}
                for i, insight in enumerate(self.insights[:5]):  # Show top 5 insights
                    insights_response["answer"] += f"\n{i+1}. {insight['description']}"
                    insights_response["answer"] += f"\n   {Colors.YELLOW}Confidence:{Colors.ENDC} {insight['confidence']}%"
                    insights_response["answer"] += f"\n   {Colors.CYAN}Suggested Action:{Colors.ENDC} {insight['action']}\n"
                
                return insights_response
                
            elif query.lower() == "insights on":
                self.auto_insights_mode = True
                return {"answer": "Automatic insights generation enabled."}
                
            elif query.lower() == "insights off":
                self.auto_insights_mode = False
                return {"answer": "Automatic insights generation disabled."}
                
            elif query.lower().startswith("visualize "):
                # Handle visualization requests
                data_text = query[10:]  # Remove "visualize " prefix
                base64_img, viz_path = self.data_visualizer.generate_visualization(data_text)
                
                if base64_img:
                    self.charts_generated.append(viz_path)
                    return {
                        "answer": f"Visualization created and saved to {viz_path}",
                        "visualization": base64_img
                    }
                else:
                    return {"answer": f"Failed to create visualization: {viz_path}"}
                    
            elif query.lower().startswith("summarize "):
                # Handle document summarization
                doc_name = query[10:]  # Remove "summarize " prefix
                # Find the document in context
                context_docs = self.get_relevant_context(doc_name)
                if not context_docs:
                    return {"answer": f"Could not find a document matching '{doc_name}'"}
                
                # Get the most relevant document
                doc_text = context_docs[0].page_content
                summary = self.document_summarizer.summarize_document(doc_text)
                return {"answer": f"Summary of '{doc_name}':\n\n{summary}"}
                
            elif query.lower().startswith("extract "):
                # Handle pattern extraction
                command = query[8:].strip()  # Remove "extract " prefix
                
                # Parse the command to get pattern type and text
                if " from " in command:
                    pattern_type, text = command.split(" from ", 1)
                    pattern_type = pattern_type.strip().lower()
                    
                    # Get relevant documents if needed
                    if not text or text.isspace():
                        context_docs = self.get_relevant_context(pattern_type)
                        text = "\n\n".join([doc.page_content for doc in context_docs])
                    
                    # Do the extraction
                    if pattern_type in self.pattern_extractor.patterns:
                        matches = self.pattern_extractor.extract_pattern(text, pattern_type)
                        if matches:
                            result = {"answer": f"Extracted {pattern_type}s:\n"}
                            for i, match in enumerate(matches):
                                result["answer"] += f"{i+1}. {match}\n"
                            return result
                        else:
                            return {"answer": f"No {pattern_type} patterns found in the text."}
                    else:
                        return {"answer": f"Unknown pattern type: {pattern_type}. Available patterns: {', '.join(self.pattern_extractor.patterns.keys())}"}
                else:
                    return {"answer": "Invalid extract command format. Use 'extract [pattern] from [text]'"}
                    
            elif query.lower().startswith("compare "):
                # Handle document comparison
                doc_names = query[8:].strip()  # Remove "compare " prefix
                doc_list = [name.strip() for name in doc_names.split(',')]
                
                all_docs = []
                for doc_name in doc_list:
                    context_docs = self.get_relevant_context(doc_name)
                    if context_docs:
                        all_docs.extend(context_docs)
                
                if len(all_docs) < 2:
                    return {"answer": "Could not find enough documents to compare. Please specify document names separated by commas."}
                
                comparison = self.cross_doc_analyzer.compare_documents(all_docs)
                return {"answer": comparison["comparison"]}
            
            # For regular queries, use the parent implementation but with added features
            result = super().process_query(query)
            
            if "error" in result:
                return result
            
            # Get context documents for additional features
            context_docs = self.get_relevant_context(query)
            
            # Add automatic insights if enabled
            if self.auto_insights_mode:
                new_insights = self.insight_generator.generate_insights(context_docs, top_k=1)
                if new_insights:
                    self.insights.extend(new_insights)
                    # Add insight preview to response
                    insight = new_insights[0]
                    result["insight"] = f"\n{Colors.GREEN}New Insight:{Colors.ENDC} {insight['description']}"
            
            # Check for visualization opportunity
            should_visualize = self.detect_visualization_opportunity(query, context_docs)
            
            if should_visualize and self.visualization_mode:
                # Extract tabular data from context
                context_text = "\n\n".join([doc.page_content for doc in context_docs])
                
                # Check if we have enough data to visualize
                has_data = bool(re.search(r"(\d+[,.]?)+", context_text)) and len(context_text) > 100
                
                if has_data:
                    # Try to generate a visualization
                    try:
                        base64_img, viz_path = self.data_visualizer.generate_visualization(
                            context_text, 
                            title=f"Visualization for: {query[:30]}..."
                        )
                        
                        if base64_img:
                            self.charts_generated.append(viz_path)
                            result["visualization_note"] = f"\n{Colors.CYAN}A visualization was automatically generated at: {viz_path}{Colors.ENDC}"
                            result["visualization"] = base64_img
                    except Exception as e:
                        logger.exception(f"Error generating visualization: {str(e)}")
            
            # Track the run in LangSmith
            if self.langsmith_enabled:
                run_id = self.track_run(query, result)
                if run_id:
                    result["langsmith_run_id"] = run_id
            
            return result
            
        except Exception as e:
            logger.exception("Error in enhanced query processing")
            return {
                "answer": f"Error processing your question: {str(e)}",
                "error": str(e)
            }

    def detect_visualization_opportunity(self, query, context_docs):
        """Determine if a query would benefit from visualization"""
        # Skip if visualization mode is disabled
        if not hasattr(self, 'visualization_mode') or not self.visualization_mode:
            return False
            
        # Simple keyword detection
        viz_keywords = ['chart', 'graph', 'plot', 'visualize', 'visualization', 'trend', 
                       'compare', 'distribution', 'percentage', 'proportion']
        
        # Check for keywords in the query
        if any(keyword in query.lower() for keyword in viz_keywords):
            return True
            
        # Check if query is asking for numerical comparisons
        comparison_patterns = ['compare', 'difference between', 'versus', 'vs', 
                              'higher', 'lower', 'most', 'least', 'ranking']
        if any(pattern in query.lower() for pattern in comparison_patterns):
            # Check if we have numerical data in the context
            context_text = "\n".join([doc.page_content for doc in context_docs])
            # Simple heuristic: If we have multiple numbers and comparison keywords, visualization might help
            numbers = re.findall(r'\d+(?:\.\d+)?%?', context_text)
            if len(numbers) > 5:  # Arbitrary threshold
                return True
                
        return False

if __name__ == "__main__":
    # Check for command line arguments
    refresh_flag = any(arg.lower() in ['--refresh', '-r'] for arg in sys.argv[1:])
    
    chat = DocumentChatV3(auto_refresh=True)
    chat.run_session()