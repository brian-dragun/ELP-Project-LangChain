#!/usr/bin/env python3
import os
import sys
import json
import glob
import hashlib
import time
import threading
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, ClassVar, Type, Union

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
from qa_system import llm, logger, prompt
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import Field, field_validator, BaseModel

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
        
        # Welcome message
        print(f"\n{Colors.HEADER}" + "="*50 + Colors.ENDC)
        print(f"{Colors.BOLD}🤖 Document-Aware Chat System v2{Colors.ENDC}")
        print(f"{Colors.HEADER}" + "="*50 + Colors.ENDC)
        print("Ask questions about your office data across Chicago, LA, Miami, NYC, and Philadelphia.")
        print(f"{Colors.CYAN}Enhanced with: Memory, Reasoning, Self-Questioning, Fact Checking, and Confidence Scoring{Colors.ENDC}")
        print(f"{Colors.YELLOW}Commands:{Colors.ENDC}")
        print("  exit/quit - End the session")
        print("  help - Show sample questions")
        print("  clear - Clear chat history")
        print("  refresh - Manually refresh document database")
        print("  facts - Show learned facts")
        print("  reasoning on/off - Toggle reasoning steps visibility")
        print(f"{Colors.HEADER}" + "="*50 + Colors.ENDC + "\n")

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
        """Re-ingest all documents in the documents folder"""
        print("Refreshing document database...")
        
        try:
            # Configure document loading
            loader = DirectoryLoader(
                self.documents_path,
                glob="**/*.csv"
            )
            documents = loader.load()
            print(f"Loaded {len(documents)} documents")
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)
            print(f"Split into {len(splits)} chunks")
            
            # Remove existing collection if it exists
            try:
                self.client.delete_collection(name="document_collection")
            except:
                pass
            
            # Create vector store with new documents
            vectorstore = Chroma.from_documents(
                documents=splits,
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
        """Show sample questions the user can ask"""
        help_text = f"""
{Colors.GREEN}Sample questions you can ask:{Colors.ENDC}
{Colors.YELLOW}-----------------------------{Colors.ENDC}
1. "How many employees work in the Los Angeles office?"
2. "What is the badge utilization rate in Chicago compared to NYC?"
3. "Which city has the highest meeting room utilization on weekends?"
4. "What is the LEED certification level for the Chicago office?"
5. "When does the Los Angeles office lease end?"
6. "Based on current trends, what will the meeting room utilization in NYC be by the end of 2025?"
7. "Which office has the best balance of sustainability and financial efficiency?"
8. "How does our office space cost in LA compare to predicted market rates for 2026?"

{Colors.CYAN}Advanced Commands:{Colors.ENDC}
- "reasoning on" or "reasoning off": Toggle display of reasoning steps
- "facts": Show facts the system has learned
- "refresh": Manually refresh document database
- "clear": Clear conversation history
- "help": Show this help message
- "exit" or "quit": End the session

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
                    self.summary_memory.clear()
                    print(f"{Colors.YELLOW}Chat history cleared.{Colors.ENDC}")
                    continue
                
                # Show facts command
                elif user_input.lower() == 'facts':
                    self.show_facts()
                    continue
                
                # Refresh command
                elif user_input.lower() == 'refresh':
                    print(f"{Colors.YELLOW}Manually refreshing document database...{Colors.ENDC}")
                    self._refresh_documents()
                    continue
                
                # Toggle reasoning mode
                elif user_input.lower() == 'reasoning on':
                    self.reasoning_mode = True
                    print(f"{Colors.GREEN}Reasoning steps display enabled.{Colors.ENDC}")
                    continue
                elif user_input.lower() == 'reasoning off':
                    self.reasoning_mode = False
                    print(f"{Colors.YELLOW}Reasoning steps display disabled.{Colors.ENDC}")
                    continue
                elif user_input.lower() == 'reasoning':
                    self.reasoning_mode = not self.reasoning_mode
                    status = "enabled" if self.reasoning_mode else "disabled"
                    print(f"{Colors.GREEN if self.reasoning_mode else Colors.YELLOW}Reasoning steps display {status}.{Colors.ENDC}")
                    continue
                
                # Process regular queries
                print(f"\n{Colors.GREEN}🤖 Assistant:{Colors.ENDC} ", end="")
                result = self.process_query(user_input)
                self.display_response(result)
                
                # Show reasoning if enabled and available
                if self.reasoning_mode and "reasoning" in result and result["reasoning"]:
                    print(f"\n{Colors.BLUE}Reasoning Process:{Colors.ENDC}")
                    reasoning_lines = result["reasoning"].split('\n')
                    for line in reasoning_lines:
                        if line.strip().startswith("STEP") or "step" in line.lower()[:10]:
                            print(f"{Colors.YELLOW}{line}{Colors.ENDC}")
                        else:
                            print(f"{Colors.BLUE}{line}{Colors.ENDC}")
                
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

if __name__ == "__main__":
    # Check for command line arguments
    refresh_flag = any(arg.lower() in ['--refresh', '-r'] for arg in sys.argv[1:])
    
    chat = DocumentChatV2(auto_refresh=True)
    chat.run_session()