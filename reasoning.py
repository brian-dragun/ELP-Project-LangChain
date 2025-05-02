#!/usr/bin/env python3
"""
Simplified Reasoning System for document intelligence.
This version relies more on the AI's inherent capabilities with less template scaffolding.
"""

import logging
import re
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from langchain_chroma import Chroma
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReasoningService:
    """
    Simplified reasoning service that relies more on the AI model's inherent capabilities
    while providing minimal scaffolding.
    """
    
    def __init__(self, llm=None, chroma_path="./chroma_db"):
        """
        Initialize the reasoning service.
        
        Args:
            llm: The language model to use for reasoning
            chroma_path: Path to the ChromaDB database
        """
        self.llm = llm
        self.chroma_path = chroma_path
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Initialize vectorstore if ChromaDB exists
        try:
            self.client = chromadb.PersistentClient(path=chroma_path)
            self.vectorstore = Chroma(
                client=self.client,
                collection_name="document_collection",
                embedding_function=self.embeddings
            )
            # Use a higher k value to ensure we retrieve enough context
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 15})
        except Exception as e:
            logger.warning(f"Could not initialize ChromaDB: {e}")
            self.client = None
            self.vectorstore = None
            self.retriever = None
    
    def retrieve_context(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for a query with adaptive search.
        
        Args:
            query: The user query
            
        Returns:
            List of retrieved documents
        """
        if not self.retriever:
            logger.warning("Retriever not initialized")
            return []
        
        try:
            # First try to get specific matches with the exact query
            docs = self.retriever.invoke(query)
            
            # If query contains "all offices" but we didn't find the all-offices document,
            # try a more explicit query to find it
            if "all office" in query.lower() or "all location" in query.lower():
                if not any("all_employee_counts" in doc.metadata.get("content_type", "") 
                          for doc in docs if hasattr(doc, "metadata")):
                    # Try a more explicit query to find the all-offices document
                    all_offices_docs = self.retriever.invoke("total employee count across all listed offices")
                    # Add these documents to the results if they're not already included
                    doc_ids = set(id(doc) for doc in docs)
                    for doc in all_offices_docs:
                        if id(doc) not in doc_ids:
                            docs.append(doc)
            
            return docs
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def process_query(self, query: str, detail_level: str = 'standard') -> Dict[str, Any]:
        """
        Process a user query with minimal scaffolding, relying more on the AI's capabilities.
        
        Args:
            query: The user query
            detail_level: Level of detail for reasoning ('basic', 'standard', or 'detailed')
            
        Returns:
            Dictionary with response and metadata
        """
        try:
            # Step 1: Retrieve relevant documents with wider search
            context_docs = self.retrieve_context(query)
            
            if not context_docs:
                return {
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "context_sources": []
                }
            
            # Extract document sources for tracking
            context_sources = [
                doc.metadata.get('source', 'unknown') 
                if hasattr(doc, 'metadata') else 'unknown' 
                for doc in context_docs
            ]
            
            # Format the context
            context_text = "\n\n".join([doc.page_content for doc in context_docs])
            
            # Step 2: Use a single unified prompt that relies on the AI's reasoning capabilities
            reasoning_level = {
                'basic': "Provide a brief, direct answer with minimal explanation.",
                'standard': "Include step-by-step reasoning and explain your thought process.",
                'detailed': "Provide detailed reasoning with multiple steps, addressing potential assumptions and limitations."
            }.get(detail_level, "Include step-by-step reasoning and explain your thought process.")
            
            # Unified prompt that handles both answer generation and reasoning
            unified_prompt = f"""Answer the following question based on the provided context.

Question: {query}

Context:
{context_text}

{reasoning_level}

If you can't find a direct answer in the context, use your reasoning to provide the best possible answer based on what's available.
If there's truly not enough information to answer, say so clearly.

Begin by thinking through the problem step by step, then provide your final answer.
"""
            
            # Generate response
            response = self.llm.invoke(unified_prompt)
            
            # Extract the final answer and reasoning
            # Modern LLMs will typically structure their response with reasoning followed by answer
            reasoning, answer = self._extract_reasoning_and_answer(response)
            
            # Perform confidence analysis
            confidence = self._analyze_confidence(response, context_docs)
            
            result = {
                "answer": answer,
                "reasoning": reasoning,
                "confidence": confidence,
                "context_sources": context_sources
            }
            
            return result
            
        except Exception as e:
            logger.exception("Error processing query")
            return {
                "answer": f"Error processing your question: {str(e)}",
                "error": str(e)
            }
    
    def _extract_reasoning_and_answer(self, response: str) -> tuple:
        """
        Extract reasoning and answer from the AI's response.
        Modern LLMs typically structure their response with reasoning followed by a conclusion.
        
        Args:
            response: The full response from the LLM
            
        Returns:
            Tuple of (reasoning, answer)
        """
        # Check for common patterns indicating the final answer
        final_answer_patterns = [
            r"(The final answer is:)(.*?)($|\.)",
            r"(In conclusion,)(.*?)($|\.)",
            r"(Therefore,)(.*?)($|\.)",
            r"(To summarize,)(.*?)($|\.)"
        ]
        
        for pattern in final_answer_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                answer = match.group(2).strip()
                reasoning = response.replace(match.group(0), "").strip()
                return reasoning, answer
        
        # If no pattern matches, use the last paragraph as the answer
        paragraphs = response.split("\n\n")
        if len(paragraphs) > 1:
            answer = paragraphs[-1].strip()
            reasoning = "\n\n".join(paragraphs[:-1]).strip()
            return reasoning, answer
        
        # Fall back to returning the whole response as the answer
        return "", response.strip()
    
    def _analyze_confidence(self, response: str, context_docs: List[Document]) -> Dict[str, Any]:
        """
        Perform a simple confidence analysis based on response and context.
        
        Args:
            response: The generated response
            context_docs: The context documents
            
        Returns:
            Dictionary with confidence scores
        """
        # Check for hedging language that indicates uncertainty
        uncertainty_phrases = [
            "I don't have enough information",
            "cannot be determined",
            "unclear",
            "insufficient data",
            "not mentioned",
            "might be",
            "could be",
            "possibly",
            "uncertain"
        ]
        
        uncertainty_count = sum(1 for phrase in uncertainty_phrases if phrase.lower() in response.lower())
        
        # Calculate base confidence score
        base_confidence = 1.0 - (uncertainty_count * 0.1)
        
        # Adjust for context relevance - check if key terms from the response appear in the context
        response_words = set(word.lower() for word in re.findall(r'\b\w+\b', response) 
                            if len(word) > 3 and word.lower() not in ['this', 'that', 'with', 'from'])
        
        context_text = " ".join([doc.page_content for doc in context_docs])
        context_words = set(word.lower() for word in re.findall(r'\b\w+\b', context_text) 
                           if len(word) > 3)
        
        word_overlap = len(response_words.intersection(context_words)) / max(1, len(response_words))
        
        # Calculate final confidence score
        confidence = (base_confidence * 0.7) + (word_overlap * 0.3)
        confidence = min(max(confidence, 0.1), 0.95)  # Bound between 0.1 and 0.95
        
        return {
            "overall": confidence,
            "has_uncertainty": uncertainty_count > 0
        }