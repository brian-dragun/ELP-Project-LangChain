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
from config import Config

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
            
            # Use retrieval settings from Config
            search_kwargs = {
                "k": Config.RETRIEVAL_DEFAULT_K
            }
            
            # Add MMR settings if enabled
            if Config.RETRIEVAL_USE_MMR:
                search_kwargs["search_type"] = "mmr"
                search_kwargs["fetch_k"] = search_kwargs["k"] * 2  # Fetch more docs for MMR
                search_kwargs["lambda_mult"] = 1 - Config.RETRIEVAL_MMR_DIVERSITY  # Convert diversity to lambda
            
            self.retriever = self.vectorstore.as_retriever(search_kwargs=search_kwargs)
            logger.info(f"Initialized retriever with search_kwargs: {search_kwargs}")
        except Exception as e:
            logger.warning(f"Could not initialize ChromaDB: {e}")
            self.client = None
            self.vectorstore = None
            self.retriever = None
    
    def retrieve_context(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for a query with adaptive search.
        Uses different retrieval strategies based on query type.
        
        Args:
            query: The user query
            
        Returns:
            List of retrieved documents
        """
        if not self.retriever:
            logger.warning("Retriever not initialized")
            return []
        
        try:
            # Determine the query type to adjust retrieval parameters
            query_type = self._classify_query_type(query)
            logger.info(f"Query classified as {query_type}: {query}")
            
            # Configure retrieval settings based on query type
            search_kwargs = {}
            
            # Set k value based on query type
            if query_type == "factual":
                search_kwargs["k"] = Config.RETRIEVAL_FACTUAL_K
            elif query_type == "comparative":
                search_kwargs["k"] = Config.RETRIEVAL_COMPARATIVE_K
                # For comparative, we might want to fetch more but filter for diversity
                if Config.RETRIEVAL_USE_MMR:
                    search_kwargs["search_type"] = "mmr"
                    search_kwargs["fetch_k"] = Config.RETRIEVAL_COMPARATIVE_FETCH_K
                    search_kwargs["lambda_mult"] = 1 - Config.RETRIEVAL_MMR_DIVERSITY
            elif query_type == "analytical":
                search_kwargs["k"] = Config.RETRIEVAL_ANALYTICAL_K
                # For analytical, always use MMR with higher diversity
                if Config.RETRIEVAL_USE_MMR:
                    search_kwargs["search_type"] = "mmr"
                    search_kwargs["fetch_k"] = Config.RETRIEVAL_ANALYTICAL_K * 2
                    # Use slightly higher diversity for analytical queries
                    search_kwargs["lambda_mult"] = 1 - (Config.RETRIEVAL_MMR_DIVERSITY * 1.2)
            else:
                # Default case
                search_kwargs["k"] = Config.RETRIEVAL_DEFAULT_K
                if Config.RETRIEVAL_USE_MMR:
                    search_kwargs["search_type"] = "mmr"
                    search_kwargs["fetch_k"] = Config.RETRIEVAL_DEFAULT_K * 2
                    search_kwargs["lambda_mult"] = 1 - Config.RETRIEVAL_MMR_DIVERSITY
            
            # Create a temporary retriever with these settings
            temp_retriever = self.vectorstore.as_retriever(search_kwargs=search_kwargs)
            logger.info(f"Using retrieval settings for {query_type} query: {search_kwargs}")
            
            # Get documents with configured retriever
            docs = temp_retriever.invoke(query)
            
            # Special case handling for queries about all offices
            if "all office" in query.lower() or "all location" in query.lower():
                if not any("all_employee_counts" in doc.metadata.get("content_type", "") 
                          for doc in docs if hasattr(doc, "metadata")):
                    # Use default settings for this supplementary query
                    default_retriever = self.vectorstore.as_retriever(search_kwargs={"k": Config.RETRIEVAL_DEFAULT_K})
                    all_offices_docs = default_retriever.invoke("total employee count across all listed offices")
                    # Add these documents to the results if they're not already included
                    doc_ids = set(id(doc) for doc in docs)
                    for doc in all_offices_docs:
                        if id(doc) not in doc_ids:
                            docs.append(doc)
            
            return docs
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def _classify_query_type(self, query: str) -> str:
        """
        Classify the query type to determine appropriate retrieval settings.
        
        Args:
            query: The user query
            
        Returns:
            Query type: "factual", "comparative", "analytical", or "default"
        """
        query_lower = query.lower()
        
        # Factual queries ask for specific facts or data
        factual_indicators = [
            "how many", "what is the", "where is", "when was", "who is",
            "list the", "tell me the", "find the", "show me", "give me the",
            "what are the", "is there a", "does the", "do the"
        ]
        
        # Comparative queries ask to compare or contrast
        comparative_indicators = [
            "compare", "versus", "vs", "difference between", "similarities between",
            "how does", "which one", "better than", "worse than", "higher than", "lower than",
            "more than", "less than", "stronger", "weaker", "taller", "shorter"
        ]
        
        # Analytical queries ask for reasoning, analysis, or multi-step thinking
        analytical_indicators = [
            "analyze", "evaluate", "explain why", "reason for", "implications of",
            "impact of", "effect of", "cause of", "relationship between",
            "how might", "what would happen if", "why does", "what can be inferred",
            "what conclusions", "recommend", "suggest", "how should", "optimize",
            "strategy for", "plan for"
        ]
        
        # Check for indicators in the query
        if any(indicator in query_lower for indicator in analytical_indicators):
            return "analytical"
        elif any(indicator in query_lower for indicator in comparative_indicators):
            return "comparative"
        elif any(indicator in query_lower for indicator in factual_indicators):
            return "factual"
        else:
            return "default"
    
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