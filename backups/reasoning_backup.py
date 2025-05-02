#!/usr/bin/env python3
"""
Unified Reasoning System for document intelligence.
This module provides centralized reasoning capabilities for the application.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple, Union
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
    Unified reasoning service that provides consistent reasoning capabilities
    across different parts of the application.
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
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
        except Exception as e:
            logger.warning(f"Could not initialize ChromaDB: {e}")
            self.client = None
            self.vectorstore = None
            self.retriever = None
    
    def retrieve_context(self, query: str, k: int = 5, search_type: str = "mmr") -> List[Document]:
        """
        Retrieve relevant documents for a query with adaptive retrieval strategies.
        
        Args:
            query: The user query
            k: Number of documents to retrieve
            search_type: Retrieval strategy ('similarity', 'mmr', etc.)
            
        Returns:
            List of retrieved documents
        """
        if not self.retriever:
            logger.warning("Retriever not initialized")
            return []
        
        # Determine the best retrieval strategy based on query type
        query_type = self._classify_query_type(query)
        
        # Configure retrieval based on query type
        if query_type == "factual":
            # For factual questions, prioritize specific content types and use similarity search
            retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": k, "filter": self._get_factual_filter(query)}
            )
        elif query_type == "comparative":
            # For comparative questions, use MMR to get diverse results
            retriever = self.vectorstore.as_retriever(
                search_type="mmr", 
                search_kwargs={"k": k, "fetch_k": k*3, "lambda_mult": 0.7}
            )
        else:
            # Default retriever
            retriever = self.vectorstore.as_retriever(
                search_type=search_type,
                search_kwargs={"k": k}
            )
        
        try:
            return retriever.invoke(query)
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def _classify_query_type(self, query: str) -> str:
        """
        Classify the type of query to determine the best reasoning approach.
        
        Args:
            query: The user query
            
        Returns:
            Query type: "factual", "comparative", "analytical", or "general"
        """
        query_lower = query.lower()
        
        # Check for factual queries (who, what, when, where)
        if re.search(r'\b(what|who|when|where|how many|how much)\b', query_lower):
            # Specific entity questions are factual
            if any(entity in query_lower for entity in [
                "employee", "employees", "headcount", "staff", 
                "cost", "price", "amount", "badge", "energy", "water", 
                "diversity", "recycling", "utilization"
            ]):
                return "factual"
        
        # Check for comparative queries
        if re.search(r'\b(compare|comparison|versus|vs|difference|better|worse|higher|lower|most|least)\b', query_lower):
            if re.search(r'\b(city|office|location|between)\b', query_lower):
                return "comparative"
        
        # Check for analytical queries
        if re.search(r'\b(why|how|explain|analyze|trend|pattern|relationship|correlation|impact|effect)\b', query_lower):
            return "analytical"
        
        # Default to general
        return "general"
    
    def _get_factual_filter(self, query: str) -> Dict[str, Any]:
        """
        Generate appropriate filters for factual queries.
        
        Args:
            query: The user query
            
        Returns:
            Filter dictionary for ChromaDB
        """
        query_lower = query.lower()
        
        # Employee count queries
        if any(term in query_lower for term in ["employee", "employees", "headcount", "staff"]):
            return {"content_type": {"$in": ["employee_count", "employee_comparison", "all_employee_counts"]}}
        
        # ESG metrics queries
        if any(term in query_lower for term in ["esg", "environment", "social", "governance", "green", "sustainable", "leed"]):
            return {"file_type": "csv"}  # Currently all ESG data is in CSVs
        
        # Energy consumption queries
        if any(term in query_lower for term in ["energy", "electricity", "power", "kwh", "consumption"]):
            return {"source": {"$contains": "energy"}}
            
        # No specific filter needed
        return {}
    
    def generate_reasoning(self, query: str, context_docs: List[Document], detail_level: str = 'standard') -> str:
        """
        Generate reasoning for a query based on context documents.
        Supports multiple detail levels for different use cases.
        
        Args:
            query: The user query
            context_docs: List of context documents
            detail_level: Level of detail in reasoning ('basic', 'standard', or 'detailed')
            
        Returns:
            Reasoning steps as a string
        """
        if not self.llm:
            logger.warning("LLM not initialized")
            return "Reasoning could not be generated: LLM not available."
        
        # Convert documents to text
        context_text = "\n\n".join([doc.page_content for doc in context_docs[:3]])  # Limit context for reasoning
        
        # Different reasoning prompts based on detail level
        prompts = {
            'basic': f"""Given this question: "{query}" and the following context:
                
{context_text}

Provide a concise answer with minimal explanation. Focus on the key facts.
""",
            'standard': f"""Given this question: "{query}" and the following context:
                
{context_text}

Think step by step to answer the question:
1. What specific information does the question ask for?
2. What relevant data can we find in the context?
3. How can we calculate or determine the answer?
4. What is the answer based on the provided context?
""",
            'detailed': f"""Given this question: "{query}" and the following context:
                
{context_text}

Walk through your reasoning process step by step:
1. What are the key elements of this question?
2. What relevant information do we have in the context?
3. What calculations or comparisons need to be made?
4. What assumptions or limitations should be considered?
5. What conflicting information, if any, exists in the context?
6. What is the logical path to the answer?

Format your response with clear STEP 1, STEP 2, etc. headings.
"""
        }
        
        # Use the appropriate prompt based on detail level
        prompt = prompts.get(detail_level, prompts['standard'])
        
        # Generate reasoning
        try:
            return self.llm.invoke(prompt)
        except Exception as e:
            logger.error(f"Error generating reasoning: {e}")
            return f"Error generating reasoning: {str(e)}"
    
    def extract_facts(self, query: str, context_docs: List[Document]) -> List[Dict[str, Any]]:
        """
        Extract factual information from context documents.
        
        Args:
            query: The user query
            context_docs: List of context documents
            
        Returns:
            List of extracted facts with metadata
        """
        if not self.llm:
            return []
            
        context_text = "\n\n".join([doc.page_content for doc in context_docs])
        
        # Prompt for fact extraction
        prompt = f"""Extract 5-7 key facts from the following documents that are relevant to: "{query}"
        
Documents:
{context_text}

For each fact, include:
- The specific fact
- The source document or data point it comes from
- A confidence score (1-100%) indicating how certain you are about this fact

Format each fact as: "FACT: [fact] | SOURCE: [source] | CONFIDENCE: [score]"
"""
        
        try:
            facts_text = self.llm.invoke(prompt)
            
            # Parse facts
            facts = []
            fact_pattern = r"FACT: (.*?) \| SOURCE: (.*?) \| CONFIDENCE: (\d+)%"
            matches = re.findall(fact_pattern, facts_text, re.DOTALL)
            
            for match in matches:
                if len(match) == 3:
                    fact, source, confidence = match
                    facts.append({
                        "fact": fact.strip(),
                        "source": source.strip(),
                        "confidence": int(confidence) / 100.0,
                        "timestamp": datetime.now().isoformat()
                    })
            
            return facts
        except Exception as e:
            logger.error(f"Error extracting facts: {e}")
            return []
    
    def generate_self_questions(self, query: str) -> List[str]:
        """
        Generate clarifying questions for a query.
        
        Args:
            query: The user query
            
        Returns:
            List of clarifying questions
        """
        if not self.llm:
            return []
            
        prompt = f"""Based on the user's question: "{query}", 
        generate 2-3 clarifying questions that would help you better understand what they're asking.
        Focus on ambiguities, implied assumptions, or missing context that would help you give a better answer.
        Return only the questions, one per line."""
        
        try:
            response = self.llm.invoke(prompt)
            questions = [q.strip() for q in response.split('\n') if q.strip() and '?' in q]
            return questions
        except Exception as e:
            logger.error(f"Error generating self-questions: {e}")
            return []
    
    def verify_response(self, statement: str, context_docs: List[Document]) -> Dict[str, Any]:
        """
        Verify a statement against the provided context.
        
        Args:
            statement: The statement to verify
            context_docs: The context documents to verify against
            
        Returns:
            Dictionary with verification results
        """
        if not self.llm:
            return {"status": "ERROR", "reason": "LLM not available"}
            
        context_text = "\n\n".join([doc.page_content for doc in context_docs])
        
        prompt = f"""Verify if the following statement is supported by the provided documents.
        
Statement: "{statement}"

Documents:
{context_text}

Respond with:
- SUPPORTED: If the statement is directly supported by the documents
- PARTIALLY SUPPORTED: If parts of the statement are supported but not all
- UNSUPPORTED: If the statement contradicts the documents or has no supporting evidence
- UNCLEAR: If there isn't enough information to verify

Also include:
- The relevant text from the documents that supports or contradicts the statement
- A confidence score (1-100%) for your verification
"""
        
        try:
            verification_text = self.llm.invoke(prompt)
            
            # Extract verification status
            status_match = re.search(r'(SUPPORTED|PARTIALLY SUPPORTED|UNSUPPORTED|UNCLEAR)', verification_text)
            status = status_match.group(1) if status_match else "UNCLEAR"
            
            # Extract confidence score
            confidence_match = re.search(r'(\d+)%', verification_text)
            confidence = int(confidence_match.group(1)) if confidence_match else 0
            
            return {
                "status": status,
                "confidence": confidence / 100.0,
                "details": verification_text,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error verifying response: {e}")
            return {"status": "ERROR", "reason": str(e)}
    
    def analyze_confidence(self, response: str, query: str, context_docs: List[Document]) -> Dict[str, Any]:
        """
        Analyze confidence levels for different parts of a response.
        
        Args:
            response: The response to analyze
            query: The original query
            context_docs: The context documents used
            
        Returns:
            Dictionary with confidence analysis
        """
        if not self.llm:
            return {"overall": 0}
            
        context_text = "\n\n".join([doc.page_content for doc in context_docs[:2]])  # Limit context
        
        prompt = f"""Analyze the following answer to a user query and assign confidence levels to different parts of the response.

User Query: "{query}"

Context Used:
{context_text}

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
        
        try:
            confidence_analysis = self.llm.invoke(prompt)
            
            # Extract confidence scores
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
            
            return {"overall": 0, "analysis": confidence_analysis}
        except Exception as e:
            logger.error(f"Error analyzing confidence: {e}")
            return {"overall": 0, "error": str(e)}
    
    def format_answer(self, query: str, context_docs: List[Document], 
                     facts: List[Dict[str, Any]] = None, 
                     reasoning: str = None,
                     self_questions: List[str] = None) -> str:
        """
        Generate a final answer by combining context, facts, and reasoning.
        
        Args:
            query: The user query
            context_docs: Retrieved context documents
            facts: Extracted facts (optional)
            reasoning: Reasoning steps (optional)
            self_questions: Clarifying questions (optional)
            
        Returns:
            Formatted answer
        """
        if not self.llm:
            return "Could not generate answer: LLM not available"
            
        # Format context
        context_str = "\n\n".join([doc.page_content for doc in context_docs])
        
        # Format facts section
        facts_str = ""
        if facts:
            facts_str = "Relevant Facts:\n"
            for fact in facts:
                confidence_pct = int(fact["confidence"] * 100)
                facts_str += f"- {fact['fact']} (Confidence: {confidence_pct}%)\n"
            facts_str += "\n"
        
        # Format self-questions section
        questions_str = ""
        if self_questions:
            questions_str = "Clarifying Questions:\n"
            for question in self_questions:
                questions_str += f"- {question}\n"
            questions_str += "\n"
        
        # Format reasoning section
        reasoning_str = ""
        if reasoning:
            reasoning_str = f"Reasoning:\n{reasoning}\n\n"
        
        # Current date
        current_date = datetime.now().strftime('%B %d, %Y')
        
        # Format the full prompt
        full_prompt = f"""Answer the following question based on the provided context, relevant facts, and reasoning.

Current date: {current_date}

{facts_str}{questions_str}{reasoning_str}
Document Context:
{context_str}

Question: {query}

Provide a clear and detailed answer using only the information from the context and known facts.
If the answer cannot be determined from the available information, say "I don't have enough information to answer this question."
"""

        try:
            return self.llm.invoke(full_prompt)
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}"
    
    def _reframe_query(self, query: str) -> str:
        """
        Reframe user query into a standardized format that matches better with document content.
        This helps handle variations in how users might phrase the same question.
        """
        if not self.llm:
            return query
            
        # Common reframing patterns for specific query types
        query_lower = query.lower()
        
        # Define standard query templates that work well with our document structure
        query_templates = {
            "employee_count_all": "what is the total employee count across all listed offices",
            "employee_count_city": "how many employees are in the {city} office",
            "peak_entry_all": "what are the peak entry times for each office based on badge data",
            "peak_entry_city": "what is the peak entry time for the {city} office based on badge data",
            "energy_consumption_compare": "compare the energy consumption between offices",
            "meeting_room_utilization": "what is the meeting room utilization rate across offices",
            "lease_costs": "what are the lease costs for each office location"
        }
        
        # Use semantic matching to find the closest template
        if len(query.split()) > 3:  # Only for non-trivial queries
            try:
                # Create a prompt that finds the best matching template category
                prompt = f"""Determine which category this question belongs to:
                "{query}"
                
                Categories:
                1. employee_count_all - Questions about total employee numbers across all offices
                2. employee_count_city - Questions about employee numbers in a specific city office
                3. peak_entry_all - Questions about peak entry/arrival times across all offices
                4. peak_entry_city - Questions about peak entry/arrival time for a specific city
                5. energy_consumption_compare - Questions comparing energy usage between offices
                6. meeting_room_utilization - Questions about meeting room usage/occupancy
                7. lease_costs - Questions about office lease expenses
                8. other - Doesn't clearly fit any category
                
                First identify any specific city mentioned (New York, NYC, Chicago, Philadelphia, Miami, LA, Los Angeles).
                Then return the category name followed by the city if applicable.
                
                For example:
                - "how many people work for us?" → "employee_count_all"
                - "when do most staff arrive at the NYC office?" → "peak_entry_city New York"
                
                Return only the category and city, nothing else."""
                
                # Get the category and possible city
                categorization = self.llm.invoke(prompt).strip()
                
                if categorization:
                    parts = categorization.split()
                    if parts and parts[0] in query_templates:
                        category = parts[0]
                        template = query_templates[category]
                        
                        # Handle city substitution if needed and present
                        if "_city" in category and len(parts) > 1:
                            city = " ".join(parts[1:])
                            # Normalize city names
                            if city.lower() in ["nyc", "new york"]:
                                city = "New York City"
                            elif city.lower() in ["la"]:
                                city = "Los Angeles"
                            elif city.lower() in ["philly", "phl"]:
                                city = "Philadelphia"
                                
                            template = template.format(city=city)
                            
                        return template
            except Exception as e:
                logger.error(f"Error in semantic query matching: {e}")
        
        # Fall back to pattern matching for specific query types if semantic matching fails
        if any(term in query_lower for term in ["all office", "every office", "each office", "total employee", 
                                              "overall headcount", "company wide", "total staff",
                                              "all location", "all five", "all 5"]):
            if any(term in query_lower for term in ["employee", "headcount", "staff", "people", "personnel", "work"]):
                return query_templates["employee_count_all"]
                
        # For peak entry times queries
        if "peak" in query_lower and ("entry" in query_lower or "arrival" in query_lower or "badge" in query_lower):
            if "all" in query_lower or "each" in query_lower or "every" in query_lower:
                return query_templates["peak_entry_all"]
        
        # If no specific pattern matches, use LLM to reframe complex queries
        if len(query.split()) > 5 and not any(x in query_lower for x in ["what is", "how many"]):
            try:
                prompt = f"""Reframe this question to match common document retrieval patterns without changing its meaning:
                "{query}"
                
                For example:
                - "how many people work for us?" → "what is the total employee count across all offices"
                - "when do most people come to work?" → "what are the peak entry times based on badge data"
                - "which office uses the most energy?" → "what is the energy consumption comparison between offices"
                - "where do we have the highest room utilization?" → "what is the meeting room utilization rate across offices"
                - "which city has the most expensive lease?" → "what are the lease costs for each office location"
                
                Return only the reframed question, nothing else."""
                
                reframed = self.llm.invoke(prompt).strip()
                if reframed and len(reframed) > 10:  # Basic validation
                    return reframed
            except Exception as e:
                logger.error(f"Error reframing query: {e}")
                
        # Return original query if no reframing was done
        return query

    def process_query(self, query: str, detail_level: str = 'standard') -> Dict[str, Any]:
        """
        Process a user query with the unified reasoning system.
        
        Args:
            query: The user query
            detail_level: Level of detail for reasoning ('basic', 'standard', or 'detailed')
            
        Returns:
            Dictionary with response and metadata
        """
        try:
            # Reframe the query for better document matching
            original_query = query
            reframed_query = self._reframe_query(query)
            
            # Check for special case queries
            if "peak entry" in reframed_query.lower() and ("all offices" in reframed_query.lower() or "each office" in reframed_query.lower()):
                # Handle peak entry time across all offices
                result = self.process_peak_entry_time_query(reframed_query, detail_level)
                if original_query != reframed_query:
                    # Add note about reframing if query was changed
                    if "answer" in result:
                        result["answer"] = f"Understanding your question as: \"{reframed_query}\"\n\n{result['answer']}"
                return result
            
            # Step 1: Generate self-questions for clarification
            self_questions = self.generate_self_questions(reframed_query)
            
            # Step 2: Retrieve relevant documents
            context_docs = self.retrieve_context(reframed_query)
            
            # Step 3: Extract and store facts from context
            facts = self.extract_facts(reframed_query, context_docs)
            
            # Step 4: Generate reasoning steps
            reasoning = self.generate_reasoning(reframed_query, context_docs, detail_level)
            
            # Step 5: Generate final answer
            answer = self.format_answer(reframed_query, context_docs, facts, reasoning, self_questions)
            
            # Step 6: Verify response against documents
            verification = self.verify_response(answer, context_docs)
            
            # Step 7: Analyze confidence levels
            confidence_analysis = self.analyze_confidence(answer, reframed_query, context_docs)
            
            # Prepare the final result with all components
            result = {
                "answer": answer,
                "self_questions": self_questions,
                "reasoning": reasoning,
                "verification": verification,
                "confidence": confidence_analysis,
                "facts": facts,
                "context_sources": [
                    doc.metadata.get('source', 'unknown') 
                    if hasattr(doc, 'metadata') else 'unknown' 
                    for doc in context_docs
                ]
            }
            
            # Add note about reframing if query was changed
            if original_query != reframed_query and "answer" in result:
                result["answer"] = f"Understanding your question as: \"{reframed_query}\"\n\n{result['answer']}"
            
            return result
            
        except Exception as e:
            logger.exception("Error processing query")
            return {
                "answer": f"Error processing your question: {str(e)}",
                "error": str(e)
            }
    
    def process_peak_entry_time_query(self, query: str, detail_level: str = 'standard') -> Dict[str, Any]:
        """Process a query about peak entry times across all offices"""
        try:
            # Get badge data for all offices
            office_badge_data = self._get_badge_data_for_all_offices(query)
            
            # If we didn't get any badge data, process normally
            if not office_badge_data:
                return self.process_query(query, detail_level)
            
            # Analyze peak entry times for each office
            peak_times = self._analyze_peak_entry_times(office_badge_data)
            
            # Create a comprehensive answer
            all_docs = []
            for docs in office_badge_data.values():
                all_docs.extend(docs)
            
            # Format the answer with the peak times
            peak_times_text = "\n\n".join([f"{city}: {time}" for city, time in peak_times.items()])
            answer = f"""Based on the badge data analysis, here are the peak entry times for each office:

{peak_times_text}

Note that this analysis is based on available badge swipe data. For offices with insufficient data, accurate peak times could not be determined."""
            
            # Generate reasoning steps if needed
            reasoning = None
            if detail_level in ['standard', 'detailed']:
                reasoning = f"""### Step 1: Understand the Key Elements of the Question
The question asks for the peak entry time for each office. The key elements are "peak entry time" and "each office." This implies we need to identify the time of day when the most employees enter each office.

### Step 2: Gather Badge Data for All Offices
I searched for badge data containing swipe times for each city office: New York City, Chicago, Los Angeles, Miami, and Philadelphia.

### Step 3: Analyze Entry Patterns for Each Office
For each office with available badge data:
1. I filtered the swipe data to only include entries
2. I extracted the hour of the day from the swipe time
3. I counted entries by hour to identify the peak time

### Step 4: Determine the Peak Entry Time for Each Office
Based on the hourly entry counts, I identified the following peak times:
{peak_times_text}

### Step 5: Address Limitations
For some offices, there might be insufficient badge data to determine an accurate peak time. The analysis is based only on the available data.
"""
            
            # Prepare the final result
            result = {
                "answer": answer,
                "reasoning": reasoning,
                "confidence": {"overall": 0.85},
                "context_sources": [
                    doc.metadata.get('source', 'unknown') 
                    if hasattr(doc, 'metadata') else 'unknown' 
                    for doc in all_docs
                ]
            }
            
            return result
            
        except Exception as e:
            logger.exception("Error processing peak entry time query")
            return {
                "answer": f"Error processing your question about peak entry times: {str(e)}",
                "error": str(e)
            }
    
    def _get_badge_data_for_all_offices(self, query):
        """Retrieve badge data for all offices"""
        all_offices_data = {}
        
        # Find badge data for each office using more specific queries
        for office in ["NYC", "Chicago", "LA", "Miami", "Philadelphia", "New York", "Los Angeles"]:
            badge_search = f"{office} badge data swipe time"
            docs = self.retriever.invoke(badge_search)
            
            if docs:
                for doc in docs:
                    # Check if this is badge data
                    if "badge" in doc.page_content.lower() and "swipe" in doc.page_content.lower():
                        source = doc.metadata.get("source", "unknown")
                        city = None
                        
                        # Extract city from source file name
                        if "/" in source:
                            file_name = source.split("/")[-1]
                            if file_name.startswith("nyc_"):
                                city = "New York City"
                            elif file_name.startswith("chi_"):
                                city = "Chicago"
                            elif file_name.startswith("la_"):
                                city = "Los Angeles"
                            elif file_name.startswith("mia_"):
                                city = "Miami"
                            elif file_name.startswith("phl_"):
                                city = "Philadelphia"
                        
                        # If we identified the city, add the document to our collection
                        if city:
                            if city not in all_offices_data:
                                all_offices_data[city] = []
                            all_offices_data[city].append(doc)
        
        return all_offices_data

    def _analyze_peak_entry_times(self, office_badge_data):
        """Analyze badge data to determine peak entry times for each office"""
        results = {}
        
        for city, docs in office_badge_data.items():
            # Combine all badge data for this city
            combined_text = "\n\n".join([doc.page_content for doc in docs])
            
            # Create a prompt to analyze peak entry times
            prompt = f"""Analyze the following badge data for {city} and determine the peak entry time:

{combined_text}

Extract all entry times from the data, group them by hour of the day, and identify the hour with the most entries.
For example, count how many entries occur between 8:00-8:59 AM, 9:00-9:59 AM, etc.

If the data is clear, respond with just the peak entry hour in this format: "HH:00 AM/PM"
If there isn't enough data to determine the peak time, respond with "Insufficient data"
"""
            try:
                peak_time = self.llm.invoke(prompt).strip()
                results[city] = peak_time
            except Exception as e:
                results[city] = f"Error analyzing data: {str(e)}"
        
        return results