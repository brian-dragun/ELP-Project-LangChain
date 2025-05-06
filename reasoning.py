#!/usr/bin/env python3
"""
Simplified Reasoning System for document intelligence.
This version relies more on the AI's inherent capabilities with less template scaffolding.
"""

import logging
import re
import traceback
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
    
    def process_query(self, query, detail_level="standard"):
        """Process a user query with detailed reasoning steps."""
        try:
            context = self.retrieve_context(query)
            prompt = self.build_prompt(query, context, detail_level)
            
            # Use the temperature from Config for the API call
            # This allows for more consistent results across different queries
            llm_response_text = self.llm.invoke(
                prompt, 
                temperature=Config.LLM_TEMPERATURE  # Use centralized temperature setting
            )
            
            parsed = self.parse_response(llm_response_text)
            
            answer = parsed.get("answer", "")
            reasoning = parsed.get("reasoning", "")

            cleaned_answer = self._clean_answer(answer)

            # If the cleaned answer is still very short or empty, and reasoning is present,
            # Try to extract a better answer from reasoning
            if reasoning and (not cleaned_answer or len(cleaned_answer.strip()) < 20):
                conclusion_markers = ["Conclusion:", "In conclusion:", "Therefore,", "Thus,", "To conclude:", "To summarize:", "Form a clear conclusion:"]
                conclusion_found_in_reasoning = False
                for marker in conclusion_markers:
                    if marker.lower() in reasoning.lower():
                        # Split by marker (case-insensitive) and take the part after it
                        parts = re.split(marker, reasoning, maxsplit=1, flags=re.IGNORECASE)
                        if len(parts) > 1:
                            conclusion_part = parts[1].split("\n")[0].strip() # Take first line after marker
                            if len(conclusion_part) > 10:
                                cleaned_answer = self._clean_answer(conclusion_part)
                                conclusion_found_in_reasoning = True
                                break
                
                # If no explicit conclusion marker found, but answer is still too short,
                # try taking the last meaningful sentence of the reasoning.
                if not conclusion_found_in_reasoning and (not cleaned_answer or len(cleaned_answer.strip()) < 20):
                    sentences = re.split(r'(?<=[.!?])\s+', reasoning.strip())
                    if sentences:
                        for sent in reversed(sentences):
                            sent = sent.strip()
                            if sent and len(sent) > 15 and sent.endswith(('.', '!', '?')):
                                cleaned_answer = self._clean_answer(sent)
                                break
            
            if not cleaned_answer and not reasoning: # If both are empty after all processing
                cleaned_answer = "I couldn't find an answer to your question or generate a reasoning process."
            elif not cleaned_answer and reasoning: # If no answer but reasoning exists
                cleaned_answer = "Please see the reasoning process for details."

            return {
                "answer": cleaned_answer,
                "reasoning": reasoning,
                "confidence": "high" if context else "low",
                "context_sources": list(set([c.metadata.get("source", "Unknown") for c in context if hasattr(c, "metadata")]))
            }
        
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}\n{traceback.format_exc()}")
            return {
                "answer": f"Sorry, I encountered an error while processing your query: {str(e)}",
                "reasoning": f"Error occurred: {traceback.format_exc()}",
                "confidence": "error",
                "context_sources": []
            }
    
    def build_prompt(self, query: str, context: List[Dict], detail_level="standard"):
        """
        Build a prompt for the LLM with context and query.
        
        Args:
            query: The user query
            context: The retrieved context documents
            detail_level: Level of detail for the reasoning ("standard" or "detailed")
            
        Returns:
            Structured prompt for the LLM
        """
        current_date = datetime.now().strftime("%Y-%m-%d")
        formatted_context = ""
        if context:
            for i, doc_obj in enumerate(context, 1):
                # Ensure doc_obj is treated as a Document object if it has metadata
                source = doc_obj.metadata.get("source", "Unknown source") if hasattr(doc_obj, 'metadata') else "Unknown source"
                content = doc_obj.page_content if hasattr(doc_obj, 'page_content') else str(doc_obj)
                formatted_context += f"Document {i} [Source: {source}]:\n{content}\n\n"
        
        if detail_level == "detailed":
            system_prompt = f"""You are an AI assistant focused on real estate analysis. Today is {current_date}.
You MUST structure your response in two distinct parts, using the exact headings specified below:

### Final Answer ###
(Provide a concise, direct answer to the user's question here. This should be the final conclusion, not the reasoning steps.)

### Reasoning Process ###
(Provide a comprehensive, step-by-step reasoning process that shows how you arrived at the final answer. Follow these analytical steps:
1. Identify relevant information from the provided documents.
2. Extract key metrics and data points.
3. Perform any necessary comparisons or calculations.
4. Interpret the results in context.
5. Form a clear conclusion (this conclusion will be summarized in the 'Final Answer' section above).
Be explicit about your thought process and reference specific documents if applicable.)

Context documents:
{formatted_context}

Question: {query}
"""
        else:
            system_prompt = f"""You are an AI assistant focused on real estate analysis. Today is {current_date}.
Please answer the user's question based on the provided documents.

Context documents:
{formatted_context}

Question: {query}
"""
        return system_prompt
    
    def parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM response to extract reasoning and answer components
        using specific headings.
        """
        answer_marker = "### Final Answer ###"
        reasoning_marker = "### Reasoning Process ###"
        
        answer = ""
        reasoning = ""

        try:
            answer_start_index = response.find(answer_marker)
            reasoning_start_index = response.find(reasoning_marker)

            if answer_start_index != -1 and reasoning_start_index != -1:
                if answer_start_index < reasoning_start_index:
                    answer_content_start = answer_start_index + len(answer_marker)
                    answer = response[answer_content_start:reasoning_start_index].strip()
                    reasoning_content_start = reasoning_start_index + len(reasoning_marker)
                    reasoning = response[reasoning_content_start:].strip()
                else: # Reasoning marker appears before answer marker
                    reasoning_content_start = reasoning_start_index + len(reasoning_marker)
                    reasoning = response[reasoning_content_start:answer_start_index].strip()
                    answer_content_start = answer_start_index + len(answer_marker)
                    answer = response[answer_content_start:].strip()
            elif answer_start_index != -1: # Only answer marker found
                answer_content_start = answer_start_index + len(answer_marker)
                answer = response[answer_content_start:].strip()
                # Attempt to find reasoning in the part before the answer marker if it looks like reasoning
                potential_reasoning = response[:answer_start_index].strip()
                if len(potential_reasoning) > len(answer) * 2 or "step" in potential_reasoning.lower(): # Heuristic
                    reasoning = potential_reasoning
            elif reasoning_start_index != -1: # Only reasoning marker found
                reasoning_content_start = reasoning_start_index + len(reasoning_marker)
                reasoning = response[reasoning_content_start:].strip()
                # Attempt to find answer in the part before the reasoning marker
                potential_answer = response[:reasoning_start_index].strip()
                if potential_answer and (len(potential_answer) < len(reasoning) or not "step" in potential_answer.lower()):
                     answer = potential_answer

            if not answer and not reasoning: # Neither marker found, use fallback
                logger.warning("Specific markers not found in LLM response. Using fallback parsing.")
                reasoning, answer = self._extract_reasoning_and_answer(response)
            elif not answer and reasoning: # Reasoning found, but no clear answer
                 # Try to extract a conclusion from the end of reasoning as the answer
                conclusion_match = re.search(r"(?:Conclusion:|Form a clear conclusion:)\s*(.*)", reasoning, re.IGNORECASE | re.DOTALL)
                if conclusion_match and conclusion_match.group(1).strip():
                    answer = conclusion_match.group(1).strip().split('\n')[0] # Take first line of conclusion
                elif len(reasoning.split('.')) > 2: # take last sentence of reasoning if no explicit conclusion
                    answer = reasoning.split('.')[-2].strip() + "."

            if not reasoning and answer and len(answer) > 300: # Answer found, but no clear reasoning, and answer is very long
                # Assume the long answer might contain reasoning, try to split it
                temp_reasoning, temp_answer = self._extract_reasoning_and_answer(answer)
                if temp_reasoning: # If fallback found reasoning within the long answer
                    reasoning = temp_reasoning
                    answer = temp_answer

            return {
                "reasoning": reasoning.strip(),
                "answer": answer.strip()
            }
            
        except Exception as e:
            logger.error(f"Error parsing response with specific markers: {str(e)}. Falling back.")
            # Fall back to the original extraction method if specific marker parsing fails
            reasoning, answer = self._extract_reasoning_and_answer(response)
            return {
                "reasoning": reasoning,
                "answer": answer
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
    
    def _clean_answer(self, answer: str) -> str:
        """Cleans the answer by removing common LaTeX artifacts."""
        if not answer:
            return ""
        # Remove $\boxed{...}$
        answer = re.sub(r"\\boxed{([^}]*)}", r"\1", answer)
        # Remove $...$ if it's just wrapping a number or simple text
        answer = re.sub(r"\$([^$]*)\$", r"\1", answer)
        # Remove \(...\) and \[...\]
        answer = re.sub(r"\\\((.*?)\\\)", r"\1", answer)
        answer = re.sub(r"\\\[(.*?)\\\]", r"\1", answer)
        # Remove any remaining isolated $
        answer = answer.replace("$", "")
        # Trim whitespace
        return answer.strip()