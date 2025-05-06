#!/usr/bin/env python3
"""
Azure AI Foundry integration for Document Intelligence System.
This module provides classes and functions to interact with Azure AI Foundry services.
"""

import logging
import json
import requests
import time
from typing import Dict, List, Any, Optional, Union
from config import Config, retry_with_exponential_backoff, APIError

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AzureFoundryClient:
    """Client for interacting with Azure AI Foundry services."""
    
    def __init__(self):
        """Initialize the Azure AI Foundry client with config values."""
        self.api_key = Config.AZURE_FOUNDRY_API_KEY
        self.endpoint = Config.AZURE_FOUNDRY_ENDPOINT.rstrip('/')
        self.project_name = Config.AZURE_FOUNDRY_PROJECT_NAME
        self.model_id = Config.AZURE_FOUNDRY_MODEL_ID
        self.deployment_name = Config.AZURE_FOUNDRY_DEPLOYMENT_NAME
        self.api_version = Config.AZURE_FOUNDRY_API_VERSION
        
        if not all([self.api_key, self.endpoint, self.project_name]):
            logger.error("Missing required Azure Foundry configuration")
            raise ValueError("Azure Foundry configuration is incomplete")
    
    def get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for Azure AI Foundry API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    @retry_with_exponential_backoff
    def query_deployment(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000) -> Dict[str, Any]:
        """
        Query the Azure AI Foundry deployment with a prompt.
        
        Args:
            prompt: The prompt to send to the model
            temperature: Controls randomness (0-1)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            The response from the Azure AI Foundry API
        """
        url = f"{self.endpoint}/deployments/{self.deployment_name}/chat/completions?api-version={self.api_version}"
        
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(url, headers=self.get_auth_headers(), json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            status_code = e.response.status_code if hasattr(e, 'response') and e.response else None
            msg = f"Error querying Azure AI Foundry: {str(e)}"
            raise APIError(msg, status_code=status_code, service="AzureFoundry")
    
    @retry_with_exponential_backoff
    def upload_document(self, file_path: str, document_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Upload a document to Azure AI Foundry for ingestion.
        
        Args:
            file_path: Path to the document file
            document_name: Optional custom name for the document
            
        Returns:
            The response from the Azure AI Foundry API
        """
        url = f"{self.endpoint}/projects/{self.project_name}/documents?api-version={self.api_version}"
        
        document_name = document_name or file_path.split('/')[-1]
        
        try:
            with open(file_path, 'rb') as f:
                files = {'file': (document_name, f)}
                response = requests.post(url, headers={"Authorization": f"Bearer {self.api_key}"}, files=files)
                response.raise_for_status()
                return response.json()
        except requests.exceptions.RequestException as e:
            status_code = e.response.status_code if hasattr(e, 'response') and e.response else None
            msg = f"Error uploading document to Azure AI Foundry: {str(e)}"
            raise APIError(msg, status_code=status_code, service="AzureFoundry")
    
    @retry_with_exponential_backoff
    def search_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search documents in Azure AI Foundry using semantic search.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            
        Returns:
            List of document search results
        """
        url = f"{self.endpoint}/projects/{self.project_name}/search?api-version={self.api_version}"
        
        payload = {
            "query": query,
            "top": top_k
        }
        
        try:
            response = requests.post(url, headers=self.get_auth_headers(), json=payload)
            response.raise_for_status()
            return response.json().get("results", [])
        except requests.exceptions.RequestException as e:
            status_code = e.response.status_code if hasattr(e, 'response') and e.response else None
            msg = f"Error searching documents in Azure AI Foundry: {str(e)}"
            raise APIError(msg, status_code=status_code, service="AzureFoundry")
    
    @retry_with_exponential_backoff
    def generate_suggested_questions(self, num_questions: int = 5) -> List[str]:
        """
        Generate suggested questions based on the document content in the project.
        
        Args:
            num_questions: Number of questions to generate
            
        Returns:
            List of suggested questions
        """
        # First retrieve a sample of documents to understand content
        url = f"{self.endpoint}/projects/{self.project_name}/documents?api-version={self.api_version}&top=10"
        
        try:
            # Get a sample of documents
            response = requests.get(url, headers=self.get_auth_headers())
            response.raise_for_status()
            documents = response.json().get("value", [])
            
            if not documents:
                logger.warning("No documents found to generate questions from")
                return [
                    "What features does this system provide?",
                    "How can I search for specific information?",
                    "What types of documents can be processed?"
                ]
            
            # Extract topics and content samples from documents
            document_snippets = []
            for doc in documents[:5]:  # Use up to 5 documents for context
                doc_id = doc.get("id")
                doc_url = f"{self.endpoint}/projects/{self.project_name}/documents/{doc_id}?api-version={self.api_version}"
                doc_response = requests.get(doc_url, headers=self.get_auth_headers())
                if doc_response.status_code == 200:
                    content = doc_response.json().get("content", "")
                    # Take a snippet to understand document content
                    document_snippets.append(content[:500])
            
            # If we couldn't get content, return default questions
            if not document_snippets:
                logger.warning("Could not retrieve document content")
                return [
                    "What information is available in the system?",
                    "Can you summarize the key topics in the documents?",
                    "What are the main themes covered in the content?"
                ]
            
            # Now use the deployment to generate questions based on content
            prompt = f"""Based on the following document snippets, generate {num_questions} specific, engaging questions 
            that a user might want to ask about this content. The questions should be diverse and cover different 
            aspects of the information available.
            
            Document snippets:
            {document_snippets}
            
            Generate {num_questions} questions that would be interesting and relevant to users wanting to learn about this content.
            Format your response as a JSON array of strings, with each string being a question.
            """
            
            response_data = self.query_deployment(prompt, temperature=0.7)
            response_text = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            # Try to parse questions from JSON format
            try:
                # Find JSON array in response
                if '[' in response_text and ']' in response_text:
                    json_str = response_text[response_text.find('['):response_text.rfind(']')+1]
                    questions = json.loads(json_str)
                    return questions[:num_questions]
                else:
                    # Fallback: extract questions line by line
                    questions = [line.strip().strip('"-').strip() 
                                for line in response_text.split('\n') 
                                if line.strip() and '?' in line]
                    return questions[:num_questions]
            except json.JSONDecodeError:
                # Fallback parsing if JSON fails
                questions = []
                for line in response_text.split('\n'):
                    line = line.strip()
                    if line and '?' in line:
                        # Clean up the line to extract just the question
                        question = line.replace('"', '').replace('- ', '').strip()
                        if question not in questions:
                            questions.append(question)
                
                return questions[:num_questions] if questions else [
                    "What are the main topics covered in these documents?",
                    "Can you summarize the key findings from the documents?",
                    "What specific information can I learn from these documents?"
                ]
        except Exception as e:
            logger.error(f"Error generating suggested questions: {str(e)}")
            return [
                "What information is available in the system?",
                "What types of documents have been uploaded?",
                "How can this system help me find specific information?"
            ]

class AzureFoundryReasoning:
    """
    Reasoning service that uses Azure AI Foundry for document intelligence.
    This class is meant to replace the local reasoning.py functionality.
    """
    
    def __init__(self):
        """Initialize the Azure AI Foundry reasoning service."""
        self.client = AzureFoundryClient()
    
    def retrieve_context(self, query: str, top_k: int = 15) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query using Azure AI Foundry search.
        
        Args:
            query: The user query
            top_k: Number of results to return
            
        Returns:
            List of retrieved documents
        """
        try:
            search_results = self.client.search_documents(query, top_k=top_k)
            return search_results
        except Exception as e:
            logger.error(f"Error retrieving documents from Azure AI Foundry: {e}")
            return []
    
    def process_query(self, query: str, detail_level: str = 'standard') -> Dict[str, Any]:
        """
        Process a user query using Azure AI Foundry.
        
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
            context_sources = [doc.get('metadata', {}).get('source', 'unknown') for doc in context_docs]
            
            # Format the context
            context_text = "\n\n".join([doc.get('content', '') for doc in context_docs])
            
            # Step 2: Create prompt based on detail level
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
            
            # Generate response using Azure Foundry
            response_data = self.client.query_deployment(unified_prompt)
            response_text = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            # Extract the final answer and reasoning
            reasoning, answer = self._extract_reasoning_and_answer(response_text)
            
            # Calculate confidence based on response
            confidence = self._analyze_confidence(response_text, context_docs)
            
            result = {
                "answer": answer,
                "reasoning": reasoning,
                "confidence": confidence,
                "context_sources": context_sources
            }
            
            return result
            
        except Exception as e:
            logger.exception("Error processing query with Azure AI Foundry")
            return {
                "answer": f"Error processing your question: {str(e)}",
                "error": str(e)
            }
    
    def _extract_reasoning_and_answer(self, response: str) -> tuple:
        """
        Extract reasoning and answer from the AI's response.
        
        Args:
            response: The full response from the LLM
            
        Returns:
            Tuple of (reasoning, answer)
        """
        # Simple approach: the reasoning is the full text except the last paragraph
        # This is a basic approach, might need refinement for your specific model outputs
        paragraphs = response.split('\n\n')
        
        if len(paragraphs) <= 1:
            # If there's only one paragraph, it's both the reasoning and the answer
            return response, response
        
        # Consider the last paragraph as the final answer
        answer = paragraphs[-1]
        # Everything else is considered reasoning
        reasoning = '\n\n'.join(paragraphs[:-1])
        
        return reasoning, answer
    
    def _analyze_confidence(self, response: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        Analyze the confidence level of the response.
        
        Args:
            response: The model's response text
            context_docs: The retrieved context documents
            
        Returns:
            Confidence level as a string: 'high', 'medium', or 'low'
        """
        # Check if response contains uncertainty markers
        uncertainty_phrases = [
            "I'm not sure", "I don't know", "not enough information", 
            "cannot determine", "unclear", "uncertain", "might be", "possibly", 
            "could be", "not specified", "insufficient data"
        ]
        
        if any(phrase in response.lower() for phrase in uncertainty_phrases):
            return "low"
        
        # If we have strong context matches and no uncertainty, confidence is high
        if len(context_docs) >= 3:
            return "high"
        
        # Default to medium confidence
        return "medium"