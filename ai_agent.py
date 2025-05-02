import os
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from langchain_chroma import Chroma
import requests
import json
import logging
import sys
from typing import Any, Dict, List, Mapping, Optional, TypedDict
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure LangSmith tracing
os.environ["LANGCHAIN_API_KEY"] = Config.LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = Config.LANGSMITH_PROJECT
os.environ["LANGCHAIN_ENDPOINT"] = Config.LANGSMITH_ENDPOINT
os.environ["LANGCHAIN_TRACING_V2"] = "true" if Config.LANGSMITH_TRACING_ENABLED else "false"

# Define the state schema
class GraphState(TypedDict):
    question: str
    context: Optional[List]
    answer: Optional[str]

# Custom Lambda Labs LLM implementation
class LambdaLabsLLM(LLM):
    api_key: str
    model_id: str
    api_base: str
    
    @property
    def _llm_type(self) -> str:
        return "lambda_labs"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 1024
        }
        
        if stop is not None:
            data["stop"] = stop
        
        api_url = f"{self.api_base}/chat/completions"
        logger.info(f"Making API call to: {api_url}")
        logger.info(f"Request data: {json.dumps(data, indent=2)}")
        
        try:
            response = requests.post(api_url, headers=headers, json=data)
            logger.info(f"Response status code: {response.status_code}")
            
            if response.status_code != 200:
                logger.error(f"API Error: {response.text}")
                raise ValueError(f"Request to Lambda Labs API failed: {response.text}")
            
            response_json = response.json()
            logger.info(f"Received response: {json.dumps(response_json, indent=2)}")
            
            return response_json["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.exception("Error calling Lambda Labs API")
            raise e

# Set Lambda Labs settings
LAMBDA_API_KEY = Config.LAMBDA_API_KEY
LAMBDA_API_BASE = Config.LAMBDA_API_BASE
LAMBDA_MODEL = Config.LAMBDA_MODEL

# Validate config
if not LAMBDA_API_KEY:
    logger.error("Lambda API key is missing from configuration")
    raise ValueError("Lambda API key is required")

logger.info(f"Initializing Lambda Labs LLM with model: {LAMBDA_MODEL}")

# Initialize Lambda Labs LLM
llm = LambdaLabsLLM(
    api_key=LAMBDA_API_KEY, 
    model_id=LAMBDA_MODEL,
    api_base=LAMBDA_API_BASE
)

def setup_retrieval_chain(query):
    """Set up a retrieval chain for the given query."""
    # Load the vector store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Check if ChromaDB exists, if not create a warning
    if not os.path.exists("./chroma_db"):
        logger.warning("ChromaDB not found. Please run ingest_documents.py first.")
        return None, None
    
    client = chromadb.PersistentClient(path="./chroma_db")
    vectorstore = Chroma(
        client=client, 
        collection_name="document_collection",
        embedding_function=embeddings
    )
    
    # Use Maximum Marginal Relevance for better diverse results
    # This helps get both general and specific documents about the same topic
    retriever = vectorstore.as_retriever(
        search_type="mmr",  # Maximum Marginal Relevance
        search_kwargs={
            "k": 8,         # Retrieve more documents
            "fetch_k": 20,  # Consider more candidates
            "lambda_mult": 0.7  # Balance between relevance and diversity
        }
    )
    
    # Create a filter for employee count queries to prioritize those documents
    if any(term in query.lower() for term in 
           ["employee", "employees", "people", "staff", "headcount"]):
        logger.info("Employee-related query detected, applying metadata filter")
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "filter": {"content_type": "employee_count"}  # First try with specific docs
            }
        )
    
    # Set up the prompt
    system_prompt = """You are an AI assistant that helps analyze documents about office data.
Answer questions based ONLY on the provided context. 
Be precise, detailed and show your step-by-step reasoning.
If the answer cannot be determined from the context, say so clearly.
If you find conflicting information, explain the discrepancy.

For questions about employee counts, check all relevant sources and specify which office/city you're referring to.
"""
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Context:\n\n{context}\n\nQuestion: {question}")
    ])
    
    # Create the chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain, retriever

# Example usage
if __name__ == "__main__":
    try:
        # Test the system with a sample question
        question = "What information can you provide about the documents?"
        logger.info(f"Starting with question: {question}")
        
        chain, retriever = setup_retrieval_chain(question)
        if chain is None:
            logger.error("Failed to set up retrieval chain")
            sys.exit(1)
        
        response = chain.invoke({"question": question})
        
        logger.info("Chain execution completed")
        
        # Ensure these prints are displayed
        print("\n" + "-"*50)
        print(f"Question: {question}")
        print(f"Answer: {response['answer']}")
        print("-"*50 + "\n")
        
        # Flush stdout to ensure output is displayed immediately
        sys.stdout.flush()
    except Exception as e:
        logger.exception("Error running the chain")
        print(f"Error: {str(e)}")
        sys.stdout.flush()