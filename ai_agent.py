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
import time
from typing import Any, Dict, List, Mapping, Optional, TypedDict
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from config import Config, APIError, retry_with_exponential_backoff
from reasoning import ReasoningService

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
    max_retries: int = Config.API_MAX_RETRIES
    timeout: int = Config.API_REQUEST_TIMEOUT
    
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
        return self._call_with_fallback(prompt, stop, **kwargs)
    
    @retry_with_exponential_backoff
    def _make_api_call(self, prompt: str, stop: Optional[List[str]] = None, model_id: Optional[str] = None, **kwargs: Any) -> str:
        """Make an API call to Lambda Labs with retry logic"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Use specified model or fallback to the default
        model = model_id if model_id else self.model_id
        
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1024)
        }
        
        if stop is not None:
            data["stop"] = stop
        
        api_url = f"{self.api_base}/chat/completions"
        logger.info(f"Making API call to: {api_url}")
        logger.info(f"Request data: {json.dumps(data, indent=2)}")
        
        try:
            response = requests.post(api_url, headers=headers, json=data, timeout=self.timeout)
            logger.info(f"Response status code: {response.status_code}")
            
            if response.status_code != 200:
                error_msg = f"Request to Lambda Labs API failed with status {response.status_code}"
                logger.error(f"API Error: {response.text}")
                raise APIError(error_msg, status_code=response.status_code, service="lambda_labs")
            
            response_json = response.json()
            logger.info(f"Received response: {json.dumps(response_json, indent=2)}")
            
            return response_json["choices"][0]["message"]["content"].strip()
        except requests.exceptions.Timeout:
            raise APIError(f"Request to Lambda Labs API timed out after {self.timeout}s", service="lambda_labs")
        except requests.exceptions.ConnectionError:
            raise APIError(f"Connection error when connecting to Lambda Labs API", service="lambda_labs")
        except Exception as e:
            logger.exception("Error calling Lambda Labs API")
            raise APIError(f"Unexpected error: {str(e)}", service="lambda_labs")

    def _call_with_fallback(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        """Try the primary model, falling back to alternatives if needed"""
        # Try with the primary model first
        try:
            return self._make_api_call(prompt, stop, self.model_id, **kwargs)
        except Exception as e:
            logger.warning(f"Error with primary model {self.model_id}: {str(e)}")
            logger.info("Attempting to use fallback model...")
            
            # Get fallback model
            fallback_model = Config.get_fallback_model()
            if fallback_model == self.model_id:
                # If somehow the fallback is the same as primary (shouldn't happen), pick a different one
                fallback_model = "gpt-3.5-turbo"  
                
            logger.info(f"Trying fallback model: {fallback_model}")
            
            try:
                return self._make_api_call(prompt, stop, fallback_model, **kwargs)
            except Exception as fallback_error:
                logger.error(f"Fallback model {fallback_model} also failed: {str(fallback_error)}")
                
                # Last resort - use extremely basic models known for robustness
                for last_resort_model in ["gpt-3.5-turbo", "mistral-7b-instruct"]:
                    try:
                        logger.info(f"Trying last resort model: {last_resort_model}")
                        return self._make_api_call(prompt, stop, last_resort_model, **kwargs)
                    except Exception as last_error:
                        logger.error(f"Last resort model {last_resort_model} failed: {str(last_error)}")
                        continue
                
                # If all models failed, raise the original error
                logger.critical("All model attempts failed. Unable to process request.")
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

# Initialize the unified reasoning service
reasoning_service = ReasoningService(llm=llm)

def setup_retrieval_chain(query):
    """Set up a retrieval chain for the given query using the unified reasoning service."""
    # Use the reasoning service to process the query with standard reasoning level
    result = reasoning_service.process_query(query, detail_level='standard')
    
    # Create a simple chain that wraps the reasoning service
    def simple_chain(query_dict):
        question = query_dict.get("question", "")
        result = reasoning_service.process_query(question, detail_level='standard')
        return {"answer": result["answer"]}
    
    # Return the simple chain and the retriever for backward compatibility
    return simple_chain, reasoning_service.retriever

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
        
        response = chain({"question": question})
        
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