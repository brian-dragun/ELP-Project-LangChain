import os
from dotenv import load_dotenv
from typing import Dict, Any, Optional
import logging
import requests
import time
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

class APIError(Exception):
    """Custom exception for API-related errors"""
    def __init__(self, message: str, status_code: Optional[int] = None, service: Optional[str] = None):
        self.message = message
        self.status_code = status_code
        self.service = service
        super().__init__(self.message)

def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    max_retries: int = 3,
    max_delay: float = 30,
    errors: tuple = (APIError, requests.exceptions.RequestException)
):
    """
    Retry a function with exponential backoff for specific exceptions.
    
    Args:
        func: The function to execute with retry logic
        initial_delay: Initial delay between retries in seconds
        exponential_base: Base value for exponential calculation
        max_retries: Maximum number of retries
        max_delay: Maximum delay in seconds
        errors: Tuple of errors to catch and retry
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        retries = 0
        delay = initial_delay
        
        while True:
            try:
                return func(*args, **kwargs)
            
            except errors as e:
                if retries >= max_retries:
                    logger.error(f"Max retries ({max_retries}) exceeded. Error: {str(e)}")
                    raise e
                
                # If rate limiting is explicitly mentioned or status code indicates it
                is_rate_limit = False
                status_code = None
                
                if isinstance(e, APIError) and e.status_code:
                    status_code = e.status_code
                    if e.status_code in (429, 500, 502, 503, 504):
                        is_rate_limit = True
                
                if isinstance(e, requests.exceptions.RequestException) and hasattr(e, 'response') and e.response:
                    status_code = e.response.status_code
                    if e.response.status_code in (429, 500, 502, 503, 504):
                        is_rate_limit = True
                
                # Adjust delay for rate limit issues
                if is_rate_limit and status_code:
                    wait_time = delay * (exponential_base ** retries)
                    wait_time = min(wait_time, max_delay)
                    logger.warning(f"Rate limit or server error ({status_code}). Retrying in {wait_time:.1f}s. Error: {str(e)}")
                    time.sleep(wait_time)
                    retries += 1
                else:
                    # For other errors, just retry with backoff
                    wait_time = delay * (exponential_base ** retries)
                    wait_time = min(wait_time, max_delay)
                    logger.warning(f"API call failed. Retrying in {wait_time:.1f}s. Error: {str(e)}")
                    time.sleep(wait_time)
                    retries += 1
                    
    return wrapper

class Config:
    """Configuration manager for application settings and API keys."""
    
    # OpenAI settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Azure settings
    AZURE_AI_SERVICES_KEY = os.getenv("AZURE_AI_SERVICES_KEY")
    AZURE_AI_SERVICES_ENDPOINT = os.getenv("AZURE_AI_SERVICES_ENDPOINT")
    AZURE_AI_SERVICES_REGION = os.getenv("AZURE_AI_SERVICES_REGION")
    AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    
    # Azure AI Foundry settings
    AZURE_FOUNDRY_API_KEY = os.getenv("AZURE_FOUNDRY_API_KEY")
    AZURE_FOUNDRY_ENDPOINT = os.getenv("AZURE_FOUNDRY_ENDPOINT")
    AZURE_FOUNDRY_PROJECT_NAME = os.getenv("AZURE_FOUNDRY_PROJECT_NAME")
    AZURE_FOUNDRY_MODEL_ID = os.getenv("AZURE_FOUNDRY_MODEL_ID")
    AZURE_FOUNDRY_DEPLOYMENT_NAME = os.getenv("AZURE_FOUNDRY_DEPLOYMENT_NAME")
    AZURE_FOUNDRY_API_VERSION = os.getenv("AZURE_FOUNDRY_API_VERSION", "2023-10-01-preview")
    
    # LangChain settings
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    
    # LangSmith settings
    LANGSMITH_PROJECT = os.getenv("LANGCHAIN_PROJECT")
    LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")
    LANGSMITH_TRACING_ENABLED = os.getenv("LANGCHAIN_TRACING", "true").lower() == "true"
    
    # Lambda Labs settings
    LAMBDA_API_KEY = os.getenv("LAMBDA_API_KEY")
    LAMBDA_API_BASE = os.getenv("LAMBDA_API_BASE")
    LAMBDA_MODEL = os.getenv("LAMBDA_MODEL")
    
    # LLM Settings
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))
    
    # Retrieval settings
    RETRIEVAL_DEFAULT_K = int(os.getenv("RETRIEVAL_DEFAULT_K", "15"))
    RETRIEVAL_FACTUAL_K = int(os.getenv("RETRIEVAL_FACTUAL_K", "5"))
    RETRIEVAL_COMPARATIVE_K = int(os.getenv("RETRIEVAL_COMPARATIVE_K", "8"))
    RETRIEVAL_COMPARATIVE_FETCH_K = int(os.getenv("RETRIEVAL_COMPARATIVE_FETCH_K", "20"))
    RETRIEVAL_ANALYTICAL_K = int(os.getenv("RETRIEVAL_ANALYTICAL_K", "12"))
    RETRIEVAL_USE_MMR = os.getenv("RETRIEVAL_USE_MMR", "true").lower() == "true"
    RETRIEVAL_MMR_DIVERSITY = float(os.getenv("RETRIEVAL_MMR_DIVERSITY", "0.3"))
    
    # API request settings
    API_MAX_RETRIES = int(os.getenv("API_MAX_RETRIES", "3"))
    API_REQUEST_TIMEOUT = int(os.getenv("API_REQUEST_TIMEOUT", "30"))
    
    @classmethod
    def get_all_configs(cls) -> Dict[str, Any]:
        """Return all configuration values as a dictionary."""
        return {key: value for key, value in cls.__dict__.items() 
                if not key.startswith('_') and not callable(value)}
                
    @classmethod
    def validate_config(cls) -> bool:
        """Check if all required configuration values are set."""
        required_vars = [
            "OPENAI_API_KEY", 
            "AZURE_AI_SERVICES_KEY",
            "AZURE_AI_SERVICES_ENDPOINT",
            "AZURE_FOUNDRY_API_KEY",
            "AZURE_FOUNDRY_ENDPOINT",
            "AZURE_FOUNDRY_PROJECT_NAME"
        ]
        
        missing = [var for var in required_vars if getattr(cls, var) is None]
        if missing:
            logger.error(f"Missing required configuration: {', '.join(missing)}")
            return False
        return True
        
    @classmethod
    def get_fallback_model(cls) -> str:
        """Get the fallback model name if the preferred model is unavailable."""
        primary_model = cls.LAMBDA_MODEL
        # Define fallback chain - from most to least preferred
        fallbacks = [
            "llama-4-maverick-17b-128e-instruct-fp8",
            "llama-4-geopolitics-17b-128e-instruct-fp8",
            "mistral-large-2402-2024-04-02",
            "claude-3-opus-20240229",
            "gpt-4-turbo",
            "gpt-3.5-turbo"
        ]
        
        # If primary model is in fallbacks, start from the next one
        if primary_model in fallbacks:
            index = fallbacks.index(primary_model)
            return fallbacks[index + 1] if index + 1 < len(fallbacks) else fallbacks[0]
        
        return fallbacks[0]