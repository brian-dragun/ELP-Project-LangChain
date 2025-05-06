import os
import logging
from typing import Any, Dict, List, Optional
from langchain_core.language_models import LLM
from config import Config

# Configure logging
logger = logging.getLogger(__name__)

class LambdaLabsLLM:
    """Client for Lambda Labs API"""
    
    def __init__(self, model_name: str, temperature: float = 0.1):
        """
        Initialize Lambda Labs LLM client.
        
        Args:
            model_name: Name of the model to use
            temperature: Temperature for text generation (higher = more creative)
        """
        self.model_name = model_name
        self.temperature = temperature
        logger.info(f"Initializing Lambda Labs LLM with model: {model_name}")
        
        # Ensure we're using the correct model
        if not self.model_name or self.model_name.lower() != "llama-4-maverick-17b-128e-instruct-fp8":
            logger.warning(f"Model name '{self.model_name}' is not the specified Llama Maverick model. Using default.")
            self.model_name = "llama-4-maverick-17b-128e-instruct-fp8"
        
        try:
            self.llm = self._get_lambda_model()
            logger.info(f"Successfully initialized Lambda Labs LLM with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error initializing Lambda model: {e}")
            logger.info("Attempting to initialize LambdaLabs API directly")
            try:
                from langchain.llms import LambdaAPI
                self.llm = LambdaAPI(
                    lambda_api_key=Config.LAMBDA_API_KEY,
                    model_name=self.model_name,
                    temperature=self.temperature,
                    max_tokens=Config.LLM_MAX_TOKENS
                )
                logger.info("Successfully initialized LambdaAPI directly")
            except Exception as lambda_error:
                logger.error(f"Error initializing LambdaAPI: {lambda_error}")
                logger.warning("Falling back to OpenAI model as last resort")
                try:
                    from langchain_openai import ChatOpenAI
                    self.llm = ChatOpenAI(
                        model="gpt-3.5-turbo", 
                        temperature=self.temperature,
                        openai_api_key=Config.OPENAI_API_KEY
                    )
                except Exception as fallback_error:
                    logger.error(f"Error initializing fallback LLM: {fallback_error}")
                    raise RuntimeError("Failed to initialize any LLM model")
            
    def _get_lambda_model(self) -> LLM:
        """Get the Lambda Labs model."""
        # First, try to use the LambdaLabs API directly
        try:
            # Check if LambdaLabs API package is available
            try:
                from langchain_lambda import LambdaLLM
                return LambdaLLM(
                    model=self.model_name,
                    api_url=Config.LAMBDA_API_BASE,
                    api_key=Config.LAMBDA_API_KEY,
                    temperature=self.temperature,
                    max_tokens=Config.LLM_MAX_TOKENS
                )
            except ImportError:
                # If langchain_lambda is not available, try another approach
                pass
                
            # Try standard OpenAI-compatible API approach
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model_name=self.model_name, 
                temperature=self.temperature,
                base_url=Config.LAMBDA_API_BASE,
                api_key=Config.LAMBDA_API_KEY,
                max_tokens=Config.LLM_MAX_TOKENS
            )
        except Exception as e:
            logger.error(f"Error setting up Lambda Labs model via API: {e}")
            raise
            
    def invoke(self, prompt: str, temperature: Optional[float] = None, **kwargs) -> str:
        """
        Invoke Lambda Labs API with a prompt.
        
        Args:
            prompt: Prompt to send to the API
            temperature: Optional override for temperature
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Generated text response
        """
        # Use the instance temperature unless explicitly overridden
        temp = temperature if temperature is not None else self.temperature
        
        try:
            # Pass the temperature to the model
            if hasattr(self.llm, 'temperature') and callable(getattr(self.llm, 'invoke', None)):
                response = self.llm.invoke(prompt, **kwargs)
            else:
                response = self.llm.invoke(prompt, **kwargs)
                
            # Handle different response types (string vs dict/object)
            if isinstance(response, str):
                return response
            elif hasattr(response, 'content'):
                return response.content
            elif hasattr(response, 'message') and hasattr(response.message, 'content'):
                return response.message.content
            else:
                return str(response)
        except Exception as e:
            logger.error(f"Error invoking LLM: {e}")
            return f"Error: {str(e)}"

# Initialize the LLM with the temperature from Config
llm = LambdaLabsLLM(
    model_name="llama-4-maverick-17b-128e-instruct-fp8",  # Always use Llama Maverick
    temperature=Config.LLM_TEMPERATURE  # Use the centralized temperature setting
)