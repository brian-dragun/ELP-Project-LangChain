import os
import logging
import requests
import json
import time
from typing import Any, Dict, List, Optional
from langchain_core.language_models import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from config import Config, retry_with_exponential_backoff, APIError

# Configure logging
logger = logging.getLogger(__name__)

class LambdaLabsLLM(LLM):
    api_key: str = Config.LAMBDA_API_KEY
    model_id: str = Config.LAMBDA_MODEL
    api_base: str = Config.LAMBDA_API_BASE
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
            "temperature": kwargs.get("temperature", Config.LLM_TEMPERATURE),
            "max_tokens": kwargs.get("max_tokens", Config.LLM_MAX_TOKENS)
        }
        
        if stop is not None:
            data["stop"] = stop
            
        # Log request details
        logger.info(f"Making API call to: {self.api_base}/chat/completions")
        logger.info(f"Request data: {json.dumps(data, indent=2)}")
        
        # Make the API call
        response = requests.post(
            f"{self.api_base}/chat/completions",
            headers=headers,
            json=data,
            timeout=self.timeout
        )
        
        # Handle errors
        if response.status_code != 200:
            error_msg = f"API call failed with status code {response.status_code}: {response.text}"
            logger.error(error_msg)
            raise APIError(error_msg, status_code=response.status_code, service="lambda")
        
        logger.info(f"Response status code: {response.status_code}")
        
        response_json = response.json()
        logger.info(f"Received response: {json.dumps(response_json, indent=2)}")
        
        # Extract the generated text
        try:
            generated_text = response_json["choices"][0]["message"]["content"]
            return generated_text
        except (KeyError, IndexError) as e:
            logger.error(f"Failed to extract response content: {e}")
            return "Error: Could not extract response from API"
            
    def _call_with_fallback(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """Call the API with fallback logic if the primary model fails"""
        try:
            return self._make_api_call(prompt, stop, **kwargs)
        except Exception as e:
            logger.warning(f"Primary model call failed: {e}. Attempting fallback...")
            try:
                # Try the fallback model
                fallback_model = Config.get_fallback_model()
                logger.info(f"Using fallback model: {fallback_model}")
                return self._make_api_call(prompt, stop, model_id=fallback_model, **kwargs)
            except Exception as fallback_error:
                logger.error(f"Fallback model also failed: {fallback_error}")
                return f"Error: API call failed with both primary and fallback models. Error: {str(fallback_error)}"
                
    def invoke(self, prompt: str, temperature: Optional[float] = None, **kwargs) -> str:
        """
        Invoke LLM with additional parameters.
        
        Args:
            prompt: The prompt to send to the LLM
            temperature: Optional temperature override (uses Config.LLM_TEMPERATURE by default)
            **kwargs: Additional parameters to pass to the LLM
            
        Returns:
            Generated text response
        """
        # Use the temperature from kwargs or Config if not explicitly provided
        if temperature is None:
            temperature = Config.LLM_TEMPERATURE
        
        kwargs["temperature"] = temperature
        
        # Set max_tokens from Config if not provided
        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = Config.LLM_MAX_TOKENS
        
        try:
            return self._call(prompt, **kwargs)
        except Exception as e:
            logger.error(f"Error in LLM invocation: {e}")
            return f"Error generating response: {str(e)}"

# Initialize the LLM with settings from Config
llm = LambdaLabsLLM()