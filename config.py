import os
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration manager for application settings and API keys."""
    
    # OpenAI settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Azure settings
    AZURE_AI_SERVICES_KEY = os.getenv("AZURE_AI_SERVICES_KEY")
    AZURE_AI_SERVICES_ENDPOINT = os.getenv("AZURE_AI_SERVICES_ENDPOINT")
    AZURE_AI_SERVICES_REGION = os.getenv("AZURE_AI_SERVICES_REGION")
    AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    
    # LangChain settings
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    
    # Lambda Labs settings
    LAMBDA_API_KEY = os.getenv("LAMBDA_API_KEY")
    LAMBDA_API_BASE = os.getenv("LAMBDA_API_BASE")
    LAMBDA_MODEL = os.getenv("LAMBDA_MODEL")
    
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
            "AZURE_AI_SERVICES_ENDPOINT"
        ]
        
        missing = [var for var in required_vars if getattr(cls, var) is None]
        if missing:
            print(f"Missing required configuration: {', '.join(missing)}")
            return False
        return True