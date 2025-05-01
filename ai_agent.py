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

# Configure and create Chroma client explicitly
logger.info("Setting up ChromaDB client")
client = chromadb.PersistentClient(path="./chroma_db")

# Load the existing vector store
logger.info("Loading vector store")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    client=client,
    collection_name="document_collection",
    embedding_function=embeddings
)

# Create retriever from vectorstore
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Create prompt template for question answering
template = """Answer the following question based on the provided context:

Context:
{context}

Question: {question}

Provide a detailed answer using only the information from the context. If the answer cannot be determined from the context, say "I don't have enough information to answer this question."

Important: Do not use LaTeX formatting like \boxed{} in your answer. For highlighting important information, use emoji, asterisks, or brackets instead.

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# Define the graph nodes
def retrieve(state: GraphState) -> GraphState:
    logger.info(f"Retrieving context for question: {state['question']}")
    question = state["question"]
    docs = retriever.invoke(question)
    logger.info(f"Retrieved {len(docs)} documents")
    return {"context": docs, "question": question}

def generate_answer(state: GraphState) -> GraphState:
    logger.info("Generating answer")
    try:
        context_str = "\n\n".join([doc.page_content for doc in state["context"]])
        formatted_prompt = prompt.format(context=context_str, question=state["question"])
        logger.info(f"Formatted prompt with context of length: {len(context_str)}")
        
        response = llm.invoke(formatted_prompt)
        logger.info(f"Generated response: {response}")
        
        return {"answer": response}
    except Exception as e:
        logger.exception("Error generating answer")
        return {"answer": f"Error generating answer: {str(e)}"}

# Create the graph with the state schema
logger.info("Creating StateGraph")
graph = StateGraph(state_schema=GraphState)

# Add nodes
graph.add_node("retrieve", retrieve)
graph.add_node("generate_answer", generate_answer)

# Add edges
graph.add_edge("retrieve", "generate_answer")
graph.set_entry_point("retrieve")
graph.set_finish_point("generate_answer")

# Compile the graph
logger.info("Compiling the graph")
chain = graph.compile()

# Example usage
if __name__ == "__main__":
    try:
        # Test the system with a sample question
        question = "What information can you provide about the documents?"
        logger.info(f"Starting with question: {question}")
        
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