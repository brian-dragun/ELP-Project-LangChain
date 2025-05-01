import os
import sys

# Import the document processing function from elp_ai_agent.py
from ai_agent import process_documents

def main():
    """
    Run the document ingestion process.
    """
    try:
        print("Starting document ingestion process...")
        vectordb = process_documents()
        print(f"Document ingestion complete! Vector database saved to ./chroma_db")
        print("You can now use elp_ai_agent.py or ask_question.py to query your documents.")
    except Exception as e:
        print(f"Error during document ingestion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()