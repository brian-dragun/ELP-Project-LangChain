#!/usr/bin/env python3
import sys
from qa_system import chain, logger

def main():
    if len(sys.argv) < 2:
        print("Usage: python ask_question.py 'Your question about the documents?'")
        return

    # Get the question from command-line argument
    question = " ".join(sys.argv[1:])
    print("\n" + "-"*50)
    print(f"Question: {question}")
    print("-"*50)
    
    try:
        # Invoke the QA chain
        response = chain.invoke({"question": question})
        
        # Print the answer
        print("\nAnswer:")
        print("-"*50)
        print(response["answer"])
        print("-"*50 + "\n")
    except Exception as e:
        logger.exception("Error running the chain")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()