#!/usr/bin/env python3
import sys
from ai_agent import setup_qa_system, ask_question

def main():
    # Check if a question was provided as a command-line argument
    if len(sys.argv) < 2:
        print("Usage: ./ask_question.py 'Your question here'")
        sys.exit(1)
    
    # Get the question from command-line arguments
    question = sys.argv[1]
    
    # Set up the QA chain
    qa_chain = setup_qa_system()
    
    # Ask the question
    result = ask_question(qa_chain, question)
    
    # Display the result
    print(f"\nAnswer: {result}")

if __name__ == "__main__":
    main()