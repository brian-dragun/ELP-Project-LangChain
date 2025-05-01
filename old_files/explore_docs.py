#!/usr/bin/env python3
import os
import sys
import pandas as pd
import glob

def list_documents():
    """List all CSV files in the documents directory."""
    doc_path = os.path.join(os.path.dirname(__file__), 'documents')
    files = glob.glob(f"{doc_path}/*.csv")
    
    print("\nAvailable documents:")
    print("-"*50)
    
    # Group files by city
    cities = {}
    for file in files:
        filename = os.path.basename(file)
        prefix = filename.split('_')[0] if '_' in filename else 'other'
        
        if prefix not in cities:
            cities[prefix] = []
        
        cities[prefix].append(filename)
    
    # Print organized by city
    for city, files in cities.items():
        print(f"\n{city.upper()}:")
        for file in sorted(files):
            print(f"  - {file}")

def read_document(filename):
    """Read and display a CSV file from the documents directory."""
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    doc_path = os.path.join(os.path.dirname(__file__), 'documents', filename)
    
    if not os.path.exists(doc_path):
        # Try to find a match
        all_docs = glob.glob(os.path.join(os.path.dirname(__file__), 'documents', '*.csv'))
        possible_matches = [os.path.basename(f) for f in all_docs if filename.lower() in f.lower()]
        
        if possible_matches:
            print(f"\nFile '{filename}' not found. Did you mean one of these?")
            for match in possible_matches:
                print(f"  - {match}")
        else:
            print(f"\nFile '{filename}' not found.")
        
        return
    
    try:
        df = pd.read_csv(doc_path)
        print(f"\nContents of {filename}:")
        print("-"*50)
        print(df.to_string())
        print("\nDataFrame information:")
        print("-"*50)
        df.info()
        print("\nSummary statistics:")
        print("-"*50)
        print(df.describe())
    except Exception as e:
        print(f"Error reading file: {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python explore_docs.py list                  # List all available documents")
        print("  python explore_docs.py read <filename.csv>   # Read and display a document")
        return
    
    command = sys.argv[1].lower()
    
    if command == 'list':
        list_documents()
    elif command == 'read' and len(sys.argv) > 2:
        read_document(sys.argv[2])
    else:
        print("Invalid command. Use 'list' or 'read <filename>'.")

if __name__ == "__main__":
    main()