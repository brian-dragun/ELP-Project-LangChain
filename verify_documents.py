#!/usr/bin/env python3
"""
Utility script to verify document ingestion and retrieval,
particularly for CSV files containing employee counts.
"""

import os
import sys
import csv
import logging
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_csv_file(file_path):
    """Process a CSV file and print its contents to verify ingestion"""
    logger.info(f"Processing CSV file: {file_path}")
    try:
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            print(f"\nContents of {os.path.basename(file_path)}:")
            for i, row in enumerate(reader):
                print(f"  Row {i+1}: {row}")
                # Highlight employee count if found
                if 'NumberOfEmployees' in row:
                    print(f"  *** Employee Count: {row['NumberOfEmployees']} ***")
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
    print()

def check_document_chunks(file_path):
    """Print how documents are being chunked"""
    logger.info(f"Checking chunking for: {file_path}")
    try:
        with open(file_path, "r") as f:
            content = f.read()
        
        # Create chunks like the ingest script does
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ",", " ", ""]
        )
        
        chunks = text_splitter.split_text(content)
        
        print(f"\nChunking for {os.path.basename(file_path)}:")
        print(f"Total chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i+1}:\n  {chunk[:150]}...\n")
        
        return len(chunks)
    except Exception as e:
        logger.error(f"Error chunking {file_path}: {e}")
        return 0

def list_vector_store_entries():
    """List entries in the vector store to verify document inclusion"""
    logger.info("Checking vector store entries")
    try:
        # Initialize embeddings and vector store
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        client = chromadb.PersistentClient(path="./chroma_db")
        vectorstore = Chroma(
            client=client,
            collection_name="document_collection",
            embedding_function=embeddings
        )
        
        # Get all documents
        collection = client.get_collection("document_collection")
        result = collection.get(include=["documents", "metadatas"])
        
        print("\nVector Store Contents:")
        print(f"Total documents in vector store: {len(result['ids'])}")
        
        # Find CSVs with employee data
        employee_docs = []
        for i, (doc_id, doc_content) in enumerate(zip(result['ids'], result['documents'])):
            if "NumberOfEmployees" in doc_content or "employees" in doc_content.lower():
                employee_docs.append((doc_id, doc_content, result['metadatas'][i]))
        
        if employee_docs:
            print("\nDocuments containing employee data:")
            for i, (doc_id, content, metadata) in enumerate(employee_docs):
                print(f"  Document {i+1} (ID: {doc_id}):")
                print(f"  Content: {content[:200]}...")
                print(f"  Metadata: {metadata}")
                print()
        else:
            print("\nNo documents containing employee data found!")
            
        return len(result['ids'])
    except Exception as e:
        logger.error(f"Error accessing vector store: {e}")
        return 0

def test_retrieval(query):
    """Test document retrieval directly"""
    logger.info(f"Testing retrieval for query: '{query}'")
    try:
        # Initialize embeddings and vector store
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        client = chromadb.PersistentClient(path="./chroma_db")
        vectorstore = Chroma(
            client=client,
            collection_name="document_collection",
            embedding_function=embeddings
        )
        
        # Create a retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # Perform retrieval
        docs = retriever.invoke(query)
        
        # Print results
        print(f"\nQuery: {query}")
        print(f"Retrieved {len(docs)} documents")
        for i, doc in enumerate(docs):
            print(f"  Document {i+1}:")
            print(f"  Content: {doc.page_content[:200]}...")
            print(f"  Metadata: {doc.metadata}")
            print()
            
        return len(docs)
    except Exception as e:
        logger.error(f"Error testing retrieval: {e}")
        return 0

def main():
    """Main function to verify document processing and retrieval"""
    # 1. Check if CSV files are being properly read
    csv_files = [
        "/home/bdragun/ELP-Project-LangChain/documents/phl_esg_metrics.csv"
    ]
    for csv_file in csv_files:
        process_csv_file(csv_file)
        check_document_chunks(csv_file)
    
    # 2. Check vector store contents
    doc_count = list_vector_store_entries()
    print(f"\nTotal documents in vector store: {doc_count}")
    
    # 3. Test retrieval with relevant queries
    test_retrieval("How many employees are in Philadelphia?")
    test_retrieval("What is the employee count in Philly?")
    test_retrieval("How many people work in the Philadelphia office?")
    
    print("\nTo fix issues with document ingestion, run:")
    print("rm -rf chroma_db")
    print("python ingest_documents.py")

if __name__ == "__main__":
    main()