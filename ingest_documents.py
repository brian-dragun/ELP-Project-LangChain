#!/usr/bin/env python3
"""
Script to ingest documents into a vector database.
"""

import os
import glob
import logging
import sys
import csv
from typing import List, Dict, Any, Callable
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
import time
import tqdm  # Import tqdm for progress bars

# Import document loaders
from langchain_community.document_loaders import (
    CSVLoader,
    TextLoader,
    PyPDFLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredExcelLoader,
    UnstructuredWordDocumentLoader,
    DirectoryLoader
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Map file extensions to appropriate loaders
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    ".pdf": (PyPDFLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".htm": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".xlsx": (UnstructuredExcelLoader, {}),
    ".xls": (UnstructuredExcelLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {})
}

def process_csv_file(file_path: str, text_splitter: RecursiveCharacterTextSplitter) -> List[Document]:
    """
    Process a CSV file with better metadata and chunking for employee data.
    """
    logger.info(f"Processing CSV file: {file_path}")
    filename = os.path.basename(file_path)
    documents = []
    
    try:
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)  # Get headers
            
            # Process each row as a separate document with detailed metadata
            for row_num, row in enumerate(reader, start=2):  # Start at 2 for header+1
                # Create a dictionary from the row
                row_dict = {headers[i]: value for i, value in enumerate(row) if i < len(headers)}
                
                # Extract key information for metadata
                city = row_dict.get("City", "")
                if not city and "city" in [h.lower() for h in headers]:
                    # Try case-insensitive match
                    for header in headers:
                        if header.lower() == "city":
                            city = row_dict.get(header, "")
                
                # Extract employee count with fallbacks for different column names
                employee_count = None
                employee_fields = ["NumberOfEmployees", "Employee_Count", "Employees", "HeadCount"]
                for field in employee_fields:
                    if field in row_dict and row_dict[field]:
                        employee_count = row_dict[field]
                        break
                
                # Create a well-structured content string
                content = [f"File: {filename}, Row: {row_num}"]
                
                # Add important fields first for better retrieval
                priority_fields = ["City", "NumberOfEmployees", "Employee_Count", "Employees"]
                for field in priority_fields:
                    if field in row_dict and row_dict[field]:
                        content.append(f"{field}: {row_dict[field]}")
                
                # Add remaining fields
                for header in headers:
                    if header not in priority_fields and header in row_dict:
                        content.append(f"{header}: {row_dict[header]}")
                
                # Join all content with newlines
                page_content = "\n".join(content)
                
                # Create metadata dictionary
                metadata = {
                    "source": file_path,
                    "file_type": "csv",
                    "row": row_num,
                }
                
                # Add important fields to metadata for better retrieval
                if city:
                    metadata["city"] = city
                if employee_count:
                    metadata["employee_count"] = employee_count
                
                # Create document
                doc = Document(page_content=page_content, metadata=metadata)
                documents.append(doc)
                
                # If we have employee count data, create an extra document focused on that
                if employee_count and city:
                    emp_content = f"The {city} office has {employee_count} employees. This information comes from {filename}."
                    emp_doc = Document(
                        page_content=emp_content,
                        metadata={
                            "source": file_path,
                            "file_type": "csv",
                            "content_type": "employee_count",
                            "city": city,
                            "employee_count": employee_count
                        }
                    )
                    documents.append(emp_doc)
                    logger.info(f"Created specific employee count document for {city}: {employee_count} employees")
                    
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
    
    # Return documents without further splitting for CSV files
    return documents

def load_document(file_path: str) -> List[Document]:
    """
    Load a document using the appropriate loader based on file extension
    """
    ext = os.path.splitext(file_path)[1].lower()
    if not os.path.isfile(file_path):
        logger.warning(f"File not found: {file_path}")
        return []
    
    try:
        # Special handling for CSV files (custom processing)
        if ext == ".csv":
            docs = process_csv_file(file_path, None)
            logger.info(f"Processed {file_path} with custom CSV processing: {len(docs)} documents")
            return docs
        
        # Use the appropriate loader from mapping
        if ext in LOADER_MAPPING:
            loader_class, loader_args = LOADER_MAPPING[ext]
            loader = loader_class(file_path, **loader_args)
            docs = loader.load()
            logger.info(f"Processed {file_path} with {loader_class.__name__}: {len(docs)} documents")
            return docs
        
        # Default to text loader if extension not recognized
        logger.warning(f"No specific loader for {ext}, defaulting to TextLoader")
        loader = TextLoader(file_path, encoding="utf8")
        docs = loader.load()
        logger.info(f"Processed {file_path} with TextLoader: {len(docs)} documents")
        return docs
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return []

def post_process_documents(docs: List[Document]) -> List[Document]:
    """
    Enhance documents with additional metadata and derived documents
    """
    enhanced_docs = []
    
    # Track employee counts by city to create comparison documents later
    employee_counts = {}
    
    for doc in docs:
        # Add the original document
        enhanced_docs.append(doc)
        
        # Analyze content for special data patterns
        content = doc.page_content.lower()
        source = doc.metadata.get('source', 'unknown')
        file_name = os.path.basename(source) if isinstance(source, str) else 'unknown'
        
        # Extract special data patterns and create focused documents
        # Example: Employee counts
        if any(term in content for term in ["employee", "employees", "headcount", "staff", "people"]):
            import re
            # Try to extract city and employee count patterns
            city_match = re.search(r'(?:city|office|location):\s*([a-zA-Z\s]+)', content, re.IGNORECASE)
            employee_match = re.search(r'(?:employees|headcount|staff|people):\s*(\d+)', content, re.IGNORECASE)
            
            # Check metadata if we couldn't find in content
            if not city_match and "city" in doc.metadata:
                city = doc.metadata["city"]
                city_match = True  # Fake match since we got it from metadata
            
            if not employee_match and "employee_count" in doc.metadata:
                count = doc.metadata["employee_count"]
                employee_match = True  # Fake match since we got it from metadata
            
            if city_match and employee_match:
                city = city_match.group(1).strip() if not isinstance(city_match, bool) else doc.metadata["city"]
                count = employee_match.group(1).strip() if not isinstance(employee_match, bool) else doc.metadata["employee_count"]
                
                # Store for comparison documents
                employee_counts[city] = count
                
                emp_content = f"The {city} office has {count} employees. This information comes from {file_name}."
                emp_doc = Document(
                    page_content=emp_content,
                    metadata={
                        "source": source,
                        "content_type": "employee_count",
                        "city": city,
                        "employee_count": count,
                        "derived": True
                    }
                )
                enhanced_docs.append(emp_doc)
                logger.info(f"Created derived employee count document for {city}: {count} employees")
    
    # Create comparison documents for each pair of cities
    cities = list(employee_counts.keys())
    for i in range(len(cities)):
        for j in range(i + 1, len(cities)):
            city1, city2 = cities[i], cities[j]
            count1, count2 = employee_counts[city1], employee_counts[city2]
            total = int(count1) + int(count2)
            
            comparison_content = f"The {city1} office has {count1} employees and the {city2} office has {count2} employees. " \
                               f"Together, the {city1} and {city2} offices have a total of {total} employees."
            
            comp_doc = Document(
                page_content=comparison_content,
                metadata={
                    "source": "derived_comparison",
                    "content_type": "employee_comparison",
                    "cities": f"{city1}_{city2}",
                    "total_employees": str(total)
                }
            )
            enhanced_docs.append(comp_doc)
            logger.info(f"Created comparison document for {city1} and {city2} with total of {total} employees")
    
    # Create a comprehensive employee count document if we have multiple cities
    if len(cities) > 1:
        all_cities_content = "Employee counts by city:\n"
        total = 0
        for city, count in employee_counts.items():
            all_cities_content += f"- The {city} office has {count} employees.\n"
            total += int(count)
        
        all_cities_content += f"\nThe total number of employees across all listed offices is {total}."
        
        all_doc = Document(
            page_content=all_cities_content,
            metadata={
                "source": "derived_all_cities",
                "content_type": "all_employee_counts",
                "total_employees": str(total)
            }
        )
        enhanced_docs.append(all_doc)
        logger.info(f"Created comprehensive employee count document for {len(cities)} cities with total of {total} employees")
    
    return enhanced_docs

def ingest_documents(documents_dir: str = './documents') -> int:
    """
    Ingest documents into the vector database.
    
    Args:
        documents_dir: Directory containing documents to ingest
        
    Returns:
        Number of documents ingested
    """
    logger.info(f"Ingesting documents from {documents_dir}")
    
    # Define a text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # Find all files in the documents directory
    all_files = []
    for ext in LOADER_MAPPING.keys():
        all_files.extend(glob.glob(os.path.join(documents_dir, "**", f"*{ext}"), recursive=True))
    
    # Also include files with extensions not in mapping
    all_files.extend(glob.glob(os.path.join(documents_dir, "**", "*.*"), recursive=True))
    # Remove duplicates
    all_files = list(set(all_files))
    
    if not all_files:
        logger.warning(f"No documents found in {documents_dir}")
        return 0
    
    logger.info(f"Found {len(all_files)} files")
    
    all_documents = []
    # Process each file with a progress bar
    for file_path in tqdm.tqdm(all_files, desc="Processing files"):
        docs = load_document(file_path)
        
        # Process non-CSV documents through the text splitter
        if os.path.splitext(file_path)[1].lower() != ".csv":
            docs = text_splitter.split_documents(docs)
            
        # Apply post-processing to extract special data
        docs = post_process_documents(docs)
        
        all_documents.extend(docs)
        logger.info(f"Processed {file_path}: {len(docs)} documents")
    
    # Initialize the embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Initialize ChromaDB
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Clear the collection if it exists
    try:
        client.delete_collection("document_collection")
        logger.info("Deleted existing collection")
    except:
        pass
    
    # Create a new collection
    vectorstore = Chroma.from_documents(
        documents=all_documents,
        embedding=embeddings,
        client=client,
        collection_name="document_collection"
    )
    
    logger.info(f"Ingested {len(all_documents)} documents into the vector database")
    return len(all_documents)

def main():
    """Main function."""
    try:
        num_documents = ingest_documents()
        print(f"Successfully ingested {num_documents} documents.")
    except Exception as e:
        logger.exception("Error ingesting documents")
        sys.exit(1)

if __name__ == "__main__":
    main()