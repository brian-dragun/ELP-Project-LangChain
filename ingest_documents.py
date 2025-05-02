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
    Process a CSV file by grouping data by city to reduce document count.
    Creates comprehensive city-based documents rather than one document per row.
    """
    logger.info(f"Processing CSV file: {file_path}")
    filename = os.path.basename(file_path)
    documents = []
    
    try:
        # Read the entire CSV file into a pandas DataFrame for better processing
        import pandas as pd
        df = pd.read_csv(file_path)
        
        # Get the headers
        headers = df.columns.tolist()
        
        # Case where the CSV contains city information - group by city
        if 'City' in df.columns:
            # Group data by city to create comprehensive city documents
            for city, city_data in df.groupby('City'):
                # Create one comprehensive document for each city
                content_parts = [f"Information about {city} from {filename}:"]
                
                # Add a summary header
                metrics = []
                if 'NumberOfEmployees' in city_data:
                    employees = city_data['NumberOfEmployees'].iloc[0]
                    metrics.append(f"{employees} employees")
                if 'BadgeUtilizationRate%' in city_data:
                    badge_rate = city_data['BadgeUtilizationRate%'].iloc[0]
                    metrics.append(f"{badge_rate}% badge utilization")
                if 'EnergyConsumptionkWhPerSeatPerYear' in city_data:
                    energy = city_data['EnergyConsumptionkWhPerSeatPerYear'].iloc[0]
                    metrics.append(f"{energy} kWh/seat/year energy consumption")
                    
                if metrics:
                    content_parts.append(f"Summary: {city} office has {', '.join(metrics)}.")
                
                # Convert the data to a string representation with clear formatting
                city_data_string = city_data.to_string(index=False)
                content_parts.append(f"Raw data:\n{city_data_string}")
                
                # Combine all parts into one content string
                page_content = "\n\n".join(content_parts)
                
                # Create metadata
                metadata = {
                    "source": file_path,
                    "file_type": "csv",
                    "city": city,
                    "document_type": "city_comprehensive"
                }
                
                # Add important metrics to metadata if available
                if 'NumberOfEmployees' in city_data:
                    metadata["employee_count"] = str(city_data['NumberOfEmployees'].iloc[0])
                
                # Create the comprehensive city document
                doc = Document(page_content=page_content, metadata=metadata)
                documents.append(doc)
                logger.info(f"Created comprehensive document for {city} from {filename}")
                
                # Also create a specialized employee count document if that data exists
                if 'NumberOfEmployees' in city_data:
                    employee_count = city_data['NumberOfEmployees'].iloc[0]
                    emp_content = f"The {city} office has {employee_count} employees. This information comes from {filename}."
                    emp_doc = Document(
                        page_content=emp_content,
                        metadata={
                            "source": file_path,
                            "file_type": "csv",
                            "content_type": "employee_count",
                            "city": city,
                            "employee_count": str(employee_count)
                        }
                    )
                    documents.append(emp_doc)
        
        # If there's no City column, create a document for the whole CSV
        else:
            # Create a document for the whole file
            content = f"Data from {filename}:\n\n"
            content += df.to_string(index=False)
            
            doc = Document(
                page_content=content,
                metadata={
                    "source": file_path,
                    "file_type": "csv",
                    "document_type": "full_csv"
                }
            )
            documents.append(doc)
            logger.info(f"Created document for entire CSV file {filename}")
        
        # Create a summary document with key statistics
        summary_content = f"Summary of {filename}:\n"
        summary_content += f"- Contains {len(df)} rows and {len(df.columns)} columns\n"
        summary_content += f"- Columns: {', '.join(df.columns)}\n"
        
        # Add summary statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if df[col].notna().any():  # Only if there are non-NA values
                summary_content += f"- {col}: avg={df[col].mean():.2f}, min={df[col].min()}, max={df[col].max()}\n"
        
        summary_doc = Document(
            page_content=summary_content,
            metadata={
                "source": file_path,
                "file_type": "csv",
                "document_type": "summary"
            }
        )
        documents.append(summary_doc)
        
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
    Enhance documents with additional metadata and derived documents.
    More efficient implementation that avoids duplicate processing.
    """
    enhanced_docs = []
    
    # Track employee counts by city to create comparison documents later
    employee_counts = {}
    city_doc_exists = set()  # Track city documents we've already created
    
    for doc in docs:
        # Add the original document
        enhanced_docs.append(doc)
        
        # Skip post-processing for documents that are already specialized
        if doc.metadata.get("content_type") in ["employee_count", "city_comprehensive", "summary"]:
            # For city comprehensive docs, record employee count data if available
            if doc.metadata.get("document_type") == "city_comprehensive" and "employee_count" in doc.metadata:
                city = doc.metadata.get("city")
                count = doc.metadata.get("employee_count")
                if city and count:
                    employee_counts[city] = count
                    city_doc_exists.add(city)
            continue
            
        # For other docs, analyze content for special data patterns
        content = doc.page_content.lower()
        source = doc.metadata.get('source', 'unknown')
        file_name = os.path.basename(source) if isinstance(source, str) else 'unknown'
        
        # Extract employee count data if not already processed
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
                
                # Only store if we haven't already processed this city
                if city not in city_doc_exists:
                    # Store for comparison documents
                    employee_counts[city] = count
                    
                    # Create a specialized employee count document
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
                    city_doc_exists.add(city)
    
    # Create comparison documents for each pair of cities (only once per pair)
    cities = sorted(list(employee_counts.keys()))  # Sort to ensure consistent pairing
    
    # One comprehensive document with ALL cities
    if len(cities) > 1:
        all_cities_content = "Employee counts by city:\n"
        total = 0
        for city in cities:
            count = employee_counts[city]
            all_cities_content += f"- The {city} office has {count} employees.\n"
            total += int(count)
        
        all_cities_content += f"\nThe total number of employees across all listed offices is {total}."
        
        all_doc = Document(
            page_content=all_cities_content,
            metadata={
                "source": "derived_all_cities",
                "content_type": "all_employee_counts",
                "total_employees": str(total),
                "cities": ",".join(cities)
            }
        )
        enhanced_docs.append(all_doc)
        logger.info(f"Created comprehensive employee count document for {len(cities)} cities with total of {total} employees")
        
        # Create binary comparison documents (useful for specific queries)
        for i in range(len(cities)):
            for j in range(i + 1, len(cities)):
                city1, city2 = cities[i], cities[j]
                count1, count2 = employee_counts[city1], employee_counts[city2]
                total_pair = int(count1) + int(count2)
                
                comparison_content = f"The {city1} office has {count1} employees and the {city2} office has {count2} employees. " \
                                   f"Together, the {city1} and {city2} offices have a total of {total_pair} employees."
                
                comp_doc = Document(
                    page_content=comparison_content,
                    metadata={
                        "source": "derived_comparison",
                        "content_type": "employee_comparison",
                        "cities": f"{city1}_{city2}",
                        "city1": city1,
                        "city2": city2,
                        "count1": count1,
                        "count2": count2,
                        "total_employees": str(total_pair)
                    }
                )
                enhanced_docs.append(comp_doc)
    
    return enhanced_docs

def ingest_documents(documents_dir: str = './documents', show_progress: bool = True, verbose: bool = True) -> int:
    """
    Ingest documents into the vector database.
    
    Args:
        documents_dir: Directory containing documents to ingest
        show_progress: Whether to display a progress bar (set to False when called programmatically)
        verbose: Whether to display detailed logging information
        
    Returns:
        Number of documents ingested
    """
    if verbose:
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
    
    if verbose:
        logger.info(f"Found {len(all_files)} files")
    
    all_documents = []
    # Process each file with a progress bar
    file_iterator = tqdm.tqdm(all_files, desc="Processing files") if show_progress else all_files
    for file_path in file_iterator:
        docs = load_document(file_path)
        
        # Process non-CSV documents through the text splitter
        if os.path.splitext(file_path)[1].lower() != ".csv":
            docs = text_splitter.split_documents(docs)
            
        # Apply post-processing to extract special data
        docs = post_process_documents(docs)
        
        all_documents.extend(docs)
        if verbose:
            logger.info(f"Processed {file_path}: {len(docs)} documents")
    
    # Initialize the embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Initialize ChromaDB
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Clear the collection if it exists
    try:
        client.delete_collection("document_collection")
        if verbose:
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
    
    if verbose:
        logger.info(f"Ingested {len(all_documents)} documents into the vector database")
    return len(all_documents)

def ingest_file(file_path: str, chroma_path: str = "./chroma_db") -> int:
    """
    Ingest a single file into the vector database.
    
    Args:
        file_path: Path to the file to ingest
        chroma_path: Path to the ChromaDB directory
        
    Returns:
        Number of documents ingested
    """
    if not os.path.exists(file_path):
        logger.warning(f"File not found: {file_path}")
        return 0
    
    logger.info(f"Ingesting single file: {file_path}")
    
    # Define a text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # Process the file
    docs = load_document(file_path)
    
    # Process non-CSV documents through the text splitter
    if os.path.splitext(file_path)[1].lower() != ".csv":
        docs = text_splitter.split_documents(docs)
        
    # Apply post-processing to extract special data
    docs = post_process_documents(docs)
    
    if not docs:
        logger.warning(f"No documents created from {file_path}")
        return 0
        
    logger.info(f"Processed {file_path}: {len(docs)} documents")
    
    # Initialize the embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=chroma_path)
    
    # Get existing collection or create new one
    try:
        collection = client.get_collection("document_collection")
        
        # Add documents to existing collection
        vectorstore = Chroma(
            client=client,
            collection_name="document_collection",
            embedding_function=embeddings
        )
        
        # First, delete any existing documents with the same source
        vectorstore.delete(
            where={"source": file_path}
        )
        
        # Add new documents
        vectorstore.add_documents(docs)
        
    except Exception:
        # Create a new collection if it doesn't exist
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            client=client,
            collection_name="document_collection"
        )
    
    logger.info(f"Ingested {len(docs)} documents into the vector database from {file_path}")
    return len(docs)

def check_document_changes(documents_dir: str = './documents', hash_file: str = './documents_hash.txt') -> Dict[str, str]:
    """
    Check if any documents have been added, modified, or removed.
    Uses both modification time and content hashing for more reliable change detection.
    
    Args:
        documents_dir: Directory containing documents to check
        hash_file: Path to the file containing document hashes
        
    Returns:
        A dictionary with file paths as keys and status as values ('added', 'modified', 'unchanged', 'deleted')
    """
    import hashlib
    
    # Find all files in the documents directory
    all_files = []
    for ext in LOADER_MAPPING.keys():
        all_files.extend(glob.glob(os.path.join(documents_dir, "**", f"*{ext}"), recursive=True))
    
    # Also include files with extensions not in mapping
    all_files.extend(glob.glob(os.path.join(documents_dir, "**", "*.*"), recursive=True))
    # Remove duplicates and filter out non-files
    all_files = [f for f in set(all_files) if os.path.isfile(f)]
    
    # Get current hash for each file (both mtime and content hash)
    current_hashes = {}
    for file_path in all_files:
        try:
            # Get modification time
            mtime = str(os.path.getmtime(file_path))
            
            # Get content hash (first 8KB only for performance)
            with open(file_path, 'rb') as f:
                content = f.read(8192)
                content_hash = hashlib.md5(content).hexdigest()
                
            # Combine both hashes
            combined_hash = f"{mtime}:{content_hash}"
            current_hashes[file_path] = combined_hash
        except (IOError, OSError) as e:
            # If we can't read the file for some reason, just use mtime
            logger.warning(f"Could not read file {file_path} for hashing: {e}")
            current_hashes[file_path] = str(os.path.getmtime(file_path))
    
    # Load previous hashes if they exist
    previous_hashes = {}
    previous_files = set()
    
    if os.path.exists(hash_file):
        with open(hash_file, 'r') as f:
            for line in f:
                if '=' in line:  # New format with combined hash
                    path, combined_hash = line.strip().split('=', 1)
                    previous_hashes[path] = combined_hash
                    previous_files.add(path)
                elif ':' in line:  # Old format with just mtime
                    path, mtime = line.strip().split(':', 1)
                    previous_hashes[path] = mtime
                    previous_files.add(path)
    
    # Compare hashes to detect changes
    changes = {}
    
    # Check for new or modified files
    for file_path in all_files:
        if file_path not in previous_hashes:
            changes[file_path] = 'added'
        elif previous_hashes[file_path] != current_hashes.get(file_path):
            changes[file_path] = 'modified'
        else:
            changes[file_path] = 'unchanged'
    
    # Check for deleted files
    for file_path in previous_files:
        if file_path not in all_files:
            changes[file_path] = 'deleted'
    
    return changes

def save_document_hashes(documents_dir: str = './documents', hash_file: str = './documents_hash.txt'):
    """
    Save the current hash of documents using both modification time and content hashing.
    
    Args:
        documents_dir: Directory containing documents
        hash_file: Path to the file to store document hashes
    """
    import hashlib
    
    # Find all files in the documents directory
    all_files = []
    for ext in LOADER_MAPPING.keys():
        all_files.extend(glob.glob(os.path.join(documents_dir, "**", f"*{ext}"), recursive=True))
    
    # Also include files with extensions not in mapping
    all_files.extend(glob.glob(os.path.join(documents_dir, "**", "*.*"), recursive=True))
    # Remove duplicates and filter out non-files
    all_files = [f for f in set(all_files) if os.path.isfile(f)]
    
    with open(hash_file, 'w') as f:
        for file_path in all_files:
            if os.path.exists(file_path):
                try:
                    # Get modification time
                    mtime = str(os.path.getmtime(file_path))
                    
                    # Get content hash (first 8KB only for performance)
                    with open(file_path, 'rb') as file:
                        content = file.read(8192)
                        content_hash = hashlib.md5(content).hexdigest()
                    
                    # Store combined hash in new format
                    f.write(f"{file_path}={mtime}:{content_hash}\n")
                except (IOError, OSError) as e:
                    # Fall back to mtime-only if there's an error
                    logger.warning(f"Could not hash file contents for {file_path}: {e}")
                    f.write(f"{file_path}={os.path.getmtime(file_path)}:00000000\n")

def update_database_incrementally(documents_dir: str = './documents', chroma_path: str = './chroma_db', 
                          hash_file: str = './documents_hash.txt', show_progress: bool = True, verbose: bool = True) -> Dict[str, int]:
    """
    Update the vector database incrementally by only processing files that have changed.
    
    Args:
        documents_dir: Directory containing documents
        chroma_path: Path to the ChromaDB database
        hash_file: Path to the file containing document hashes
        show_progress: Whether to display a progress bar
        verbose: Whether to display detailed logging information
        
    Returns:
        Dictionary with counts of files processed by status
    """
    if verbose:
        logger.info(f"Checking for document changes in {documents_dir}")
    
    # Check if ChromaDB exists
    db_exists = os.path.exists(chroma_path) and os.path.isdir(chroma_path)
    if not db_exists:
        if verbose:
            logger.info("No existing database found, performing full ingestion")
        num_docs = ingest_documents(documents_dir, show_progress=show_progress, verbose=verbose)
        save_document_hashes(documents_dir, hash_file)
        return {'full_rebuild': 1, 'added': 0, 'modified': 0, 'deleted': 0, 'unchanged': 0}
    
    # Check document changes
    changes = check_document_changes(documents_dir, hash_file)
    
    # Count changes by type
    change_counts = {'added': 0, 'modified': 0, 'deleted': 0, 'unchanged': 0}
    for status in change_counts.keys():
        change_counts[status] = sum(1 for file_status in changes.values() if file_status == status)
    
    if verbose:
        logger.info(f"Document changes: {change_counts}")
    
    # If nothing changed, we're done
    if change_counts['added'] == 0 and change_counts['modified'] == 0 and change_counts['deleted'] == 0:
        if verbose:
            logger.info("No document changes detected, database is up to date")
        return change_counts
    
    # Define a text splitter for document processing
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # Initialize embeddings and ChromaDB
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=chroma_path)
    
    try:
        # Get existing collection
        collection = client.get_collection("document_collection")
        vectorstore = Chroma(
            client=client,
            collection_name="document_collection",
            embedding_function=embeddings
        )
        
        # Process changed files with a progress bar if requested
        changed_files = [path for path, status in changes.items() 
                        if status in ('added', 'modified')]
        
        if changed_files:
            file_iterator = tqdm.tqdm(changed_files, desc="Processing changed files") if show_progress else changed_files
            
            for file_path in file_iterator:
                # Skip files that don't exist (should not happen, but just in case)
                if not os.path.exists(file_path):
                    continue
                
                status = changes.get(file_path)
                if verbose:
                    logger.info(f"Processing {status} file: {file_path}")
                
                # Process the file
                docs = load_document(file_path)
                
                # Process non-CSV documents through the text splitter
                if os.path.splitext(file_path)[1].lower() != ".csv":
                    docs = text_splitter.split_documents(docs)
                
                # Apply post-processing to extract special data
                docs = post_process_documents(docs)
                
                if docs:
                    # Delete existing documents with the same source
                    vectorstore.delete(
                        where={"source": file_path}
                    )
                    
                    # Add new documents
                    vectorstore.add_documents(docs)
                    
                    if verbose:
                        logger.info(f"Updated database with {len(docs)} documents from {file_path}")
        
        # Handle deleted files
        deleted_files = [path for path, status in changes.items() if status == 'deleted']
        if deleted_files:
            if verbose:
                logger.info(f"Removing {len(deleted_files)} deleted files from the database")
            
            for file_path in deleted_files:
                vectorstore.delete(
                    where={"source": file_path}
                )
                if verbose:
                    logger.info(f"Removed documents from {file_path}")
        
        # Also handle documents that were derived from the files we're updating
        if changed_files or deleted_files:
            # Remove any documents that refer to deleted cities
            for filepath in deleted_files:
                filename = os.path.basename(filepath)
                vectorstore.delete(
                    where={"source": {"$contains": filename}}
                )
            
            # For employee count changes, update the special documents
            if any(".csv" in file_path.lower() and changes[file_path] in ('added', 'modified') for file_path in changed_files):
                # Remove all derived comparison documents
                vectorstore.delete(
                    where={"source": "derived_comparison"}
                )
                vectorstore.delete(
                    where={"source": "derived_all_cities"}
                )
                
                # Create new derived documents
                if verbose:
                    logger.info("Regenerating cross-reference documents")
                
                # Get all employee count documents
                city_docs = []
                for file_path in glob.glob(os.path.join(documents_dir, "*.csv")):
                    if os.path.exists(file_path):
                        docs = load_document(file_path)
                        city_docs.extend([doc for doc in docs if "city" in doc.metadata and "employee_count" in doc.metadata])
                
                # Generate new cross-reference documents
                cross_ref_docs = post_process_documents(city_docs)
                
                # Filter to just the derived documents
                derived_docs = [doc for doc in cross_ref_docs 
                               if doc.metadata.get("source") in ("derived_comparison", "derived_all_cities")]
                
                if derived_docs:
                    vectorstore.add_documents(derived_docs)
                    if verbose:
                        logger.info(f"Added {len(derived_docs)} derived cross-reference documents")
        
        # Save updated hashes
        save_document_hashes(documents_dir, hash_file)
        
        if verbose:
            logger.info("Incremental database update completed successfully")
        
        return change_counts
    
    except Exception as e:
        # If anything goes wrong with incremental update, fall back to full rebuild
        if verbose:
            logger.error(f"Error during incremental update: {str(e)}")
            logger.info("Falling back to full database rebuild")
        
        num_docs = ingest_documents(documents_dir, show_progress=show_progress, verbose=verbose)
        save_document_hashes(documents_dir, hash_file)
        return {'full_rebuild': 1, 'added': 0, 'modified': 0, 'deleted': 0, 'unchanged': 0}

def main():
    """Main function for command line use."""
    try:
        # Parse command line arguments if needed
        import argparse
        parser = argparse.ArgumentParser(description='Ingest documents into vector database')
        parser.add_argument('--dir', type=str, default='./documents', help='Directory containing documents to ingest')
        parser.add_argument('--file', type=str, help='Single file to ingest')
        parser.add_argument('--no-progress', action='store_true', help='Hide progress bar')
        parser.add_argument('--incremental', action='store_true', help='Perform incremental update')
        args = parser.parse_args()
        
        if args.incremental:
            change_counts = update_database_incrementally(args.dir, show_progress=not args.no_progress)
            print(f"Incremental update completed. Changes: {change_counts}")
        elif args.file:
            num_documents = ingest_file(args.file)
            print(f"Successfully ingested {num_documents} documents from {args.file}.")
            save_document_hashes()
        else:
            num_documents = ingest_documents(args.dir, show_progress=not args.no_progress)
            print(f"Successfully ingested {num_documents} documents from {args.dir}.")
            save_document_hashes()
    except Exception as e:
        logger.exception("Error ingesting documents")
        sys.exit(1)

if __name__ == "__main__":
    main()