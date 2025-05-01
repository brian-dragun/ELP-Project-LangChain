from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import os
import chromadb
from langchain_chroma import Chroma

# Configure document loading
# Update this to match your document types - using the appropriate loader
loader = DirectoryLoader(
    './documents',  # Path to your documents folder
    glob="**/*.csv",  # Load specifically CSV files
)
documents = loader.load()
print(f"Loaded {len(documents)} documents")

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(documents)
print(f"Split into {len(splits)} chunks")

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Configure and create Chroma client explicitly
client = chromadb.PersistentClient(path="./chroma_db")

# Create vector store with the explicit client
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    client=client,
    collection_name="document_collection"
)
print(f"Created vector store at ./chroma_db with collection name: document_collection")