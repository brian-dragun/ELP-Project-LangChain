import streamlit as st
import os
import json
import time
import pandas as pd
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

# Import your existing components
from reasoning import ReasoningService
from ai_agent import llm
import ingest_documents
from config import Config

# Initialize reasoning service
@st.cache_resource
def get_reasoning_service():
    """Initialize and cache the reasoning service"""
    return ReasoningService(llm=llm, chroma_path="./chroma_db")

# Page configuration
st.set_page_config(
    page_title="ELP Document Intelligence System", 
    page_icon="🏢",
    layout="wide"
)

# Sidebar for navigation
st.sidebar.title("ELP Document Intelligence")
page = st.sidebar.radio("Navigation", ["Chat Interface", "Document Management", "Facts & Insights", "System Status"])

# Initialize session state variables if they don't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "reasoning_enabled" not in st.session_state:
    st.session_state.reasoning_enabled = True  # Changed from False to True to enable by default
if "facts" not in st.session_state:
    st.session_state.facts = []

# Function to display chat messages with better handling of long responses
def display_chat_messages():
    for i, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant", avatar="🏢"):
                # Get content and check if it's potentially truncated
                content = message["content"]
                is_long = len(content) > 2500
                
                # Display reasoning if enabled
                if "reasoning" in message and st.session_state.reasoning_enabled:
                    with st.expander("Reasoning"):
                        st.write(message["reasoning"])
                
                # Display the content with option to expand if long
                if is_long:
                    with st.expander("Show complete response", expanded=True):
                        st.write(content)
                else:
                    st.write(content)
                
                # Show sources if available
                if "sources" in message and message["sources"]:
                    with st.expander("Sources"):
                        st.write(", ".join(message["sources"]))

# Chat Interface Page
if page == "Chat Interface":
    st.title("Document Intelligence Chat")
    
    # Chat interface
    display_chat_messages()
    
    # Toggle for reasoning
    st.sidebar.subheader("Display Options")
    st.session_state.reasoning_enabled = st.sidebar.toggle("Show reasoning steps", st.session_state.reasoning_enabled)
    
    # Input for user query
    user_query = st.chat_input("Ask me about your office data...")
    
    if user_query:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_query)
        
        # Process the query
        service = get_reasoning_service()
        
        # Show processing indicator
        with st.chat_message("assistant", avatar="🏢"):
            with st.spinner("Analyzing documents..."):
                # Process query with detailed reasoning if enabled
                detail_level = "detailed" if st.session_state.reasoning_enabled else "standard"
                response = service.process_query(user_query, detail_level=detail_level)
                
                # Extract parts from response
                answer = response.get("answer", "Sorry, I couldn't process your question.")
                reasoning = response.get("reasoning", "")
                confidence = response.get("confidence", "unknown")
                sources = response.get("context_sources", [])
            
            # Display response with expander if it's long
            if len(answer) > 2500:
                with st.expander("Complete Answer", expanded=True):
                    st.write(answer)
            else:
                st.write(answer)
            
            # Show sources if available
            if sources:
                with st.expander("Sources"):
                    st.write(", ".join(sources))
            
            # Add assistant message to chat history
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": answer,
                "reasoning": reasoning,
                "confidence": confidence,
                "sources": sources
            })

# Document Management Page
elif page == "Document Management":
    st.title("Document Management")
    
    # Display current documents
    st.subheader("Document Database Status")
    try:
        # Count documents in the documents directory
        doc_count = sum(len(files) for _, _, files in os.walk("./documents"))
        st.info(f"Documents in directory: {doc_count}")
        
        # List document types
        doc_types = {}
        for root, _, files in os.walk("./documents"):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                doc_types[ext] = doc_types.get(ext, 0) + 1
        
        # Display document types as a bar chart
        if doc_types:
            fig, ax = plt.subplots()
            sns.barplot(x=list(doc_types.keys()), y=list(doc_types.values()))
            ax.set_xlabel("Document Type")
            ax.set_ylabel("Count")
            ax.set_title("Document Types in Database")
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Error loading document stats: {str(e)}")
    
    # Document database management
    st.subheader("Database Management")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Refresh Document Database", type="primary"):
            with st.spinner("Refreshing document database..."):
                try:
                    ingest_documents.update_database_incrementally()
                    st.success("Document database refreshed!")
                except Exception as e:
                    st.error(f"Error refreshing database: {str(e)}")
    
    with col2:
        if st.button("Rebuild Entire Database", type="secondary"):
            confirm = st.checkbox("Are you sure? This will delete and rebuild the entire database.")
            if confirm:
                with st.spinner("Rebuilding document database..."):
                    try:
                        # Delete existing database
                        import shutil
                        if os.path.exists("./chroma_db"):
                            shutil.rmtree("./chroma_db")
                        
                        # Rebuild
                        ingest_documents.ingest_documents()
                        st.success("Document database rebuilt!")
                    except Exception as e:
                        st.error(f"Error rebuilding database: {str(e)}")

# Facts & Insights Page
elif page == "Facts & Insights":
    st.title("Facts & Insights")
    
    # Load facts if available
    try:
        if os.path.exists("fact_memory.json"):
            with open("fact_memory.json", "r") as f:
                facts = json.load(f)
                st.session_state.facts = facts
    except Exception as e:
        st.error(f"Error loading facts: {str(e)}")
    
    # Display facts
    st.subheader("Extracted Facts")
    if st.session_state.facts:
        # Convert to DataFrame for better display
        facts_df = pd.DataFrame(st.session_state.facts)
        st.dataframe(facts_df, use_container_width=True)
    else:
        st.info("No facts have been extracted yet. Use the chat interface to extract facts from your documents.")
    
    # Manual fact extraction
    st.subheader("Extract Facts from Document")
    doc_query = st.text_input("Enter document name or topic for fact extraction")
    if st.button("Extract Facts"):
        if doc_query:
            with st.spinner("Extracting facts..."):
                service = get_reasoning_service()
                # This is a placeholder - you'll need to implement fact extraction if not already available
                # Assuming your reasoning service has a extract_facts method
                facts = service.extract_facts(doc_query)
                st.success(f"Extracted {len(facts)} facts!")
                st.write(facts)
        else:
            st.warning("Please enter a document name or topic")

# System Status Page
else:  # System Status
    st.title("System Status")
    
    # Display configuration
    st.subheader("System Configuration")
    
    # Display LLM information
    st.info(f"Primary LLM Model: {Config.LAMBDA_MODEL if hasattr(Config, 'LAMBDA_MODEL') else 'Not specified'}")
    
    # Memory usage
    st.subheader("Memory Usage")
    
    # Calculate approximate memory usage of chroma_db
    def get_dir_size(path='.'):
        total = 0
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_file():
                    total += entry.stat().st_size
                elif entry.is_dir():
                    total += get_dir_size(entry.path)
        return total
    
    if os.path.exists("./chroma_db"):
        chroma_size = get_dir_size("./chroma_db") / (1024 * 1024)  # Convert to MB
        st.metric("Vector Database Size", f"{chroma_size:.2f} MB")
    
    # Load system statistics
    st.subheader("API Usage Statistics")
    # Add your API usage statistics here if available
    
    # System health checks
    st.subheader("Health Checks")
    checks = {
        "Document Database": os.path.exists("./chroma_db"),
        "LLM API Access": hasattr(Config, 'LAMBDA_API_KEY') and Config.LAMBDA_API_KEY is not None,
        "Fact Memory": os.path.exists("fact_memory.json")
    }
    
    for check, status in checks.items():
        if status:
            st.success(f"✅ {check} is available")
        else:
            st.error(f"❌ {check} is not available")

# Main content footer
st.sidebar.markdown("---")
st.sidebar.caption("Temporary Web Interface")
st.sidebar.caption("ELP Document Intelligence System")

# Run the Streamlit app
if __name__ == "__main__":
    # The code above defines the app
    pass