import streamlit as st
# Import other modules AFTER this
import logging
import time
import pandas as pd
import traceback
from datetime import datetime
import os
import json
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from contextlib import contextmanager

# This MUST be the first st.* command in the file
st.set_page_config(
    page_title="ELP Document Intelligence System",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Document Intelligence System for real estate portfolio analysis and management."
    }
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("streamlit_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("elp_web_interface")

# Import your existing components
try:
    from reasoning import ReasoningService
    from ai_agent import llm
    import ingest_documents
    from config import Config
    logger.info("Successfully imported core components")
except Exception as e:
    logger.error(f"Error importing core components: {str(e)}")
    st.error(f"Failed to load required components: {str(e)}")

# Context manager for error handling
@contextmanager
def safe_execution(error_msg="An error occurred"):
    """Context manager for safely executing code with proper error handling"""
    try:
        yield
    except Exception as e:
        logger.error(f"{error_msg}: {str(e)}\n{traceback.format_exc()}")
        st.error(f"{error_msg}: {str(e)}")

# Initialize reasoning service with proper error handling and caching
@st.cache_resource(ttl=3600, show_spinner=False)
def get_reasoning_service():
    """Initialize and cache the reasoning service with error handling"""
    try:
        logger.info("Initializing reasoning service")
        service = ReasoningService(llm=llm, chroma_path="./chroma_db")
        logger.info("Reasoning service initialized successfully")
        return service
    except Exception as e:
        logger.error(f"Failed to initialize reasoning service: {str(e)}\n{traceback.format_exc()}")
        st.error(f"Failed to initialize reasoning service: {str(e)}")
        # Return a minimal implementation that won't crash the app
        class FallbackService:
            def process_query(self, query, detail_level="standard"):
                return {
                    "answer": f"ERROR: The reasoning service is unavailable. Please check logs or restart the application. Details: {str(e)}",
                    "reasoning": "Service initialization failed",
                    "confidence": "unknown",
                    "context_sources": []
                }
            def extract_facts(self, query):
                return []
        return FallbackService()

# Function to safely load facts with error handling
@st.cache_data(ttl=600)
def load_facts():
    """Load facts from file with proper error handling"""
    try:
        if os.path.exists("fact_memory.json"):
            with open("fact_memory.json", "r") as f:
                return json.load(f)
        return []
    except Exception as e:
        logger.error(f"Error loading facts: {str(e)}")
        return []

# Function to safely get document stats
@st.cache_data(ttl=300)
def get_document_stats():
    """Get document statistics with error handling"""
    try:
        # Count documents in the documents directory
        doc_count = sum(len(files) for _, _, files in os.walk("./documents"))
        
        # List document types
        doc_types = {}
        for root, _, files in os.walk("./documents"):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                doc_types[ext] = doc_types.get(ext, 0) + 1
        
        return {
            "count": doc_count,
            "types": doc_types
        }
    except Exception as e:
        logger.error(f"Error getting document stats: {str(e)}")
        return {"count": 0, "types": {}}

# Safely calculate directory size
@st.cache_data(ttl=600)
def get_dir_size(path='.'):
    """Safely calculate directory size with error handling"""
    try:
        if not os.path.exists(path):
            return 0
        total = 0
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_file():
                    total += entry.stat().st_size
                elif entry.is_dir():
                    total += get_dir_size(entry.path)
        return total
    except Exception as e:
        logger.error(f"Error calculating directory size: {str(e)}")
        return 0

# Initialize session state safely
def init_session_state():
    """Initialize session state variables safely"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "reasoning_enabled" not in st.session_state:
        st.session_state.reasoning_enabled = True
    if "facts" not in st.session_state:
        st.session_state.facts = []
    if "app_initialized" not in st.session_state:
        st.session_state.app_initialized = True
        logger.info("Session state initialized")

# Function to generate and cache suggested questions
@st.cache_data(ttl=1800)
def get_suggested_questions():
    """Generate and cache suggested questions based on available data"""
    try:
        # List of questions grouped by category
        return {
            "Occupancy & Planning": [
                "Which office had the highest average occupancy in April 2025?",
                "What was the week-over-week change in occupancy across all offices for the past month?",
                "Which meeting rooms in Chicago have less than 50% utilization?",
                "What's the correlation between office occupancy and day of the week across our locations?"
            ],
            "Energy Cost Analysis": [
                "Which city has the highest average energy cost?",
                "How does energy usage per employee compare across all offices?",
                "What's the correlation between occupancy rates and energy consumption in Miami?",
                "Which office has shown the most improvement in energy efficiency over the last quarter?"
            ],
            "ESG Metrics": [
                "Which office performs best environmentally?",
                "Calculate the carbon footprint per employee across all offices.",
                "How do our ESG metrics compare to industry benchmarks?",
                "What would be the environmental impact of implementing a 4-day work week in our LA office?"
            ],
            "Strategic Decisions": [
                "Which city should consolidate based on utilization?",
                "What is the impact of relocating 25% of NYC staff to Miami?",
                "Based on current trends, where should we invest in expanding office space next year?",
                "What would be the financial impact of converting 30% of NYC office space to hot-desking?"
            ],
            "Badge & Employee Data": [
                "How many employees are assigned to LA?",
                "Compare badge utilization between Chicago and NYC.",
                "Which departments have the highest and lowest office attendance rates?",
                "What percentage of employees use the office less than once per week?"
            ],
            "Facilities & Leasing": [
                "What is the lease cost per square foot in Philadelphia?",
                "What is our total leased square footage in LA?",
                "Which office has the best and worst cost-per-employee ratio?",
                "When are our lease renewal dates and what are the projected market rates at those times?"
            ]
        }
    except Exception as e:
        logger.error(f"Error generating suggested questions: {str(e)}")
        # Return a minimal set of questions that won't cause problems
        return {"General Questions": ["Which office had the highest occupancy?", 
                                     "Compare energy usage between offices", 
                                     "How many employees are in each location?"]}

# Initialize session state
init_session_state()

# Sidebar for navigation with error handling
st.sidebar.title("ELP Document Intelligence")
try:
    page = st.sidebar.radio("Navigation", ["Chat Interface", "Document Management", "Facts & Insights", "System Status"])
except Exception as e:
    logger.error(f"Error in navigation sidebar: {str(e)}")
    page = "Chat Interface"  # Default to chat interface if there's an error

# Function to display chat messages with better handling of long responses and error handling
def display_chat_messages():
    try:
        for i, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant", avatar="🏢"):
                    # Get content and check if it's potentially truncated
                    content = message.get("content", "No content available")
                    reasoning = message.get("reasoning", "")
                    is_long = len(content) > 2500
                    
                    # First display the main answer content
                    if is_long:
                        with st.expander("Complete Answer", expanded=True):
                            st.write(content)
                    else:
                        st.write(content)
                    
                    # Then display reasoning if enabled and available
                    if st.session_state.reasoning_enabled and reasoning:
                        with st.expander("Reasoning Process", expanded=False):
                            st.write(reasoning)
                    
                    # Show sources if available
                    if "sources" in message and message["sources"]:
                        with st.expander("Sources", expanded=False):
                            st.write(", ".join(message["sources"]))
    except Exception as e:
        logger.error(f"Error displaying chat messages: {str(e)}")
        st.error("Error displaying chat history. Try refreshing the page.")

# Chat Interface Page
if page == "Chat Interface":
    st.title("Document Intelligence Chat")
    
    # Chat interface with error handling
    with safe_execution("Error displaying chat interface"):
        display_chat_messages()
    
    # Toggle for reasoning
    st.sidebar.subheader("Display Options")
    with safe_execution("Error with display options"):
        st.session_state.reasoning_enabled = st.sidebar.toggle("Show reasoning steps", st.session_state.reasoning_enabled)
    
    # Add suggested questions section
    with safe_execution("Error displaying suggested questions"):
        question_categories = get_suggested_questions()
        st.sidebar.subheader("Suggested Questions")
        category = st.sidebar.selectbox("Select a category", list(question_categories.keys()))
        
        if category:
            for question in question_categories[category]:
                if st.sidebar.button(question, key=f"btn_{hash(question)}", use_container_width=True):
                    # Store the question to be used by the main flow directly
                    # instead of trying to manipulate the chat_input widget
                    if "temp_question" not in st.session_state:
                        st.session_state.temp_question = question
                    # Use rerun() instead of experimental_rerun()
                    st.rerun()

    # Input for user query with better handling of suggested questions
    if "temp_question" in st.session_state:
        # Get the question from the temp storage
        user_query = st.session_state.temp_question
        # Clear it so we don't reuse it on the next refresh
        del st.session_state.temp_question
    else:
        # Normal user input
        user_query = st.chat_input("Ask me about your office data...")

    if user_query:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_query)
        
        # Process the query with comprehensive error handling
        with st.chat_message("assistant", avatar="🏢"):
            try:
                with st.spinner("Analyzing documents..."):
                    # Get the reasoning service
                    service = get_reasoning_service()
                    
                    # Process query with detailed reasoning if enabled
                    detail_level = "detailed" if st.session_state.reasoning_enabled else "standard"
                    start_time = time.time()
                    response = service.process_query(user_query, detail_level=detail_level)
                    process_time = time.time() - start_time
                    logger.info(f"Query processed in {process_time:.2f} seconds")
                    
                    # Extract parts from response
                    answer = response.get("answer", "Sorry, I couldn't process your question.")
                    reasoning = response.get("reasoning", "")
                    confidence = response.get("confidence", "unknown")
                    sources = response.get("context_sources", [])
                
                # Display the main answer first
                if len(answer) > 2500:
                    with st.expander("Complete Answer", expanded=True):
                        st.write(answer)
                else:
                    st.write(answer)
                
                # Immediately show reasoning if available, but collapsed by default
                if st.session_state.reasoning_enabled and reasoning:
                    with st.expander("Reasoning Process", expanded=False):
                        st.write(reasoning)
                
                # Show sources if available
                if sources:
                    with st.expander("Sources", expanded=False):
                        st.write(", ".join(sources))
                
                # Add assistant message to chat history
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": answer,
                    "reasoning": reasoning,
                    "confidence": confidence,
                    "sources": sources
                })
                
                # Log processing time if it's unusually long
                if process_time > 10:
                    logger.warning(f"Long processing time ({process_time:.2f}s) for query: {user_query[:100]}...")
                    
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}\n{traceback.format_exc()}")
                error_message = f"Sorry, I encountered an error while processing your query: {str(e)}"
                st.error(error_message)
                
                # Add error response to chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": error_message,
                    "reasoning": f"Error occurred: {traceback.format_exc()}",
                    "confidence": "error",
                    "sources": []
                })

# Document Management Page
elif page == "Document Management":
    st.title("Document Management")
    
    # Display current documents with proper error handling
    st.subheader("Document Database Status")
    with safe_execution("Error loading document statistics"):
        doc_stats = get_document_stats()
        st.info(f"Documents in directory: {doc_stats['count']}")
        
        # Display document types as a bar chart
        if doc_stats["types"]:
            try:
                fig, ax = plt.subplots()
                sns.barplot(x=list(doc_stats["types"].keys()), y=list(doc_stats["types"].values()))
                ax.set_xlabel("Document Type")
                ax.set_ylabel("Count")
                ax.set_title("Document Types in Database")
                st.pyplot(fig)
            except Exception as e:
                logger.error(f"Error creating document type chart: {str(e)}")
                st.error(f"Could not create document type visualization: {str(e)}")
                # Fallback to simple text display
                st.write("Document types:")
                for ext, count in doc_stats["types"].items():
                    st.write(f"- {ext}: {count}")
    
    # Document database management with improved UX
    st.subheader("Database Management")
    col1, col2 = st.columns(2)
    
    with col1:
        refresh_btn = st.button("Refresh Document Database", type="primary", use_container_width=True)
        if refresh_btn:
            with st.spinner("Refreshing document database..."):
                with safe_execution("Error refreshing database"):
                    start_time = time.time()
                    ingest_documents.update_database_incrementally()
                    process_time = time.time() - start_time
                    st.success(f"Document database refreshed in {process_time:.2f} seconds!")
                    # Clear cached document stats to force refresh
                    get_document_stats.clear()
    
    with col2:
        rebuild_btn = st.button("Rebuild Entire Database", type="secondary", use_container_width=True)
        if rebuild_btn:
            confirm = st.checkbox("Are you sure? This will delete and rebuild the entire database.")
            if confirm:
                with st.spinner("Rebuilding document database..."):
                    with safe_execution("Error rebuilding database"):
                        # Delete existing database
                        import shutil
                        if os.path.exists("./chroma_db"):
                            shutil.rmtree("./chroma_db")
                        
                        # Rebuild
                        start_time = time.time()
                        ingest_documents.ingest_documents()
                        process_time = time.time() - start_time
                        st.success(f"Document database rebuilt in {process_time:.2f} seconds!")
                        # Clear caches to force refresh
                        get_document_stats.clear()
                        get_reasoning_service.clear()

# Facts & Insights Page
elif page == "Facts & Insights":
    st.title("Facts & Insights")
    
    # Load facts with better error handling
    with safe_execution("Error loading facts"):
        facts = load_facts()
        st.session_state.facts = facts
    
    # Display facts
    st.subheader("Extracted Facts")
    with safe_execution("Error displaying facts"):
        if st.session_state.facts:
            # Convert to DataFrame for better display
            facts_df = pd.DataFrame(st.session_state.facts)
            st.dataframe(facts_df, use_container_width=True)
        else:
            st.info("No facts have been extracted yet. Use the chat interface to extract facts from your documents.")
    
    # Manual fact extraction with improved error handling
    st.subheader("Extract Facts from Document")
    doc_query = st.text_input("Enter document name or topic for fact extraction")
    extract_btn = st.button("Extract Facts", type="primary")
    
    if extract_btn:
        if doc_query:
            with st.spinner("Extracting facts..."):
                with safe_execution("Error extracting facts"):
                    service = get_reasoning_service()
                    facts = service.extract_facts(doc_query)
                    if facts:
                        st.success(f"Extracted {len(facts)} facts!")
                        st.write(facts)
                    else:
                        st.warning("No facts were extracted. Try a different query.")
        else:
            st.warning("Please enter a document name or topic")

# System Status Page
else:  # System Status
    st.title("System Status")
    
    # Display configuration with error handling
    st.subheader("System Configuration")
    
    # Display LLM information
    with safe_execution("Error retrieving model information"):
        model_name = Config.LAMBDA_MODEL if hasattr(Config, 'LAMBDA_MODEL') else "Not specified"
        st.info(f"Primary LLM Model: {model_name}")
    
    # Memory usage with safe calculation
    st.subheader("Memory Usage")
    
    with safe_execution("Error calculating storage usage"):
        if os.path.exists("./chroma_db"):
            chroma_size = get_dir_size("./chroma_db") / (1024 * 1024)  # Convert to MB
            st.metric("Vector Database Size", f"{chroma_size:.2f} MB")
            
            # Add warning if database is getting large
            if chroma_size > 500:
                st.warning(f"Vector database is quite large ({chroma_size:.2f} MB). Consider optimizing or pruning.")
    
    # System health checks with improved display
    st.subheader("Health Checks")
    with safe_execution("Error running health checks"):
        checks = {
            "Document Database": os.path.exists("./chroma_db"),
            "LLM API Access": hasattr(Config, 'LAMBDA_API_KEY') and Config.LAMBDA_API_KEY is not None,
            "Fact Memory": os.path.exists("fact_memory.json"),
            "Documents Folder": os.path.exists("./documents")
        }
        
        # Display health checks as a colored table
        health_data = []
        for check, status in checks.items():
            status_str = "✅ Available" if status else "❌ Unavailable"
            health_data.append({"Component": check, "Status": status_str})
        
        health_df = pd.DataFrame(health_data)
        st.dataframe(health_df, use_container_width=True, hide_index=True)
        
        # Show overall health status
        if all(checks.values()):
            st.success("All systems operational")
        elif any(checks.values()):
            st.warning("Some systems are not available")
        else:
            st.error("Critical system failure - all components unavailable")
    
    # Add app execution metrics
    st.subheader("Application Metrics")
    with safe_execution("Error retrieving app metrics"):
        # Get memory usage of the Python process
        import psutil
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            st.metric("Current Memory Usage", f"{memory_mb:.1f} MB")
        except:
            st.info("Process memory information unavailable")
    
    # Add log viewer
    st.subheader("Application Logs")
    with safe_execution("Error accessing logs"):
        if os.path.exists("streamlit_app.log"):
            with open("streamlit_app.log", "r") as f:
                log_content = f.readlines()
                # Show last 10 log entries
                recent_logs = log_content[-10:] if len(log_content) > 10 else log_content
                with st.expander("Recent Logs", expanded=False):
                    for log in recent_logs:
                        st.text(log.strip())
        else:
            st.info("No application logs available")

# Main content footer with better formatting
st.sidebar.markdown("---")
st.sidebar.caption("Temporary Web Interface")
st.sidebar.caption("ELP Document Intelligence System")
st.sidebar.info("Version 1.0.1")

# Add a cache clearing option in sidebar footer
if st.sidebar.button("Clear Cache & Refresh", type="secondary", use_container_width=True):
    # Clear all st.cache_data and st.cache_resource
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.chat_history = []
    st.rerun()

# Run the Streamlit app with error handling
if __name__ == "__main__":
    try:
        logger.info("Web interface started successfully")
    except Exception as e:
        logger.error(f"Error in main app execution: {str(e)}")