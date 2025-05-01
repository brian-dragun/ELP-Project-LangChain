#!/usr/bin/env python3
import os
import sys
import json
import glob
import hashlib
import time
import threading
from qa_system import llm, logger, prompt
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentChat:
    def __init__(self, auto_refresh=True):
        self.history = []
        self.documents_path = os.path.join(os.path.dirname(__file__), 'documents')
        self.chroma_path = os.path.join(os.path.dirname(__file__), 'chroma_db')
        self.hash_file = os.path.join(os.path.dirname(__file__), 'documents_hash.txt')
        self.client = chromadb.PersistentClient(path=self.chroma_path)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.monitor_running = False
        self.last_check_time = time.time()
        
        # Check if documents have changed and refresh if needed
        if auto_refresh and self._documents_changed():
            self._refresh_documents()
        
        self.vectorstore = Chroma(
            client=self.client,
            collection_name="document_collection",
            embedding_function=self.embeddings
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # Start background document monitoring thread
        if auto_refresh:
            self._start_document_monitor()
        
        # Welcome message
        print("\n" + "="*50)
        print("🤖 Document-Aware Chat System")
        print("="*50)
        print("Ask questions about your office data across Chicago, LA, Miami, NYC, and Philadelphia.")
        print("Type 'exit' or 'quit' to end the session.")
        print("Type 'help' for sample questions.")
        print("Type 'clear' to clear chat history.")
        print("Type 'refresh' to manually refresh the document database.")
        print("Documents folder will be automatically monitored for changes.")
        print("="*50 + "\n")

    def _start_document_monitor(self):
        """Start a background thread to monitor document changes"""
        self.monitor_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_documents, daemon=True)
        self.monitor_thread.start()
        
    def _monitor_documents(self):
        """Monitor the documents folder for changes in background"""
        check_interval = 10  # Check every 10 seconds
        
        while self.monitor_running:
            time.sleep(check_interval)
            
            # Check if documents have changed
            if self._documents_changed():
                current_time = time.time()
                # Only refresh if it's been at least 60 seconds since last refresh
                # to avoid excessive refreshes when multiple files are being added
                if current_time - self.last_check_time > 60:
                    print("\nNew documents detected! Refreshing database...")
                    self._refresh_documents()
                    self.last_check_time = current_time
                    print("\n👤 You: ", end="")  # Restore the prompt
                    sys.stdout.flush()  # Ensure the prompt is displayed

    def _get_documents_hash(self):
        """Generate a hash of all documents in the documents folder"""
        files = sorted(glob.glob(f"{self.documents_path}/*.*"))
        hash_content = ""
        
        for file_path in files:
            hash_content += file_path
            if os.path.exists(file_path):
                hash_content += str(os.path.getmtime(file_path))
        
        return hashlib.md5(hash_content.encode()).hexdigest()

    def _documents_changed(self):
        """Check if documents have been added, modified, or removed"""
        current_hash = self._get_documents_hash()
        
        # If hash file exists, compare with stored hash
        if os.path.exists(self.hash_file):
            with open(self.hash_file, 'r') as f:
                stored_hash = f.read().strip()
                return current_hash != stored_hash
        
        return True  # If hash file doesn't exist, consider it changed

    def _save_documents_hash(self):
        """Save the current hash of documents"""
        current_hash = self._get_documents_hash()
        with open(self.hash_file, 'w') as f:
            f.write(current_hash)

    def _refresh_documents(self):
        """Re-ingest all documents in the documents folder"""
        print("Refreshing document database...")
        
        try:
            # Configure document loading
            loader = DirectoryLoader(
                self.documents_path,
                glob="**/*.csv"
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
            
            # Remove existing collection if it exists
            try:
                self.client.delete_collection(name="document_collection")
            except:
                pass
            
            # Create vector store with new documents
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                client=self.client,
                collection_name="document_collection"
            )
            
            # Update vector store and retriever references
            self.vectorstore = vectorstore
            self.retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            
            # Save the new document hash
            self._save_documents_hash()
            
            print("Document refresh complete!")
            return True
            
        except Exception as e:
            print(f"Error refreshing documents: {str(e)}")
            logger.exception("Error refreshing documents")
            return False

    def get_relevant_context(self, query):
        """Retrieve relevant document snippets for the query"""
        docs = self.retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in docs])
    
    def format_prompt_with_history(self, query, context):
        """Format the prompt with chat history and document context"""
        # Create a history string from past interactions
        history_str = ""
        if self.history:
            history_str = "Previous conversation:\n"
            for q, a in self.history[-3:]:  # Include up to last 3 interactions
                history_str += f"User: {q}\nAssistant: {a}\n\n"
        
        # Format the prompt with history and context
        full_prompt = f"""Answer the following question based on the provided context and our previous conversation.

Current date: April 30, 2025

{history_str}
Context:
{context}

Question: {query}

Provide a clear and detailed answer using only the information from the context. 
If the answer cannot be determined from the context, say "I don't have enough information to answer this question."

Answer:"""
        return full_prompt

    def process_query(self, query):
        """Process a user query and return an answer"""
        # Get relevant context
        context = self.get_relevant_context(query)
        
        # Format prompt with history and context
        full_prompt = self.format_prompt_with_history(query, context)
        
        # Generate response using LLM
        try:
            response = llm.invoke(full_prompt)
            self.history.append((query, response))
            return response
        except Exception as e:
            logger.exception("Error generating response")
            return f"Error processing your question: {str(e)}"

    def show_help(self):
        """Show sample questions the user can ask"""
        help_text = """
Sample questions you can ask:
-----------------------------
1. "How many employees work in the Los Angeles office?"
2. "What is the badge utilization rate in Chicago compared to NYC?"
3. "Which city has the highest meeting room utilization on weekends?"
4. "What is the LEED certification level for the Chicago office?"
5. "When does the Los Angeles office lease end?"
6. "What is the waste recycling rate in Philadelphia?"
7. "Which office has the highest energy consumption per seat?"
8. "Compare the office space cost per square foot across all cities."

For more sample questions, check SAMPLE_QUESTIONS.md
"""
        print(help_text)

    def save_history(self):
        """Save chat history to a file"""
        history_file = "chat_history.json"
        try:
            with open(history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
            print(f"Chat history saved to {history_file}")
        except Exception as e:
            print(f"Error saving chat history: {e}")

    def run_session(self):
        """Run an interactive chat session"""
        try:
            while True:
                # Get user input
                user_input = input("\n👤 You: ")
                
                # Check for exit commands
                if user_input.lower() in ['exit', 'quit', 'q']:
                    self.monitor_running = False  # Stop the monitoring thread
                    self.save_history()
                    print("Goodbye! Chat history has been saved.")
                    break
                
                # Check for help command
                elif user_input.lower() == 'help':
                    self.show_help()
                    continue
                
                # Check for clear history command
                elif user_input.lower() == 'clear':
                    self.history = []
                    print("Chat history cleared.")
                    continue
                
                # Check for refresh command
                elif user_input.lower() == 'refresh':
                    print("Manually refreshing document database...")
                    self._refresh_documents()
                    continue
                
                # Process regular queries
                if user_input.strip():
                    print("\n🤖 Assistant: ", end="")
                    response = self.process_query(user_input)
                    print(response)
                    print("\n" + "-"*50)
                
        except KeyboardInterrupt:
            self.monitor_running = False  # Stop the monitoring thread
            print("\nSession ended by user.")
            self.save_history()
        except Exception as e:
            self.monitor_running = False  # Stop the monitoring thread
            print(f"An error occurred: {e}")
            self.save_history()

if __name__ == "__main__":
    # Check for command line arguments
    refresh_flag = any(arg.lower() in ['--refresh', '-r'] for arg in sys.argv[1:])
    
    chat = DocumentChat(auto_refresh=True)
    chat.run_session()