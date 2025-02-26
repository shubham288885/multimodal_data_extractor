import streamlit as st
import sys
import os
from pathlib import Path
import tempfile
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.pipeline.ingestion_pipeline import IngestionPipeline
from core.pipeline.retrieval_pipeline import RetrievalPipeline
from dotenv import load_dotenv
from utils.api_validator import validate_api_keys, validate_endpoints

# Load environment variables
load_dotenv()
validate_api_keys()
validate_endpoints()

class DocumentSearchApp:
    def __init__(self):
        st.set_page_config(
            page_title="Document Search System",
            page_icon="🔍",
            layout="wide"
        )
        
        # Initialize pipelines
        self.ingestion = IngestionPipeline()
        self.retrieval = RetrievalPipeline("vector_db")
        
        # Initialize session state
        if 'processed_documents' not in st.session_state:
            st.session_state.processed_documents = []
        if 'search_history' not in st.session_state:
            st.session_state.search_history = []

    def render_sidebar(self):
        with st.sidebar:
            st.title("📚 Navigation")
            st.markdown("---")
            st.markdown("### Processed Documents")
            if st.session_state.processed_documents:
                for doc in st.session_state.processed_documents:
                    st.markdown(f"- {doc}")
            else:
                st.markdown("*No documents processed yet*")
            
            st.markdown("---")
            st.markdown("### Search History")
            if st.session_state.search_history:
                for query in st.session_state.search_history:
                    st.markdown(f"- {query}")
            else:
                st.markdown("*No search history*")

    def process_document(self, uploaded_file):
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            # Process document through pipeline
            with st.spinner('Processing document...'):
                result = self.ingestion.process_document(tmp_path)
                
                # Update UI with results
                st.success(f"Successfully processed {uploaded_file.name}")
                st.markdown("### Document Statistics")
                st.markdown(f"- Text segments: {len(result['text'])}")
                st.markdown(f"- Tables found: {len(result['tables'])}")
                st.markdown(f"- Charts detected: {len(result['charts'])}")
                
            return True
            
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
            return False

    def search(self, query, k=5):
        """Search for documents related to the query"""
        if not query:
            return []
            
        # Process the query through the retrieval pipeline
        # Set generate_answer=True to use the LLM
        results = self.retrieval.process_query(query, k=k, generate_answer=True)
        return results

    def render_main_interface(self):
        st.title("🔍 Document Search System")
        st.markdown("Upload PDF documents and search through their contents using natural language queries.")
        
        # Document Upload Section
        st.markdown("## 📤 Document Upload")
        uploaded_files = st.file_uploader(
            "Upload PDF documents", 
            type=['pdf'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if self.process_document(uploaded_file):
                    st.markdown("---")
        
        # Search Section
        self.search_documents()

    def search_documents(self):
        st.markdown("## Search Documents")
        
        # Search query input
        query = st.text_input("Enter your search query")
        
        if st.button("Search") or query:
            if query:
                with st.spinner("Searching..."):
                    results = self.search(query)
                
                # Check if results is a dictionary with 'answer' key (from LLM)
                if isinstance(results, dict) and 'answer' in results:
                    # Display the LLM-generated answer
                    st.subheader("Answer")
                    st.write(results['answer'])
                    
                    # Display the retrieved documents
                    st.subheader("Retrieved Documents")
                    for i, doc in enumerate(results['results']):
                        with st.expander(f"Document {i+1} (Score: {doc.get('score', 0.0):.4f})"):
                            # Display metadata if available
                            if 'metadata' in doc and doc['metadata']:
                                meta = doc['metadata']
                                source = meta.get('document_path', 'Unknown source')
                                page = meta.get('page_num', 'Unknown page')
                                st.caption(f"Source: {source}, Page: {page}")
                            
                            # Display text content
                            st.write(doc.get('text', 'No text available'))
                else:
                    # Handle case where no LLM answer was generated
                    st.warning("No answer could be generated. Here are the retrieved documents:")
                    
                    for i, doc in enumerate(results):
                        with st.expander(f"Document {i+1} (Score: {doc.get('score', 0.0):.4f})"):
                            # Display metadata if available
                            if 'metadata' in doc and doc['metadata']:
                                meta = doc['metadata']
                                source = meta.get('document_path', 'Unknown source')
                                page = meta.get('page_num', 'Unknown page')
                                st.caption(f"Source: {source}, Page: {page}")
                            
                            # Display text content
                            st.write(doc.get('text', 'No text available'))
            else:
                st.info("Please enter a query to search.")

    def run(self):
        self.render_sidebar()
        self.render_main_interface()

if __name__ == "__main__":
    app = DocumentSearchApp()
    app.run() 