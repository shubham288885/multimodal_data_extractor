import streamlit as st
import sys
import os
from pathlib import Path
import tempfile
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.pipeline.ingestion_pipeline import IngestionPipeline
from core.pipeline.retrieval_pipeline import RetrievalPipeline
from core.pipeline.emissions_pipeline import EmissionsPipeline
from dotenv import load_dotenv
from utils.api_validator import validate_api_keys, validate_endpoints

# Load environment variables
load_dotenv()
validate_api_keys()
validate_endpoints()

class DocumentSearchApp:
    def __init__(self):
        st.set_page_config(
            page_title="Document Search & Emissions Calculator",
            page_icon="🔍",
            layout="wide"
        )
        
        # Initialize pipelines
        self.ingestion = IngestionPipeline()
        self.retrieval = RetrievalPipeline("vector_db")
        self.emissions = EmissionsPipeline("vector_db")
        
        # Initialize session state
        if 'processed_documents' not in st.session_state:
            st.session_state.processed_documents = []
        if 'search_history' not in st.session_state:
            st.session_state.search_history = []
        if 'current_tab' not in st.session_state:
            st.session_state.current_tab = "search"
        if 'emissions_results' not in st.session_state:
            st.session_state.emissions_results = {}

    def render_sidebar(self):
        with st.sidebar:
            st.title("📚 Navigation")
            
            # Tabs
            tabs = {
                "search": "🔍 Document Search",
                "emissions": "🌍 Emissions Calculator"
            }
            
            # Create tab buttons
            st.markdown("### Main Features")
            for tab_id, tab_name in tabs.items():
                if st.button(tab_name, key=f"tab_{tab_id}"):
                    st.session_state.current_tab = tab_id
            
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
                
                # Add to processed documents list
                if uploaded_file.name not in st.session_state.processed_documents:
                    st.session_state.processed_documents.append(uploaded_file.name)
                
            return True
            
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
            return False

    def search(self, query, k=5):
        """Search for documents related to the query"""
        if not query:
            return []
            
        # Add to search history
        if query not in st.session_state.search_history:
            st.session_state.search_history.append(query)
            
        # Process the query through the retrieval pipeline
        # Set generate_answer=True to use the LLM
        results = self.retrieval.process_query(query, k=k, generate_answer=True)
        return results

    def calculate_emissions(self, uploaded_file):
        """Calculate emissions based on the document content"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            # Process document for emissions
            with st.spinner('Analyzing document for emissions...'):
                result = self.emissions.process_document_for_emissions(tmp_path)
                
                # Store emissions results
                st.session_state.emissions_results[uploaded_file.name] = result
                
                # Add to processed documents list if not already there
                if uploaded_file.name not in st.session_state.processed_documents:
                    st.session_state.processed_documents.append(uploaded_file.name)
                
                return result
                
        except Exception as e:
            st.error(f"Error calculating emissions: {str(e)}")
            return None

    def render_main_interface(self):
        st.title("🔍 Document Search & Emissions Calculator")
        
        # Display different content based on the current tab
        if st.session_state.current_tab == "search":
            self.render_search_tab()
        elif st.session_state.current_tab == "emissions":
            self.render_emissions_tab()

    def render_search_tab(self):
        st.markdown("Upload PDF documents and search through their contents using natural language queries.")
        
        # Document Upload Section
        st.markdown("## 📤 Document Upload")
        uploaded_files = st.file_uploader(
            "Upload PDF documents for search", 
            type=['pdf'],
            accept_multiple_files=True,
            key="search_upload"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if self.process_document(uploaded_file):
                    st.markdown("---")
        
        # Search Section
        self.search_documents()

    def render_emissions_tab(self):
        st.markdown("Upload bills and documents to calculate greenhouse gas emissions (Scope 1, 2, and 3).")
        
        # Document Upload Section
        st.markdown("## 📤 Document Upload for Emissions Analysis")
        uploaded_file = st.file_uploader(
            "Upload a PDF document for emissions calculation", 
            type=['pdf'],
            key="emissions_upload"
        )
        
        if uploaded_file:
            # Check if we have already calculated emissions for this file
            if uploaded_file.name in st.session_state.emissions_results:
                st.info(f"Emissions analysis for {uploaded_file.name} already available. Showing results.")
                emission_result = st.session_state.emissions_results[uploaded_file.name]
            else:
                # Calculate emissions for the new file
                emission_result = self.calculate_emissions(uploaded_file)
            
            if emission_result:
                self.display_emissions_results(emission_result)

    def display_emissions_results(self, results):
        st.markdown("## 🌍 Emissions Analysis Results")
        
        # Display activities extracted from the document
        activities = results.get('activities', [])
        if activities:
            st.subheader("Emission-Relevant Activities Identified")
            activities_table = []
            
            for activity in activities:
                # Format the details for display
                details = activity.get('details', {})
                details_str = ", ".join([f"{k}: {v}" for k, v in details.items()])
                
                activities_table.append({
                    "Activity": activity['description'],
                    "Details": details_str
                })
            
            st.table(activities_table)
        else:
            st.warning("No emission-relevant activities were identified in this document.")
        
        # Display emissions calculation
        emissions = results.get('emissions_calculation', {})
        if 'error' in emissions:
            st.error(emissions['error'])
        else:
            st.subheader("Emissions Calculation")
            
            # If there's a total emission value, show it prominently
            if 'total_scope_3_emissions' in emissions:
                total = emissions['total_scope_3_emissions']
                st.metric("Total Scope 3 Emissions (kg CO2e)", f"{total}")
            
            # Display all emission sources
            if 'emission_sources' in emissions:
                for source in emissions['emission_sources']:
                    source_name = source.get('source', 'Unknown source')
                    source_total = source.get('total_emissions', 'N/A')
                    
                    with st.expander(f"{source_name.title()} - {source_total} kg CO2e"):
                        if 'processes' in source:
                            for process in source['processes']:
                                st.markdown(f"**{process.get('name', 'Unknown process').replace('_', ' ').title()}**")
                                st.markdown(f"*{process.get('description', '')}*")
                                
                                if 'parameters' in process:
                                    params = process['parameters']
                                    st.markdown(f"Quantity: {params.get('quantity', 'N/A')}")
                                    st.markdown(f"Emission Factor: {params.get('emission_factor', 'N/A')}")
                                    st.markdown(f"Calculation: {params.get('calculation', 'N/A')}")
                                    st.markdown(f"Total Emissions: {params.get('total_emissions', 'N/A')} kg CO2e")
            
            # Display assumptions if any
            if 'assumptions' in emissions and emissions['assumptions']:
                st.subheader("Assumptions")
                for assumption in emissions['assumptions']:
                    st.markdown(f"- {assumption}")
            
            # Display data sources if any
            if 'data_sources' in emissions and emissions['data_sources']:
                st.subheader("Data Sources")
                for source in emissions['data_sources']:
                    st.markdown(f"- {source}")
            
            # Provide download option for the full results
            st.download_button(
                label="Download Full Results (JSON)",
                data=json.dumps(emissions, indent=2),
                file_name="emissions_calculation.json",
                mime="application/json"
            )

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