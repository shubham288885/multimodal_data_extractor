from .ingestion_pipeline import IngestionPipeline
from .retrieval_pipeline import RetrievalPipeline
from ..emission.emissions_calculator import EmissionsCalculator
from ..document_processor.extractor import DocumentExtractor
from typing import Dict, Any, List
import os

class EmissionsPipeline:
    """
    Pipeline for processing documents and calculating greenhouse gas emissions
    
    This pipeline combines document extraction with emissions calculation to
    provide a comprehensive emissions analysis for uploaded documents.
    """
    
    def __init__(self, milvus_collection_name="vector_db"):
        """
        Initialize the emissions pipeline
        
        Args:
            milvus_collection_name: Name of the Milvus collection to use
        """
        self.ingestion = IngestionPipeline()
        self.retrieval = RetrievalPipeline(milvus_collection_name)
        self.emissions_calculator = EmissionsCalculator()
        self.extractor = DocumentExtractor()  # Direct access to extractor for full text
        
    def process_document_for_emissions(self, document_path: str) -> Dict[str, Any]:
        """
        Process a document and calculate emissions based on its content
        
        Args:
            document_path: Path to the document
            
        Returns:
            Dictionary with extracted content, activities, and emissions calculations
        """
        # Process the document through the ingestion pipeline
        extraction_result = self.ingestion.process_document(document_path)
        print(f"Document processed with {len(extraction_result['text'])} text segments")
        
        # IMPORTANT: Get the FULL document text directly from the extractor to avoid truncation
        doc = self.extractor.extract_from_pdf(document_path)
        
        # Get document content for analysis
        document_content = []
        
        # Add text segments to document content - USING FULL TEXT
        for text_segment in doc['text']:
            if not text_segment['content'] or len(text_segment['content'].strip()) < 10:
                continue
                
            document_content.append({
                'text': text_segment['content'],  # This is the full, non-truncated text
                'metadata': {
                    'document_path': document_path,
                    'page_num': text_segment.get('page_num', 0),
                    'type': 'text'
                }
            })
        
        # Add table content if available
        for i, table in enumerate(extraction_result['tables']):
            if 'structured_data' in table and table['structured_data']:
                document_content.append({
                    'text': f"Table data: {table['structured_data']}",
                    'metadata': {
                        'document_path': document_path,
                        'type': 'table',
                        'table_index': i
                    }
                })
        
        # Add chart content if available
        for i, chart in enumerate(extraction_result['charts']):
            if 'data' in chart and chart['data']:
                document_content.append({
                    'text': f"Chart data: {chart['data']}",
                    'metadata': {
                        'document_path': document_path,
                        'type': 'chart',
                        'chart_index': i
                    }
                })
        
        print(f"Extracting emission-relevant activities from {len(document_content)} content segments")
        print(f"Content size (characters): {sum(len(segment['text']) for segment in document_content)}")
        
        # Extract emission-relevant activities from the document
        activities = self.emissions_calculator.extract_activities(document_content)
        print(f"Extracted {len(activities)} emission-relevant activities")
        
        if not activities:
            return {
                'document_path': document_path,
                'extraction_result': extraction_result,
                'activities': [],
                'emissions_calculation': {
                    'error': 'No emission-relevant activities found in the document'
                }
            }
        
        print(f"Calculating emissions for {len(activities)} activities")
        
        # Calculate emissions for the extracted activities
        emissions_calculation = self.emissions_calculator.calculate_emissions(activities)
        
        # Return the complete results
        return {
            'document_path': document_path,
            'extraction_result': extraction_result,
            'activities': activities,
            'emissions_calculation': emissions_calculation
        }
    
    def get_document_summary_for_emissions(self, document_path: str) -> Dict[str, Any]:
        """
        Get a summary of a document focused on emissions-relevant information
        
        Args:
            document_path: Path to the document
            
        Returns:
            Dictionary with document summary and key emissions information
        """
        # First extract the document content
        extraction_result = self.ingestion.process_document(document_path)
        
        # Generate a summary query focused on emissions
        query = "Extract information related to greenhouse gas emissions, energy usage, transportation, material consumption, and waste generation."
        
        # Process the query to get relevant document sections
        results = self.retrieval.process_query(query, k=5, generate_answer=True)
        
        # Get the generated summary
        if isinstance(results, dict) and 'answer' in results:
            summary = results['answer']
            relevant_sections = results['results']
        else:
            # If no answer was generated, create a simple summary
            summary = "No emissions-relevant information could be automatically extracted."
            relevant_sections = results if isinstance(results, list) else []
        
        return {
            'document_path': document_path,
            'summary': summary,
            'relevant_sections': relevant_sections
        } 