from core.document_processor.extractor import DocumentExtractor
from core.document_processor.ocr import OCRProcessor
from core.embedding.embedder import DocumentEmbedder
from pymilvus import Collection, connections
import os

class IngestionPipeline:
    def __init__(self):
        self.extractor = DocumentExtractor()
        self.ocr = OCRProcessor()
        self.embedder = DocumentEmbedder()
        
    def process_document(self, pdf_path):
        """Process a document through the complete pipeline"""
        try:
            print(f"Processing document: {pdf_path}")
            
            # Extract content from PDF
            extracted_content = self.extractor.extract_from_pdf(pdf_path)
            print(f"Extracted {len(extracted_content['text'])} text segments, {len(extracted_content['tables'])} tables, and {len(extracted_content['charts'])} charts")
            
            # Process tables with OCR if any found
            for table in extracted_content['tables']:
                table_text = self.ocr.process_table(table['image'])
                table['structured_data'] = table_text
            
            # Generate embeddings for text segments
            texts = []
            metadata = []
            for text_segment in extracted_content['text']:
                # Skip empty text segments
                if not text_segment['content'] or len(text_segment['content'].strip()) < 10:
                    continue
                    
                texts.append(text_segment['content'])
                # Add page number and position as metadata
                metadata.append({
                    "page_num": text_segment.get('page_num', 0),
                    "position": text_segment.get('position', {}),
                    "document_path": pdf_path
                })
            
            print(f"Generating embeddings for {len(texts)} text segments")
            
            # Generate embeddings with metadata
            if texts:
                embedding_results = self.embedder.embed_batch_with_metadata(texts, metadata)
                
                # Store in Milvus
                inserted_count = self._store_in_milvus(embedding_results)
                print(f"Stored {inserted_count} text segments in vector database")
            else:
                print("No valid text segments found to embed")
            
            return extracted_content
            
        except Exception as e:
            print(f"Document processing error: {str(e)}")
            raise Exception(f"Document processing failed: {str(e)}")
            
    def _store_in_milvus(self, embedding_results):
        """Store text and embeddings in Milvus"""
        try:
            # Connect to Milvus if not already connected
            try:
                connections.connect(
                    "default",
                    uri=os.getenv('MILVUS_URI'),
                    token=os.getenv('MILVUS_TOKEN')
                )
                print("Connected to Milvus for document storage")
            except Exception as e:
                print(f"Connection already exists or error: {str(e)}")
            
            collection = Collection("vector_db")
            
            # Debug information
            print(f"Storing {len(embedding_results['texts'])} text segments in Milvus")
            print(f"First text: {embedding_results['texts'][0][:100]}...")
            print(f"First metadata: {embedding_results['metadata'][0]}")
            
            # Insert data in the correct format for Milvus
            # Note: 'id' field is auto-generated, so we don't provide it
            insert_result = collection.insert([
                ["doc_" + str(id) for id in embedding_results["ids"]],  # doc_id field
                embedding_results["texts"],                            # text field
                embedding_results["embeddings"],                       # embedding field
                embedding_results["metadata"]                          # metadata field
            ])
            
            # Flush to ensure data is persisted
            collection.flush()
            
            # Print confirmation
            print(f"Successfully inserted {insert_result.insert_count} records into Milvus")
            print(f"IDs: {insert_result.primary_keys[:5]}...")
            
            return insert_result.insert_count
        except Exception as e:
            print(f"Error storing in Milvus: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            raise Exception(f"Failed to store document in vector database: {str(e)}") 