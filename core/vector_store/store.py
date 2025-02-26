import os
import numpy as np
from typing import List, Dict
from pymilvus import Collection, connections

class VectorStore:
    def __init__(self, collection_name="vector_db", dimension=1024):
        self.dimension = dimension
        self.collection_name = collection_name
        
        # Connect to Milvus Cloud
        try:
            connections.connect(
                "default",
                uri=os.getenv('MILVUS_URI'),
                token=os.getenv('MILVUS_TOKEN')
            )
            print(f"Connected to Milvus Cloud for {collection_name}")
            
            # Get or create collection
            if self._collection_exists():
                self.collection = Collection(collection_name)
                print(f"Using existing collection: {collection_name}")
            else:
                print(f"Collection {collection_name} does not exist. Please run setup_milvus.py first.")
                
            # Load collection into memory
            try:
                self.collection.load()
            except Exception as e:
                print(f"Warning: Could not load collection: {str(e)}")
                
        except Exception as e:
            print(f"Error connecting to Milvus: {str(e)}")
            
    def _collection_exists(self):
        """Check if collection exists in Milvus"""
        from pymilvus import utility
        return utility.has_collection(self.collection_name)
            
    def add_documents(self, documents: List[Dict], embeddings: np.ndarray, metadata_list=None):
        """Add documents and their embeddings to the store"""
        if len(documents) != embeddings.shape[0]:
            raise ValueError("Number of documents must match number of embeddings")
            
        # Format data for Milvus
        doc_ids = [f"doc_{i}" for i in range(len(documents))]
        texts = [doc.get('text', '') for doc in documents]
        
        # Use provided metadata or empty dicts
        if metadata_list is None:
            metadata_list = [{} for _ in documents]
            
        # Insert into Milvus
        insert_result = self.collection.insert([
            doc_ids,                # doc_id field
            texts,                  # text field
            embeddings.tolist(),    # embedding field
            metadata_list           # metadata field
        ])
        
        # Flush to ensure data is persisted
        self.collection.flush()
        
        return insert_result.insert_count
        
    def search(self, query_embedding: np.ndarray, k: int = 5):
        """Search for similar documents"""
        # Ensure collection is loaded
        try:
            self.collection.load()
        except Exception as e:
            print(f"Warning: Error loading collection: {str(e)}")
            
        # Search parameters
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        
        # Perform search
        results = self.collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=k,
            output_fields=["text", "metadata", "doc_id"]
        )
        
        # Format results
        formatted_results = []
        for hit in results[0]:
            try:
                formatted_results.append({
                    'text': hit.entity.text,
                    'metadata': hit.entity.metadata,
                    'doc_id': hit.entity.doc_id,
                    'score': hit.distance
                })
            except Exception as e:
                print(f"Error processing hit: {str(e)}")
                
        return formatted_results
    
    def save(self, path: str):
        """Save the index to disk"""
        # Milvus is a cloud service, so we don't need to save the index locally
        # This method is kept for API compatibility
        print(f"Note: Milvus collections are stored in the cloud. No local save needed.")
        return True
        
    def load(self, path: str):
        """Load the index from disk"""
        # Milvus is a cloud service, so we don't need to load the index from disk
        # This method is kept for API compatibility
        print(f"Note: Milvus collections are loaded from the cloud. No local load needed.")
        # Ensure collection is loaded in memory
        try:
            self.collection.load()
            print(f"Collection {self.collection_name} loaded into memory")
        except Exception as e:
            print(f"Warning: Error loading collection: {str(e)}") 