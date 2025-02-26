from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from dotenv import load_dotenv
import os
import random

# Load environment variables
load_dotenv()

def setup_milvus():
    print("=== Start connecting to Milvus Cloud ===")
    
    # Connect to Milvus Cloud using URI and token
    connections.connect(
        alias="default",
        uri=os.getenv('MILVUS_URI'),
        token=os.getenv('MILVUS_TOKEN')
    )
    
    collection_name = "vector_db"
    
    # Check if collection exists
    print(f"\nDoes collection {collection_name} exist in Milvus: {utility.has_collection(collection_name)}")

    # Drop collection if it exists
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    # Define fields
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=256),  # Links to Document Store
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),  # Original Text Content
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),  # NVIDIA E5 Embeddings
        FieldSchema(name="metadata", dtype=DataType.JSON)  # Any extra metadata
    ]

    # Define schema
    schema = CollectionSchema(fields, description="Vector DB for document embeddings")

    # Create collection
    collection = Collection("vector_db", schema)

    # Create IVF_FLAT index for vector field
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    collection.create_index("embedding", index_params)
        
    return collection

def test_milvus_connection():
    try:
        # Create collection
        collection = setup_milvus()
        
        # Insert some test data
        entities = [
            ["This is a test document 1", [random.random() for _ in range(1024)], {"source": "test"}],
            ["This is a test document 2", [random.random() for _ in range(1024)], {"source": "test"}]
        ]
        
        collection.insert([
            [entity[0] for entity in entities],  # text
            [entity[1] for entity in entities],  # embeddings
            [entity[2] for entity in entities]   # metadata
        ])
        
        # Load collection
        collection.load()
        
        # Perform a test search
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }
        
        results = collection.search(
            data=[[random.random() for _ in range(1024)]],  # Query vector
            anns_field="embedding",
            param=search_params,
            limit=2,
            output_fields=["text", "metadata"]
        )
        
        print("\n=== Test search results ===")
        for hits in results:
            for hit in hits:
                print(f"hit: (distance: {hit.distance}, text: {hit.entity.get('text')})")
        
        print("\nMilvus Cloud setup and test completed successfully!")
        
    except Exception as e:
        print(f"Error during Milvus setup: {e}")
    finally:
        # Clean up
        connections.disconnect("default")

if __name__ == "__main__":
    test_milvus_connection() 