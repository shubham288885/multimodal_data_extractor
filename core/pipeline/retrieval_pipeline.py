from pymilvus import Collection, connections
from ..embedding.embedder import DocumentEmbedder
from ..ranking.reranker import DocumentReranker
from ..llm.answer_generator import LLMAnswerGenerator
import os

class RetrievalPipeline:
    def __init__(self, milvus_collection_name="vector_db"):
        self.embedder = DocumentEmbedder()
        self.reranker = DocumentReranker()
        self.llm = LLMAnswerGenerator()
        
        # Connect to Milvus Cloud using URI and token
        connections.connect(
            "default",
            uri=os.getenv('MILVUS_URI'),
            token=os.getenv('MILVUS_TOKEN')
        )
        self.collection = Collection(milvus_collection_name)
        
        # Load the collection into memory
        try:
            self.collection.load()
            print(f"Collection {milvus_collection_name} loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load collection: {str(e)}")
        
    def generate_answer(self, query, results, stream=False):
        """Generate an answer to the query using the LLM"""
        if not results:
            return "No relevant documents found to answer your query."
        
        # Use the LLM to generate an answer based on the retrieved documents
        answer = self.llm.generate_answer(query, results, stream=stream)
        return answer
        
    def process_query(self, query, k=5, generate_answer=False):
        print(f"\n=== Processing query: '{query}' ===")
        
        # Try to load the collection (this is idempotent in Milvus)
        try:
            self.collection.load()
            print("Collection loaded successfully")
        except Exception as e:
            print(f"Warning: Error loading collection: {str(e)}")
        
        # Generate query embedding
        query_embedding = self.embedder.embed_text(query)
        print(f"Generated embedding of dimension: {len(query_embedding)}")
        
        # Get initial results from Milvus
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        print("Searching Milvus...")
        
        try:
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=k,
                expr=None,
                output_fields=["text", "metadata", "doc_id"]  # Request all fields we need
            )
            
            print(f"Search returned {len(results[0])} results")
            
            # Extract text from results
            documents = []
            scores = []
            metadata_list = []
            doc_ids = []
            
            for hit in results[0]:
                try:
                    # Access fields directly from the entity object
                    text = hit.entity.text
                    metadata = hit.entity.metadata
                    doc_id = hit.entity.doc_id
                    
                    print(f"Hit: doc_id={doc_id}, distance={hit.distance}")
                    print(f"Text snippet: {text[:100]}...")
                    
                    documents.append(text)
                    scores.append(hit.distance)
                    metadata_list.append(metadata)
                    doc_ids.append(doc_id)
                except Exception as e:
                    print(f"Error processing hit: {str(e)}")
                    print(f"Hit entity fields: {dir(hit.entity)}")
            
            # Rerank results if we have documents
            if documents:
                print(f"Reranking {len(documents)} documents...")
                try:
                    reranked_results = self.reranker.rerank(
                        query=query,
                        documents=documents,
                        scores=scores
                    )
                    
                    print(f"Reranking returned {len(reranked_results)} results")
                    
                    # If reranker returned no results, create results from original documents
                    if not reranked_results:
                        print("Creating results from original documents")
                        reranked_results = [
                            {"text": doc, "score": score} 
                            for doc, score in zip(documents, scores)
                        ]
                    
                    # Add metadata and doc_id to reranked results
                    for i, result in enumerate(reranked_results):
                        if i < len(metadata_list):
                            result["metadata"] = metadata_list[i]
                        if i < len(doc_ids):
                            result["doc_id"] = doc_ids[i]
                    
                    # Print final results
                    print("\n=== Final Results ===")
                    for i, result in enumerate(reranked_results):
                        print(f"Result {i+1}:")
                        print(f"  Score: {result.get('score', 0.0)}")
                        print(f"  Doc ID: {result.get('doc_id', 'unknown')}")
                        print(f"  Text: {result.get('text', '')[:150]}...")
                        print()
                    
                    # After getting the reranked results, generate an answer if requested
                    if generate_answer and reranked_results:
                        print("Generating answer with LLM...")
                        answer = self.generate_answer(query, reranked_results)
                        return {
                            "results": reranked_results,
                            "answer": answer
                        }
                    
                    return reranked_results
                except Exception as e:
                    print(f"Error during reranking: {str(e)}")
                    # Fall back to original results if reranking fails
                    results = [{"text": doc, "score": score, "metadata": meta, "doc_id": doc_id} 
                            for doc, score, meta, doc_id in zip(documents, scores, metadata_list, doc_ids)]
                    
                    # Print fallback results
                    print("\n=== Fallback Results (No Reranking) ===")
                    for i, result in enumerate(results):
                        print(f"Result {i+1}:")
                        print(f"  Score: {result.get('score', 0.0)}")
                        print(f"  Doc ID: {result.get('doc_id', 'unknown')}")
                        print(f"  Text: {result.get('text', '')[:150]}...")
                        print()
                    
                    return results
            else:
                print("No documents found in vector database")
                return []
        except Exception as e:
            print(f"Error during search: {str(e)}")
            return [] 