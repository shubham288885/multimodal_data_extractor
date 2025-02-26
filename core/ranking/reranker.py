import requests
import os

class DocumentReranker:
    def __init__(self):
        self.api_key = os.getenv('NVIDIA_RERANK_KEY')
        self.rerank_endpoint = os.getenv('NVIDIA_RERANK_ENDPOINT')
        self.session = requests.Session()
        
    def rerank(self, query, documents, scores=None):
        """Rerank documents using NVIDIA llama-3.2-nv-rerankqa-1b-v2"""
        print(f"Reranking {len(documents)} documents with query: '{query[:50]}...'")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
        }
        
        payload = {
            "model": "nvidia/llama-3.2-nv-rerankqa-1b-v2",
            "query": {"text": query},
            "passages": [{"text": doc} for doc in documents]
        }
        
        try:
            response = self.session.post(
                self.rerank_endpoint,
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            # Process the response to return a list of dictionaries
            response_data = response.json()
            print(f"Reranker API response: {response_data}")
            
            # Format the results as a list of dictionaries
            formatted_results = []
            if "passages" in response_data:
                for passage in response_data["passages"]:
                    formatted_results.append({
                        "text": passage.get("text", ""),
                        "score": passage.get("score", 0.0)
                    })
            
            # If no results from reranker, fall back to original documents
            if not formatted_results and documents:
                print("No results from reranker, falling back to original documents")
                for i, doc in enumerate(documents):
                    score = scores[i] if scores and i < len(scores) else 0.0
                    formatted_results.append({
                        "text": doc,
                        "score": score
                    })
            
            return formatted_results
        except Exception as e:
            print(f"Reranking API request failed: {str(e)}")
            # Fall back to original documents in case of error
            return [{"text": doc, "score": scores[i] if scores and i < len(scores) else 0.0} 
                    for i, doc in enumerate(documents)] 