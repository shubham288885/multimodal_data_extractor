from openai import OpenAI
import os
import tiktoken

class DocumentEmbedder:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv('NVIDIA_EMBEDDING_KEY'),
            base_url=os.getenv('NVIDIA_EMBEDDING_ENDPOINT')
        )
        self.max_tokens = 512  # Maximum tokens allowed by the API
        self.overlap = 50      # Token overlap between chunks
        # Initialize tokenizer - using cl100k_base which is used by many embedding models
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
    def embed_text(self, text):
        """Generate embeddings using NVIDIA embedding model"""
        try:
            if not text:
                raise ValueError("Input text must not be empty.")
                
            # Check if text needs truncation
            tokens = self.tokenizer.encode(text)
            if len(tokens) > self.max_tokens:
                # Truncate to max_tokens
                truncated_tokens = tokens[:self.max_tokens]
                truncated_text = self.tokenizer.decode(truncated_tokens)
                print(f"Text truncated from {len(tokens)} to {len(truncated_tokens)} tokens")
                
                response = self.client.embeddings.create(
                    input=[truncated_text],
                    model="nvidia/nv-embedqa-e5-v5",
                    encoding_format="float",
                    extra_body={"input_type": "query", "truncate": "END"}
                )
            else:
                response = self.client.embeddings.create(
                    input=[text],
                    model="nvidia/nv-embedqa-e5-v5",
                    encoding_format="float",
                    extra_body={"input_type": "query", "truncate": "END"}
                )
            return response.data[0].embedding
        except Exception as e:
            if hasattr(e, 'response') and e.response is not None:
                print(f"Error response: {e.response.text}")
            raise Exception(f"Embedding API request failed: {str(e)}")
    
    def embed_batch(self, texts):
        """Generate embeddings for a batch of texts"""
        try:
            if not texts or any(not text for text in texts):
                raise ValueError("Input texts must not be empty.")
            
            all_embeddings = []
            # Process in smaller batches
            batch_size = 5  # Reduced batch size
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                truncated_batch = []
                
                # Truncate any texts that exceed the token limit
                for text in batch_texts:
                    tokens = self.tokenizer.encode(text)
                    if len(tokens) > self.max_tokens:
                        # Truncate to max_tokens
                        truncated_tokens = tokens[:self.max_tokens]
                        truncated_text = self.tokenizer.decode(truncated_tokens)
                        print(f"Text truncated from {len(tokens)} to {len(truncated_tokens)} tokens")
                        truncated_batch.append(truncated_text)
                    else:
                        truncated_batch.append(text)
                
                # Print token counts for debugging
                for idx, text in enumerate(truncated_batch):
                    token_count = len(self.tokenizer.encode(text))
                    print(f"Text {idx} token count: {token_count}")
                
                response = self.client.embeddings.create(
                    input=truncated_batch,
                    model="nvidia/nv-embedqa-e5-v5",
                    encoding_format="float",
                    extra_body={"input_type": "query", "truncate": "END"}
                )
                all_embeddings.extend([data.embedding for data in response.data])
                
            return all_embeddings
        except Exception as e:
            if hasattr(e, 'response') and e.response is not None:
                print(f"Error response: {e.response.text}")
            raise Exception(f"Batch embedding request failed: {str(e)}")
    
    def count_tokens(self, text):
        """Count the number of tokens in the text using tiktoken"""
        return len(self.tokenizer.encode(text))
    
    def chunk_text(self, text, chunk_size=None):
        """Split text into chunks with a specific token count"""
        if chunk_size is None:
            chunk_size = self.max_tokens
            
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        # Create chunks with overlap
        for i in range(0, len(tokens), chunk_size - self.overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            # If we've processed all tokens, break
            if i + chunk_size >= len(tokens):
                break
                
        return chunks 

    def embed_batch_with_metadata(self, texts, metadata=None):
        """Generate embeddings for a batch of texts and return with metadata"""
        try:
            if not texts or any(not text for text in texts):
                raise ValueError("Input texts must not be empty.")
            
            # Generate embeddings
            embeddings = self.embed_batch(texts)
            
            # Create IDs (simple sequential IDs for now)
            ids = list(range(len(texts)))
            
            # Create metadata if not provided
            if metadata is None:
                metadata = [{"original_text": text} for text in texts]
            elif len(metadata) != len(texts):
                raise ValueError("Number of metadata items must match number of texts")
            
            return {
                "ids": ids,
                "embeddings": embeddings,
                "texts": texts,
                "metadata": metadata
            }
        except Exception as e:
            if hasattr(e, 'response') and e.response is not None:
                print(f"Error response: {e.response.text}")
            raise Exception(f"Batch embedding with metadata failed: {str(e)}") 