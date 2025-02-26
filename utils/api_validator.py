import os
from dotenv import load_dotenv

def validate_api_keys():
    """Validate that all required API keys are present"""
    required_keys = [
        'NVIDIA_YOLOX_KEY',
        'NVIDIA_DEPLOT_KEY',
        'NVIDIA_EMBEDDING_KEY',
        'NVIDIA_PADDLEOCR_KEY',
        'NVIDIA_CACHED_KEY',
        'NVIDIA_RERANK_KEY',
        'NVIDIA_LLM_KEY'
    ]
    
    missing_keys = []
    for key in required_keys:
        if not os.getenv(key):
            missing_keys.append(key)
    
    if missing_keys:
        raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")

def validate_endpoints():
    """Validate that all required endpoints are configured"""
    required_endpoints = [
        'NVIDIA_YOLOX_ENDPOINT',
        'NVIDIA_DEPLOT_ENDPOINT',
        'NVIDIA_EMBEDDING_ENDPOINT',
        'NVIDIA_PADDLEOCR_ENDPOINT',
        'NVIDIA_CACHED_ENDPOINT',
        'NVIDIA_RERANK_ENDPOINT',
        'NVIDIA_LLM_ENDPOINT'
    ]
    
    missing_endpoints = []
    for endpoint in required_endpoints:
        if not os.getenv(endpoint):
            missing_endpoints.append(endpoint)
    
    if missing_endpoints:
        raise ValueError(f"Missing required endpoints: {', '.join(missing_endpoints)}") 