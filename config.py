import torch


class Config:
    """Step 2: Configuration - Centralized configuration management"""

    # Model configurations
    LLM_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # Text splitting parameters
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100

    # Retrieval parameters
    RETRIEVAL_K = 5  # Number of documents to retrieve

    # LLM parameters
    MAX_NEW_TOKENS = 512
    TEMPERATURE = 0.3
    TOP_P = 0.9
    REPETITION_PENALTY = 1.1

    # Vector store settings
    PERSIST_DIRECTORY = "./chroma_db"
    COLLECTION_NAME = "pdf_collection"

    # Device configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"