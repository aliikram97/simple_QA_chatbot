import torch


class Config:
    """Step 2: Configuration - Centralized configuration management with auto GPU/CPU detection"""

    # Device configuration (detect first)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Model configurations - Auto-select based on device
    if DEVICE == "cuda":
        # GPU Models - Better quality, larger models
        LLM_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
        EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
        print("🚀 GPU detected - Using high-performance models")
    else:
        # CPU Models - Optimized for speed
        LLM_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Already CPU-friendly
        print("💻 CPU mode - Using optimized lightweight models")

    # Text splitting parameters
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100

    # Retrieval parameters
    RETRIEVAL_K = 5  # Number of documents to retrieve

    # LLM parameters - Adjusted for device
    if DEVICE == "cuda":
        MAX_NEW_TOKENS = 512
        TEMPERATURE = 0.0
        TOP_P = 0.9
        REPETITION_PENALTY = 1.1
    else:
        # More conservative for CPU
        MAX_NEW_TOKENS = 256
        TEMPERATURE = 0.1
        TOP_P = 0.9
        REPETITION_PENALTY = 1.2

    # Vector store settings
    PERSIST_DIRECTORY = "./chroma_db"
    COLLECTION_NAME = "pdf_collection"