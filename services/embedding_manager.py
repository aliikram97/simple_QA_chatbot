
from langchain_huggingface import HuggingFaceEmbeddings
from config import Config

class EmbeddingManager:
    """Handles embedding generation"""

    @staticmethod
    def get_embeddings(model_name: str = Config.EMBEDDING_MODEL):
        """
        Step 5: Generate Embeddings - Initialize embedding model

        Embeddings convert text into numerical vectors that capture semantic meaning.
        Similar texts have similar vector representations.

        Args:
            model_name: Name of the HuggingFace embedding model

        Returns:
            HuggingFaceEmbeddings instance
        """
        print(f"\n Loading embeddings model: {model_name}")
        print(f"   Device: {Config.DEVICE}")

        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': Config.DEVICE},
            encode_kwargs={'normalize_embeddings': True}  # Normalize for cosine similarity
        )

        print(f" Embeddings model loaded successfully!")
        return embeddings