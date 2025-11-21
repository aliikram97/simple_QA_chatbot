import time
import os
from typing import List, Optional
from langchain_community.vectorstores import Chroma

from config import Config

class VectorStoreManager:
    """Manages vector database operations"""

    @staticmethod
    def create_vector_store(
            text_chunks: List,
            embeddings,
            persist_directory: str = Config.PERSIST_DIRECTORY
    ):
        """
        Step 6: Create Vector Store - Store embeddings in ChromaDB

        The vector store allows for similarity-based retrieval of relevant documents.

        Args:
            text_chunks: List of document chunks
            embeddings: Embedding model instance
            persist_directory: Directory to persist the vector store

        Returns:
            Chroma vector store instance
        """
        print(f"\n Creating vector store...")
        print(f"   Processing {len(text_chunks)} chunks")
        print(f"   Persist directory: {persist_directory}")

        start_time = time.time()

        # Create vector store with embeddings
        vectorstore = Chroma.from_documents(
            documents=text_chunks,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=Config.COLLECTION_NAME
        )

        elapsed_time = time.time() - start_time
        print(f" Vector store created in {elapsed_time:.2f} seconds")

        return vectorstore

    @staticmethod
    def load_existing_vector_store(
            embeddings,
            persist_directory: str = Config.PERSIST_DIRECTORY
    ) -> Optional:
        """
        Step 6: Create Vector Store - Load existing vector store

        Args:
            embeddings: Embedding model instance
            persist_directory: Directory where vector store is persisted

        Returns:
            Chroma vector store instance or None if not found
        """
        try:
            if os.path.exists(persist_directory):
                print(f"\n Loading existing vector store from {persist_directory}")

                vectorstore = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=embeddings,
                    collection_name=Config.COLLECTION_NAME
                )

                print(f" Vector store loaded successfully")
                return vectorstore
            else:
                print(f"  No existing vector store found at {persist_directory}")
                return None
        except Exception as e:
            print(f" Error loading vector store: {str(e)}")
            return None

    @staticmethod
    def create_retriever(vectorstore, k: int = Config.RETRIEVAL_K):
        """
        Step 7: Build the Retriever - Create retriever from vector store

        The retriever finds the most relevant documents for a given query.

        Args:
            vectorstore: Chroma vector store
            k: Number of documents to retrieve

        Returns:
            Retriever instance
        """
        print(f"\n Creating retriever (k={k})")

        retriever = vectorstore.as_retriever(
            search_type="similarity",  # Options: "similarity", "mmr"
            search_kwargs={
                "k": k,
                "fetch_k": k * 2  # Fetch more candidates for MMR
            }
        )

        print(f" Retriever created successfully")
        return retriever

    @staticmethod
    def test_retrieval(retriever, query: str):
        """
        Step 7: Build the Retriever - Test retrieval functionality

        Args:
            retriever: Retriever instance
            query: Test query string
        """
        print(f"\n Testing retrieval with query: '{query}'")

        docs = retriever.get_relevant_documents(query)

        print(f" Retrieved {len(docs)} documents:")
        for i, doc in enumerate(docs, 1):
            print(f"\n   Document {i}:")
            print(f"   Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"   Page: {doc.metadata.get('page', 'N/A')}")
            print(f"   Content preview: {doc.page_content[:150]}...")