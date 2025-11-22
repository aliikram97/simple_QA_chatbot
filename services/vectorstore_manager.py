import time
import os
from typing import List, Optional
from langchain_community.vectorstores import Chroma
from langchain.retrievers import (
    EnsembleRetriever,
    ContextualCompressionRetriever,
    MultiQueryRetriever
)
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_community.retrievers import BM25Retriever

from config import Config


class VectorStoreManager:
    """Manages vector database operations with advanced retrieval strategies"""

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
        print(f"\n📦 Creating vector store...")
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
        print(f"✓ Vector store created in {elapsed_time:.2f} seconds")

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
                print(f"\n📂 Loading existing vector store from {persist_directory}")

                vectorstore = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=embeddings,
                    collection_name=Config.COLLECTION_NAME
                )

                print(f"✓ Vector store loaded successfully")
                return vectorstore
            else:
                print(f"⚠️  No existing vector store found at {persist_directory}")
                return None
        except Exception as e:
            print(f"❌ Error loading vector store: {str(e)}")
            return None

    @staticmethod
    def create_retriever(
            vectorstore,
            k: int = Config.RETRIEVAL_K,
            documents: Optional[List] = None,
            llm=None,
            retriever_type: str = "simple",
            **kwargs
    ):
        """
        Step 7: Build the Retriever - Create advanced retriever from vector store

        Supports multiple retrieval strategies optimized for large knowledge bases.

        Args:
            vectorstore: Chroma vector store
            k: Number of documents to retrieve
            documents: Original documents (required for hybrid/BM25)
            llm: Language model (required for multi-query/compression)
            retriever_type: Type of retriever to create
                - "simple": Basic similarity search
                - "mmr": Maximal Marginal Relevance (diversity)
                - "hybrid": Vector + BM25 combination (RECOMMENDED)
                - "compressed": Contextual compression with filtering
                - "multi_query": Multiple query variations
                - "ultimate": All techniques combined (best quality)
            **kwargs: Additional parameters for specific retriever types

        Returns:
            Retriever instance
        """
        print(f"\n🔍 Creating {retriever_type} retriever (k={k})")

        if retriever_type == "simple":
            return VectorStoreManager._create_simple_retriever(vectorstore, k)

        elif retriever_type == "mmr":
            return VectorStoreManager._create_mmr_retriever(vectorstore, k, **kwargs)

        elif retriever_type == "hybrid":
            if documents is None:
                print("⚠️  Warning: documents required for hybrid retriever, falling back to MMR")
                return VectorStoreManager._create_mmr_retriever(vectorstore, k, **kwargs)
            return VectorStoreManager._create_hybrid_retriever(vectorstore, documents, k, **kwargs)

        elif retriever_type == "compressed":
            if llm is None:
                print("⚠️  Warning: llm required for compression, falling back to MMR")
                return VectorStoreManager._create_mmr_retriever(vectorstore, k, **kwargs)
            return VectorStoreManager._create_compressed_retriever(vectorstore, llm, k, **kwargs)

        elif retriever_type == "multi_query":
            if llm is None:
                print("⚠️  Warning: llm required for multi-query, falling back to MMR")
                return VectorStoreManager._create_mmr_retriever(vectorstore, k, **kwargs)
            return VectorStoreManager._create_multi_query_retriever(vectorstore, llm, k, **kwargs)

        elif retriever_type == "ultimate":
            if documents is None or llm is None:
                print("⚠️  Warning: documents and llm required for ultimate retriever")
                return VectorStoreManager._create_mmr_retriever(vectorstore, k, **kwargs)
            return VectorStoreManager._create_ultimate_retriever(vectorstore, documents, llm, k, **kwargs)

        else:
            print(f"⚠️  Unknown retriever type: {retriever_type}, using simple")
            return VectorStoreManager._create_simple_retriever(vectorstore, k)

    @staticmethod
    def _create_simple_retriever(vectorstore, k: int):
        """Basic similarity search retriever"""
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        print(f"✓ Simple retriever created")
        return retriever

    @staticmethod
    def _create_mmr_retriever(
            vectorstore,
            k: int,
            fetch_k: int = None,
            lambda_mult: float = 0.5
    ):
        """
        MMR retriever for diverse results

        Args:
            fetch_k: Number of docs to fetch before MMR (default: k*4)
            lambda_mult: 0=max diversity, 1=max relevance (default: 0.5)
        """
        if fetch_k is None:
            fetch_k = min(k * 4, 100)  # Cap at 100 for performance

        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": fetch_k,
                "lambda_mult": lambda_mult
            }
        )
        print(f"✓ MMR retriever created (fetch_k={fetch_k}, λ={lambda_mult})")
        return retriever

    @staticmethod
    def _create_hybrid_retriever(
            vectorstore,
            documents: List,
            k: int,
            similarity_weight: float = 0.5,
            bm25_weight: float = 0.5
    ):
        """
        Hybrid retriever combining vector similarity and BM25 keyword search
        RECOMMENDED for large, diverse knowledge bases

        Args:
            similarity_weight: Weight for semantic search (0-1)
            bm25_weight: Weight for keyword search (0-1)
        """
        # Vector retriever
        vector_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )

        # BM25 retriever
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = k

        # Ensemble
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[similarity_weight, bm25_weight]
        )

        print(f"✓ Hybrid retriever created (weights: {similarity_weight:.1f}/{bm25_weight:.1f})")
        return ensemble_retriever

    @staticmethod
    def _create_compressed_retriever(
            vectorstore,
            llm,
            k: int,
            initial_k: int = None,
            similarity_threshold: float = 0.7
    ):
        """
        Compressed retriever with embeddings filtering
        Retrieves more docs, returns only the most relevant

        Args:
            initial_k: Initial docs to retrieve (default: k*3)
            similarity_threshold: Minimum similarity to keep (0-1)
        """
        if initial_k is None:
            initial_k = k * 3

        base_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": initial_k}
        )

        embeddings_filter = EmbeddingsFilter(
            embeddings=vectorstore._embedding_function,
            similarity_threshold=similarity_threshold,
            k=k
        )

        compression_retriever = ContextualCompressionRetriever(
            base_compressor=embeddings_filter,
            base_retriever=base_retriever
        )

        print(f"✓ Compressed retriever created ({initial_k}→{k}, threshold={similarity_threshold})")
        return compression_retriever

    @staticmethod
    def _create_multi_query_retriever(
            vectorstore,
            llm,
            k: int
    ):
        """
        Multi-query retriever generates query variations for better recall
        """
        base_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )

        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm
        )

        print(f"✓ Multi-query retriever created")
        return multi_query_retriever

    @staticmethod
    def _create_ultimate_retriever(
            vectorstore,
            documents: List,
            llm,
            k: int,
            initial_k: int = None
    ):
        """
        Ultimate retriever combining multiple techniques
        Best quality but slower - use for complex queries

        Pipeline: Hybrid → Compression → Multi-Query
        """
        if initial_k is None:
            initial_k = k * 4

        print(f"   Building pipeline: Hybrid → Compression → Multi-Query")

        # Stage 1: Hybrid retrieval
        vector_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": initial_k}
        )

        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = initial_k

        hybrid = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.5, 0.5]
        )

        # Stage 2: Compression
        embeddings_filter = EmbeddingsFilter(
            embeddings=vectorstore._embedding_function,
            similarity_threshold=0.6,
            k=k
        )

        compressed = ContextualCompressionRetriever(
            base_compressor=embeddings_filter,
            base_retriever=hybrid
        )

        # Stage 3: Multi-query
        final_retriever = MultiQueryRetriever.from_llm(
            retriever=compressed,
            llm=llm
        )

        print(f"✓ Ultimate retriever created ({initial_k}→{k} pipeline)")
        return final_retriever

    @staticmethod
    def test_retrieval(retriever, query: str, show_scores: bool = False):
        """
        Step 7: Build the Retriever - Test retrieval functionality

        Args:
            retriever: Retriever instance
            query: Test query string
            show_scores: Whether to show relevance scores (if available)
        """
        print(f"\n🔍 Testing retrieval with query: '{query}'")

        try:
            # Try to get documents with scores
            if show_scores and hasattr(retriever, 'vectorstore'):
                docs_and_scores = retriever.vectorstore.similarity_search_with_score(query)
                docs = [doc for doc, _ in docs_and_scores]
                scores = [score for _, score in docs_and_scores]
            else:
                docs = retriever.get_relevant_documents(query)
                scores = None
        except Exception as e:
            docs = retriever.get_relevant_documents(query)
            scores = None

        print(f"✓ Retrieved {len(docs)} documents:")

        for i, doc in enumerate(docs, 1):
            print(f"\n   📄 Document {i}:")
            if scores:
                print(f"   Score: {scores[i - 1]:.4f}")
            print(f"   Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"   Page: {doc.metadata.get('page', 'N/A')}")
            content_preview = doc.page_content[:200].replace('\n', ' ')
            print(f"   Content: {content_preview}...")

    @staticmethod
    def compare_retrievers(
            vectorstore,
            documents: List,
            llm,
            query: str,
            k: int = 5
    ):
        """
        Compare different retriever types on the same query
        Useful for evaluating which retriever works best for your use case

        Args:
            vectorstore: Chroma vector store
            documents: Original documents
            llm: Language model
            query: Test query
            k: Number of documents to retrieve
        """
        print(f"\n🔬 Comparing Retrievers")
        print(f"Query: '{query}'")
        print(f"=" * 80)

        retriever_configs = [
            ("Simple", "simple", {}),
            ("MMR", "mmr", {"lambda_mult": 0.5}),
            ("Hybrid", "hybrid", {"similarity_weight": 0.6, "bm25_weight": 0.4}),
        ]

        results = {}

        for name, rtype, kwargs in retriever_configs:
            print(f"\n--- {name} Retriever ---")
            try:
                retriever = VectorStoreManager.create_retriever(
                    vectorstore=vectorstore,
                    k=k,
                    documents=documents,
                    llm=llm,
                    retriever_type=rtype,
                    **kwargs
                )

                start = time.time()
                docs = retriever.get_relevant_documents(query)
                elapsed = time.time() - start

                results[name] = {
                    "docs": docs,
                    "time": elapsed,
                    "count": len(docs)
                }

                print(f"Retrieved: {len(docs)} docs in {elapsed:.3f}s")
                if docs:
                    print(f"First result: {docs[0].page_content[:100]}...")

            except Exception as e:
                print(f"❌ Error: {str(e)}")
                results[name] = {"error": str(e)}

        print(f"\n{'=' * 80}")
        print("Summary:")
        for name, result in results.items():
            if "error" not in result:
                print(f"  {name:15} | {result['count']} docs | {result['time']:.3f}s")

        return results