"""
Complete QA Bot with LangChain - All Phases Implementation
Study Guide: Each phase is clearly separated with detailed comments
"""

from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import gradio as gr
import os
from typing import List, Optional, Dict, Tuple
from pathlib import Path
import time

# Suppress warnings
import warnings

warnings.filterwarnings('ignore')


###############################################################################
# PHASE 1: PROJECT SETUP
###############################################################################

class Config:
    """Step 2: Configuration - Centralized configuration management"""

    # Model configurations
    LLM_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # Text splitting parameters
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    # Retrieval parameters
    RETRIEVAL_K = 3  # Number of documents to retrieve

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


###############################################################################
# PHASE 2: DOCUMENT PROCESSING PIPELINE
###############################################################################

class DocumentProcessor:
    """Handles all document processing operations"""

    @staticmethod
    def load_single_pdf(pdf_path: str) -> List:
        """
        Step 3: PDF Loading - Load a single PDF file

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of Document objects containing text and metadata
        """
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")

            print(f"📄 Loading PDF: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

            print(f"✅ Successfully loaded {len(documents)} pages from {Path(pdf_path).name}")

            # Display metadata from first page
            if documents:
                print(f"   Metadata: {documents[0].metadata}")

            return documents

        except Exception as e:
            print(f"❌ Error loading PDF {pdf_path}: {str(e)}")
            return []

    @staticmethod
    def load_multiple_pdfs(pdf_paths: List[str]) -> List:
        """
        Step 3: PDF Loading - Load multiple PDF files

        Args:
            pdf_paths: List of paths to PDF files

        Returns:
            Combined list of Document objects from all PDFs
        """
        all_documents = []
        successful_loads = 0

        print(f"\n📚 Loading {len(pdf_paths)} PDF file(s)...")

        for pdf_path in pdf_paths:
            documents = DocumentProcessor.load_single_pdf(pdf_path)
            if documents:
                all_documents.extend(documents)
                successful_loads += 1

        print(f"\n✅ Total: {len(all_documents)} pages from {successful_loads}/{len(pdf_paths)} PDF(s)")
        return all_documents

    @staticmethod
    def load_pdfs_from_directory(directory_path: str) -> List:
        """
        Step 3: PDF Loading - Load all PDFs from a directory

        Args:
            directory_path: Path to directory containing PDFs

        Returns:
            Combined list of Document objects from all PDFs
        """
        pdf_files = list(Path(directory_path).glob("*.pdf"))
        pdf_paths = [str(pdf) for pdf in pdf_files]

        print(f"🔍 Found {len(pdf_paths)} PDF files in {directory_path}")
        return DocumentProcessor.load_multiple_pdfs(pdf_paths)

    @staticmethod
    def split_documents(
            documents: List,
            chunk_size: int = Config.CHUNK_SIZE,
            chunk_overlap: int = Config.CHUNK_OVERLAP
    ) -> List:
        """
        Step 4: Text Splitting - Split documents into manageable chunks

        The overlap helps maintain context between chunks for better retrieval.

        Args:
            documents: List of Document objects
            chunk_size: Size of each text chunk in characters
            chunk_overlap: Number of overlapping characters between chunks

        Returns:
            List of split document chunks
        """
        print(f"\n✂️  Splitting documents...")
        print(f"   Chunk size: {chunk_size} characters")
        print(f"   Overlap: {chunk_overlap} characters")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],  # Hierarchical splitting
            is_separator_regex=False
        )

        splits = text_splitter.split_documents(documents)

        print(f"✅ Created {len(splits)} text chunks")

        # Show sample chunk
        if splits:
            print(f"\n📝 Sample chunk (first 200 chars):")
            print(f"   {splits[0].page_content[:200]}...")

        return splits


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
        print(f"\n🧮 Loading embeddings model: {model_name}")
        print(f"   Device: {Config.DEVICE}")

        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': Config.DEVICE},
            encode_kwargs={'normalize_embeddings': True}  # Normalize for cosine similarity
        )

        print(f"✅ Embeddings model loaded successfully!")
        return embeddings


###############################################################################
# PHASE 3: VECTOR DATABASE SETUP
###############################################################################

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
        print(f"\n💾 Creating vector store...")
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
        print(f"✅ Vector store created in {elapsed_time:.2f} seconds")

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

                print(f"✅ Vector store loaded successfully")
                return vectorstore
            else:
                print(f"⚠️  No existing vector store found at {persist_directory}")
                return None
        except Exception as e:
            print(f"❌ Error loading vector store: {str(e)}")
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
        print(f"\n🔍 Creating retriever (k={k})")

        retriever = vectorstore.as_retriever(
            search_type="similarity",  # Options: "similarity", "mmr"
            search_kwargs={
                "k": k,
                "fetch_k": k * 2  # Fetch more candidates for MMR
            }
        )

        print(f"✅ Retriever created successfully")
        return retriever

    @staticmethod
    def test_retrieval(retriever, query: str):
        """
        Step 7: Build the Retriever - Test retrieval functionality

        Args:
            retriever: Retriever instance
            query: Test query string
        """
        print(f"\n🧪 Testing retrieval with query: '{query}'")

        docs = retriever.get_relevant_documents(query)

        print(f"✅ Retrieved {len(docs)} documents:")
        for i, doc in enumerate(docs, 1):
            print(f"\n   Document {i}:")
            print(f"   Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"   Page: {doc.metadata.get('page', 'N/A')}")
            print(f"   Content preview: {doc.page_content[:150]}...")


###############################################################################
# PHASE 4: QA CHAIN CONSTRUCTION
###############################################################################

class LLMManager:
    """Manages Language Model operations"""

    @staticmethod
    def get_llm(
            model_id: str = Config.LLM_MODEL_ID,
            max_new_tokens: int = Config.MAX_NEW_TOKENS,
            temperature: float = Config.TEMPERATURE
    ):
        """
        Step 8: Initialize the LLM - Set up the language model

        Args:
            model_id: HuggingFace model identifier
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)

        Returns:
            HuggingFacePipeline instance
        """
        print(f"\n🤖 Initializing LLM: {model_id}")
        print(f"   Device: {Config.DEVICE}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if Config.DEVICE == "cuda" else torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True
        )

        print(f"✅ Model loaded on {Config.DEVICE}")

        # Create pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=Config.TOP_P,
            do_sample=True,
            repetition_penalty=Config.REPETITION_PENALTY
        )

        hf_llm = HuggingFacePipeline(pipeline=pipe)

        print(f"✅ LLM pipeline ready!")
        return hf_llm


class QAChainBuilder:
    """Builds and manages QA chains"""

    @staticmethod
    def create_prompt_template() -> PromptTemplate:
        """
        Step 10: Prompt Engineering - Design system prompt

        Returns:
            PromptTemplate instance
        """
        template = """You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.

IMPORTANT INSTRUCTIONS:
- Only use information from the provided context
- If the answer is not in the context, say "I don't have enough information to answer this question"
- Be concise but thorough
- Cite specific details from the context when possible
- Do not make up information

Context:
{context}

Question: {question}

Helpful Answer:"""

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    @staticmethod
    def create_qa_chain(retriever, llm, return_source_documents: bool = True):
        """
        Step 9: Create the QA Chain - Connect retriever and LLM

        Chain types:
        - stuff: Put all docs into context (simple, works for small docs)
        - map_reduce: Summarize each doc, then combine
        - refine: Iteratively refine answer with each doc
        - map_rerank: Score each doc's answer, return best

        Args:
            retriever: Document retriever
            llm: Language model
            return_source_documents: Whether to return source documents

        Returns:
            RetrievalQA chain
        """
        print(f"\n⛓️  Creating QA chain...")
        print(f"   Chain type: stuff")
        print(f"   Return sources: {return_source_documents}")

        prompt = QAChainBuilder.create_prompt_template()

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # Simplest approach
            retriever=retriever,
            return_source_documents=return_source_documents,
            chain_type_kwargs={"prompt": prompt},
            verbose=False
        )

        print(f"✅ QA chain created successfully")
        return qa_chain


###############################################################################
# PHASE 5: GRADIO INTERFACE
###############################################################################

class QABotInterface:
    """Manages the Gradio user interface"""

    def __init__(self):
        self.qa_chain = None
        self.vectorstore = None
        self.chat_history = []

    def initialize_system(self, pdf_files) -> str:
        """
        Step 12: Connect Backend to Frontend - Process uploaded PDFs

        Args:
            pdf_files: List of uploaded PDF files

        Returns:
            Status message
        """
        try:
            if not pdf_files:
                return "⚠️  Please upload at least one PDF file."

            print("\n" + "=" * 80)
            print("INITIALIZING QA BOT SYSTEM")
            print("=" * 80)

            # Extract file paths
            pdf_paths = [pdf_file.name for pdf_file in pdf_files]

            # Phase 2: Process documents
            documents = DocumentProcessor.load_multiple_pdfs(pdf_paths)
            if not documents:
                return "❌ Failed to load any documents. Please check your PDF files."

            text_chunks = DocumentProcessor.split_documents(documents)

            # Phase 3: Create vector store
            embeddings = EmbeddingManager.get_embeddings()
            self.vectorstore = VectorStoreManager.create_vector_store(
                text_chunks,
                embeddings
            )
            retriever = VectorStoreManager.create_retriever(self.vectorstore)

            # Phase 4: Create QA chain
            llm = LLMManager.get_llm()
            self.qa_chain = QAChainBuilder.create_qa_chain(retriever, llm)

            print("\n" + "=" * 80)
            print("SYSTEM READY")
            print("=" * 80 + "\n")

            return f"✅ Successfully processed {len(pdf_files)} PDF(s) with {len(text_chunks)} chunks. Ready for questions!"

        except Exception as e:
            error_msg = f"❌ Error during initialization: {str(e)}"
            print(error_msg)
            return error_msg

    def answer_question(self, question: str) -> Tuple[str, str]:
        """
        Step 12: Connect Backend to Frontend - Handle user queries

        Args:
            question: User's question

        Returns:
            Tuple of (answer, sources)
        """
        try:
            if self.qa_chain is None:
                return "⚠️  Please upload and process PDFs first.", ""

            if not question.strip():
                return "⚠️  Please enter a question.", ""

            print(f"\n🤔 Processing question: {question}")

            # Get answer from QA chain
            result = self.qa_chain({"query": question})
            answer = result['result']

            # Format source documents
            sources = self._format_sources(result.get('source_documents', []))

            # Add to chat history
            self.chat_history.append({
                "question": question,
                "answer": answer,
                "timestamp": time.strftime("%H:%M:%S")
            })

            print(f"✅ Answer generated")

            return answer, sources

        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            print(error_msg)
            return error_msg, ""

    def _format_sources(self, source_documents: List) -> str:
        """Format source documents for display"""
        if not source_documents:
            return "No sources available"

        sources = "📚 **Sources:**\n\n"
        for i, doc in enumerate(source_documents, 1):
            source_file = Path(doc.metadata.get('source', 'Unknown')).name
            page_num = doc.metadata.get('page', 'N/A')
            content_preview = doc.page_content[:200].replace('\n', ' ')

            sources += f"**{i}. {source_file} (Page {page_num})**\n"
            sources += f"   {content_preview}...\n\n"

        return sources

    def clear_history(self):
        """Clear chat history"""
        self.chat_history = []
        return "Chat history cleared"

    def build_interface(self):
        """
        Step 11: Build the UI - Create Gradio interface

        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(
                title="PDF QA Bot",
                theme=gr.themes.Soft(),
                css=".gradio-container {max-width: 1200px; margin: auto;}"
        ) as demo:
            gr.Markdown("""
            # 📚 PDF Question Answering Bot
            ### Built with LangChain - Complete Implementation Study Guide

            Upload PDF documents and ask questions about their content. The system uses:
            - **Document Processing**: PDF loading and text splitting
            - **Embeddings**: Semantic vector representations
            - **Vector Database**: ChromaDB for similarity search
            - **LLM**: TinyLlama for answer generation
            """)

            with gr.Row():
                # Left column: Upload and system info
                with gr.Column(scale=1):
                    gr.Markdown("### 📁 Document Upload")
                    pdf_input = gr.File(
                        label="Upload PDF Files",
                        file_count="multiple",
                        file_types=[".pdf"],
                        type="filepath"
                    )
                    process_btn = gr.Button(
                        "🚀 Process PDFs",
                        variant="primary",
                        size="lg"
                    )
                    status_output = gr.Textbox(
                        label="System Status",
                        lines=4,
                        interactive=False
                    )

                    gr.Markdown("### ℹ️ System Info")
                    gr.Markdown(f"""
                    - **LLM**: {Config.LLM_MODEL_ID.split('/')[-1]}
                    - **Embeddings**: {Config.EMBEDDING_MODEL.split('/')[-1]}
                    - **Chunk Size**: {Config.CHUNK_SIZE}
                    - **Retrieval K**: {Config.RETRIEVAL_K}
                    - **Device**: {Config.DEVICE.upper()}
                    """)

                # Right column: Q&A interface
                with gr.Column(scale=2):
                    gr.Markdown("### 💬 Ask Questions")

                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="What would you like to know about the documents?",
                        lines=3
                    )

                    with gr.Row():
                        submit_btn = gr.Button(
                            "🔍 Get Answer",
                            variant="primary",
                            size="lg"
                        )
                        clear_btn = gr.Button(
                            "🗑️ Clear",
                            size="lg"
                        )

                    answer_output = gr.Textbox(
                        label="Answer",
                        lines=8,
                        interactive=False
                    )

                    sources_output = gr.Markdown(
                        label="Sources"
                    )

            # Examples
            gr.Markdown("### 📝 Example Questions")
            gr.Examples(
                examples=[
                    ["What is the main topic of this document?"],
                    ["Can you summarize the key points?"],
                    ["What are the conclusions mentioned?"],
                ],
                inputs=question_input
            )

            # Event handlers
            process_btn.click(
                fn=self.initialize_system,
                inputs=[pdf_input],
                outputs=[status_output]
            )

            submit_btn.click(
                fn=self.answer_question,
                inputs=[question_input],
                outputs=[answer_output, sources_output]
            )

            clear_btn.click(
                fn=lambda: ("", "", ""),
                outputs=[question_input, answer_output, sources_output]
            )

        return demo


###############################################################################
# PHASE 6: TESTING AND OPTIMIZATION
###############################################################################

def run_tests(qa_chain):
    """
    Step 13: Testing - Run test queries

    Args:
        qa_chain: QA chain instance
    """
    print("\n" + "=" * 80)
    print("RUNNING TESTS")
    print("=" * 80)

    test_queries = [
        "What is the main topic?",
        "Can you provide a summary?",
        "What information is not in the documents?",  # Test "I don't know" response
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Test {i}: {query}")
        result = qa_chain({"query": query})
        print(f"Answer: {result['result'][:200]}...")


###############################################################################
# MAIN EXECUTION
###############################################################################

def main():
    """Main execution function"""

    print("\n" + "=" * 80)
    print("PDF QA BOT - COMPLETE IMPLEMENTATION")
    print("=" * 80)
    print("\nThis is a complete study guide implementation covering all phases:")
    print("  Phase 1: Project Setup")
    print("  Phase 2: Document Processing")
    print("  Phase 3: Vector Database")
    print("  Phase 4: QA Chain Construction")
    print("  Phase 5: Gradio Interface")
    print("  Phase 6: Testing")
    print("=" * 80 + "\n")

    # Create and launch interface
    bot = QABotInterface()
    demo = bot.build_interface()

    print("\n🚀 Launching Gradio interface...")
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860
    )


if __name__ == "__main__":
    # Option 1: Launch Gradio Interface (Recommended)
    main()

    # Option 2: Programmatic Usage Example (Uncomment to use)
    """
    print("\n" + "="*80)
    print("PROGRAMMATIC USAGE EXAMPLE")
    print("="*80)

    # Step-by-step execution
    pdf_paths = ["path/to/your/document.pdf"]

    # Phase 2: Process documents
    documents = DocumentProcessor.load_multiple_pdfs(pdf_paths)
    text_chunks = DocumentProcessor.split_documents(documents)

    # Phase 3: Create vector store
    embeddings = EmbeddingManager.get_embeddings()
    vectorstore = VectorStoreManager.create_vector_store(text_chunks, embeddings)
    retriever = VectorStoreManager.create_retriever(vectorstore)

    # Test retrieval
    VectorStoreManager.test_retrieval(retriever, "What is this about?")

    # Phase 4: Create QA chain
    llm = LLMManager.get_llm()
    qa_chain = QAChainBuilder.create_qa_chain(retriever, llm)

    # Phase 6: Run tests
    run_tests(qa_chain)

    # Ask questions
    result = qa_chain({"query": "What is the main topic?"})
    print(f"\nAnswer: {result['result']}")
    """