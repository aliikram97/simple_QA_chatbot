from langchain.chains import RetrievalQA
import gradio as gr
from typing import List, Tuple
from pathlib import Path
import time

from services.document_processor import DocumentProcessor
from services.llm_manager import LLMManager
from services.embedding_manager import EmbeddingManager
from services.vectorstore_manager import VectorStoreManager
from services.chain_builder import QAChainBuilder
from config import Config

# Suppress warnings
import warnings

warnings.filterwarnings('ignore')


###############################################################################
# PHASE 5: GRADIO INTERFACE
###############################################################################

class QABotInterface:
    """Manages the Gradio user interface with advanced retrieval options"""

    def __init__(self):
        self.qa_chain = None
        self.vectorstore = None
        self.chat_history = []
        self.text_chunks = None  # Store for retriever comparison
        self.llm = None

    def initialize_system(
            self,
            pdf_files,
            retriever_type: str = "hybrid",
            k: int = 10
    ) -> str:
        """
        Step 12: Connect Backend to Frontend - Process uploaded PDFs

        Args:
            pdf_files: List of uploaded PDF files
            retriever_type: Type of retriever to use
            k: Number of documents to retrieve

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

            self.text_chunks = DocumentProcessor.split_documents(documents)

            # Phase 3: Create vector store
            embeddings = EmbeddingManager.get_embeddings()
            self.vectorstore = VectorStoreManager.create_vector_store(
                self.text_chunks,
                embeddings
            )

            # Create LLM
            self.llm = LLMManager.get_llm()

            # Create advanced retriever based on selection
            retriever = VectorStoreManager.create_retriever(
                vectorstore=self.vectorstore,
                documents=self.text_chunks,
                llm=self.llm,
                k=k,
                retriever_type=retriever_type
            )

            # Phase 4: Create QA chain
            self.qa_chain = QAChainBuilder.create_qa_chain(retriever, self.llm)

            print("\n" + "=" * 80)
            print("SYSTEM READY")
            print("=" * 80 + "\n")

            return f"✅ Successfully processed {len(pdf_files)} PDF(s) with {len(self.text_chunks)} chunks using {retriever_type} retriever (k={k}). Ready for questions!"

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

            print(f"\n🔍 Processing question: {question}")

            # Get answer from QA chain
            result = self.qa_chain({"query": question})
            answer = result['result']
            answer = QAChainBuilder.post_process_answer(question, answer)

            # Format source documents
            sources = self._format_sources(result.get('source_documents', []))

            # Add to chat history
            self.chat_history.append({
                "question": question,
                "answer": answer,
                "timestamp": time.strftime("%H:%M:%S")
            })

            print(f"✓ Answer generated")

            return answer, sources

        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            print(error_msg)
            return error_msg, ""

    def compare_retrievers_ui(self, question: str) -> str:
        """
        Compare different retrievers on the same question

        Args:
            question: Test question

        Returns:
            Formatted comparison results
        """
        try:
            if self.vectorstore is None or self.text_chunks is None:
                return "⚠️  Please upload and process PDFs first."

            if not question.strip():
                return "⚠️  Please enter a question."

            print(f"\n🔬 Comparing retrievers for: {question}")

            results = VectorStoreManager.compare_retrievers(
                vectorstore=self.vectorstore,
                documents=self.text_chunks,
                llm=self.llm,
                query=question,
                k=5
            )

            # Format results for display
            output = f"# 🔬 Retriever Comparison Results\n\n**Query:** {question}\n\n---\n\n"

            for name, result in results.items():
                if "error" in result:
                    output += f"## ❌ {name} Retriever\n**Error:** {result['error']}\n\n"
                else:
                    output += f"## ✅ {name} Retriever\n"
                    output += f"- **Documents Retrieved:** {result['count']}\n"
                    output += f"- **Time:** {result['time']:.3f}s\n"

                    if result['docs']:
                        first_doc = result['docs'][0]
                        preview = first_doc.page_content[:150].replace('\n', ' ')
                        output += f"- **First Result Preview:** {preview}...\n"
                        output += f"- **Source:** {first_doc.metadata.get('source', 'Unknown')}\n"

                    output += "\n"

            return output

        except Exception as e:
            return f"❌ Error during comparison: {str(e)}"

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
        Step 11: Build the UI - Create Gradio interface with retriever options

        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(
                title="PDF QA Bot - Advanced Retrieval",
                theme=gr.themes.Soft(),
                css=".gradio-container {max-width: 1400px; margin: auto;}"
        ) as demo:
            gr.Markdown("""
            # 🤖 PDF Question Answering Bot - Advanced Retrieval Edition
            ### Built with LangChain - Enhanced with Multiple Retrieval Strategies

            Upload PDF documents and ask questions using advanced retrieval techniques. The system supports:
            - **🔍 Multiple Retriever Types**: Simple, MMR, Hybrid, Compressed, Multi-Query, Ultimate
            - **📊 Retriever Comparison**: Test which strategy works best for your queries
            - **⚡ Optimized for Large Knowledge Bases**: Efficient retrieval from extensive document collections
            """)

            with gr.Row():
                # Left column: Upload and configuration
                with gr.Column(scale=1):
                    gr.Markdown("### 📤 Document Upload")
                    pdf_input = gr.File(
                        label="Upload PDF Files",
                        file_count="multiple",
                        file_types=[".pdf"],
                        type="filepath"
                    )

                    gr.Markdown("### ⚙️ Retriever Configuration")

                    retriever_type = gr.Dropdown(
                        choices=[
                            "simple",
                            "mmr",
                            "hybrid",
                            "compressed",
                            "multi_query",
                            "ultimate"
                        ],
                        value="hybrid",
                        label="Retriever Type",
                        info="Choose retrieval strategy (hybrid recommended for large datasets)"
                    )

                    k_value = gr.Slider(
                        minimum=3,
                        maximum=20,
                        value=10,
                        step=1,
                        label="Number of Documents (k)",
                        info="How many documents to retrieve"
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

                    with gr.Accordion("ℹ️ System Info", open=False):
                        gr.Markdown(f"""
                        **Current Configuration:**
                        - **LLM**: {Config.LLM_MODEL_ID.split('/')[-1]}
                        - **Embeddings**: {Config.EMBEDDING_MODEL.split('/')[-1]}
                        - **Chunk Size**: {Config.CHUNK_SIZE}
                        - **Default K**: {Config.RETRIEVAL_K}
                        - **Device**: {Config.DEVICE.upper()}

                        **Retriever Types:**
                        - **Simple**: Basic similarity search
                        - **MMR**: Maximal Marginal Relevance (diversity)
                        - **Hybrid**: Vector + BM25 (recommended) ⭐
                        - **Compressed**: Fetch many, return best
                        - **Multi-Query**: Query variations
                        - **Ultimate**: All techniques combined
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
                            "🎯 Get Answer",
                            variant="primary",
                            size="lg"
                        )
                        clear_btn = gr.Button(
                            "🗑️ Clear",
                            size="lg"
                        )
                        compare_btn = gr.Button(
                            "🔬 Compare Retrievers",
                            variant="secondary",
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

                    # Comparison output (hidden by default)
                    with gr.Accordion("📊 Retriever Comparison Results", open=False):
                        comparison_output = gr.Markdown(
                            value="Run a comparison to see results here..."
                        )

            # Examples section
            with gr.Accordion("💡 Example Questions", open=True):
                gr.Examples(
                    examples=[
                        ["What is the main topic of this document?"],
                        ["Can you summarize the key points?"],
                        ["What are the conclusions mentioned?"],
                        ["What methodology was used?"],
                        ["Are there any recommendations?"],
                    ],
                    inputs=question_input
                )

            # Information boxes
            with gr.Accordion("📖 How to Use", open=False):
                gr.Markdown("""
                ### Step-by-Step Guide:

                1. **Upload PDFs**: Select one or more PDF documents
                2. **Configure Retriever**: Choose a retriever type (start with 'hybrid')
                3. **Adjust k value**: Set how many documents to retrieve (10 is good default)
                4. **Process**: Click "🚀 Process PDFs" to initialize the system
                5. **Ask Questions**: Type your question and click "🎯 Get Answer"
                6. **Compare** (Optional): Click "🔬 Compare Retrievers" to test different strategies

                ### Which Retriever to Use?

                - **Just starting?** Use **Hybrid** (best all-around performance)
                - **Need diversity?** Use **MMR** (reduces redundant results)
                - **Very large docs?** Use **Compressed** (fetch many, return best)
                - **Complex queries?** Use **Ultimate** (best quality, slower)
                - **Simple testing?** Use **Simple** (fastest, basic similarity)

                ### Tips for Best Results:

                - For large knowledge bases (100+ pages), use k=15-20
                - For focused queries, use k=5-10
                - Try the comparison feature to find what works best for your docs
                - Hybrid retriever works well for most use cases
                """)

            # Event handlers
            process_btn.click(
                fn=self.initialize_system,
                inputs=[pdf_input, retriever_type, k_value],
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

            compare_btn.click(
                fn=self.compare_retrievers_ui,
                inputs=[question_input],
                outputs=[comparison_output]
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
        print(f"\n🧪 Test {i}: {query}")
        result = qa_chain({"query": query})
        print(f"Answer: {result['result'][:200]}...")


def run_retriever_benchmarks(vectorstore, text_chunks, llm):
    """
    Benchmark different retrievers

    Args:
        vectorstore: Vector store instance
        text_chunks: Document chunks
        llm: Language model
    """
    print("\n" + "=" * 80)
    print("RETRIEVER BENCHMARKS")
    print("=" * 80)

    test_query = "What is the main topic of this document?"

    # Test different retriever types
    retriever_types = ["simple", "mmr", "hybrid"]

    for rtype in retriever_types:
        print(f"\n📊 Testing {rtype.upper()} retriever...")

        try:
            start = time.time()
            retriever = VectorStoreManager.create_retriever(
                vectorstore=vectorstore,
                documents=text_chunks,
                llm=llm,
                k=10,
                retriever_type=rtype
            )

            docs = retriever.get_relevant_documents(test_query)
            elapsed = time.time() - start

            print(f"✓ Retrieved {len(docs)} documents in {elapsed:.3f}s")
            if docs:
                print(f"  First result preview: {docs[0].page_content[:100]}...")

        except Exception as e:
            print(f"❌ Error: {str(e)}")


###############################################################################
# MAIN EXECUTION
###############################################################################

def main():
    """Main execution function"""

    print("\n" + "=" * 80)
    print("PDF QA BOT - ADVANCED RETRIEVAL EDITION")
    print("=" * 80)
    print("\nEnhanced implementation with multiple retrieval strategies:")
    print("  ✅ Phase 1: Project Setup")
    print("  ✅ Phase 2: Document Processing")
    print("  ✅ Phase 3: Vector Database")
    print("  ✅ Phase 4: Advanced QA Chain with Multiple Retrievers")
    print("  ✅ Phase 5: Enhanced Gradio Interface")
    print("  ✅ Phase 6: Testing & Benchmarking")
    print("=" * 80 + "\n")

    # Create and launch interface
    bot = QABotInterface()
    demo = bot.build_interface()

    print("\n🚀 Launching Gradio interface...")
    print("   Features: Multiple retrievers, comparison tool, optimized for large datasets")
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860
    )


if __name__ == "__main__":
    # Option 1: Launch Gradio Interface (Recommended)
    main()

    # Option 2: Programmatic Usage with Advanced Retrievers (Uncomment to use)
    """
    print("\n" + "="*80)
    print("PROGRAMMATIC USAGE EXAMPLE - ADVANCED RETRIEVERS")
    print("="*80)

    # Step-by-step execution
    pdf_paths = ["path/to/your/document.pdf"]

    # Phase 2: Process documents
    documents = DocumentProcessor.load_multiple_pdfs(pdf_paths)
    text_chunks = DocumentProcessor.split_documents(documents)

    # Phase 3: Create vector store
    embeddings = EmbeddingManager.get_embeddings()
    vectorstore = VectorStoreManager.create_vector_store(text_chunks, embeddings)

    # Create LLM
    llm = LLMManager.get_llm()

    # Example 1: Using Hybrid Retriever (Recommended)
    print("\n" + "="*80)
    print("EXAMPLE 1: Hybrid Retriever")
    print("="*80)
    retriever_hybrid = VectorStoreManager.create_retriever(
        vectorstore=vectorstore,
        documents=text_chunks,
        llm=llm,
        k=10,
        retriever_type="hybrid",
        similarity_weight=0.6,
        bm25_weight=0.4
    )

    # Test retrieval
    VectorStoreManager.test_retrieval(
        retriever_hybrid, 
        "What is this about?",
        show_scores=True
    )

    # Example 2: Using MMR Retriever
    print("\n" + "="*80)
    print("EXAMPLE 2: MMR Retriever")
    print("="*80)
    retriever_mmr = VectorStoreManager.create_retriever(
        vectorstore=vectorstore,
        k=10,
        retriever_type="mmr",
        fetch_k=40,
        lambda_mult=0.5
    )

    # Example 3: Compare All Retrievers
    print("\n" + "="*80)
    print("EXAMPLE 3: Compare Retrievers")
    print("="*80)
    VectorStoreManager.compare_retrievers(
        vectorstore=vectorstore,
        documents=text_chunks,
        llm=llm,
        query="What is the main topic?",
        k=5
    )

    # Example 4: Create QA Chain with chosen retriever
    print("\n" + "="*80)
    print("EXAMPLE 4: QA Chain with Hybrid Retriever")
    print("="*80)
    qa_chain = QAChainBuilder.create_qa_chain(retriever_hybrid, llm)

    # Phase 6: Run tests
    run_tests(qa_chain)

    # Run benchmarks
    run_retriever_benchmarks(vectorstore, text_chunks, llm)

    # Ask questions
    result = qa_chain({"query": "What is the main topic?"})
    print(f"\n✅ Final Answer: {result['result']}")
    """