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
                return "  Please upload at least one PDF file."

            print("\n" + "=" * 80)
            print("INITIALIZING QA BOT SYSTEM")
            print("=" * 80)

            # Extract file paths
            pdf_paths = [pdf_file.name for pdf_file in pdf_files]

            # Phase 2: Process documents
            documents = DocumentProcessor.load_multiple_pdfs(pdf_paths)
            if not documents:
                return " Failed to load any documents. Please check your PDF files."

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

            return f" Successfully processed {len(pdf_files)} PDF(s) with {len(text_chunks)} chunks. Ready for questions!"

        except Exception as e:
            error_msg = f" Error during initialization: {str(e)}"
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
                return "  Please upload and process PDFs first.", ""

            if not question.strip():
                return "  Please enter a question.", ""

            print(f"\n Processing question: {question}")

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

            print(f" Answer generated")

            return answer, sources

        except Exception as e:
            error_msg = f" Error: {str(e)}"
            print(error_msg)
            return error_msg, ""

    def _format_sources(self, source_documents: List) -> str:
        """Format source documents for display"""
        if not source_documents:
            return "No sources available"

        sources = " **Sources:**\n\n"
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
            #  PDF Question Answering Bot
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
                    gr.Markdown("###  Document Upload")
                    pdf_input = gr.File(
                        label="Upload PDF Files",
                        file_count="multiple",
                        file_types=[".pdf"],
                        type="filepath"
                    )
                    process_btn = gr.Button(
                        " Process PDFs",
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
                    gr.Markdown("###  Ask Questions")

                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="What would you like to know about the documents?",
                        lines=3
                    )

                    with gr.Row():
                        submit_btn = gr.Button(
                            " Get Answer",
                            variant="primary",
                            size="lg"
                        )
                        clear_btn = gr.Button(
                            " Clear",
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
            gr.Markdown("###  Example Questions")
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
        print(f"\n Test {i}: {query}")
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

    print("\n Launching Gradio interface...")
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