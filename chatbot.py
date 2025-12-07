from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QFileDialog, QComboBox,
    QSlider, QGroupBox, QSplitter, QTabWidget, QScrollArea,
    QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QPalette, QColor
import sys
from typing import List, Tuple
from pathlib import Path
import time

# Import your existing services
from services.document_processor import DocumentProcessor
from services.llm_manager import LLMManager
from services.embedding_manager import EmbeddingManager
from services.vectorstore_manager import VectorStoreManager
from services.chain_builder import QAChainBuilder
from config import Config

import warnings

warnings.filterwarnings('ignore')


class ProcessingThread(QThread):
    """Thread for processing PDFs without blocking UI"""
    finished = pyqtSignal(str)

    def __init__(self, pdf_paths, retriever_type, k_value, parent=None):
        super().__init__(parent)
        self.pdf_paths = pdf_paths
        self.retriever_type = retriever_type
        self.k_value = k_value
        self.parent_interface = parent

    def run(self):
        try:
            # Process documents
            documents = DocumentProcessor.load_multiple_pdfs(self.pdf_paths)
            if not documents:
                self.finished.emit("❌ Failed to load any documents. Please check your PDF files.")
                return

            self.parent_interface.text_chunks = DocumentProcessor.split_documents(documents)

            # Create vector store
            embeddings = EmbeddingManager.get_embeddings()
            self.parent_interface.vectorstore = VectorStoreManager.create_vector_store(
                self.parent_interface.text_chunks,
                embeddings
            )

            # Create LLM
            self.parent_interface.llm = LLMManager.get_llm()

            # Create retriever
            retriever = VectorStoreManager.create_retriever(
                vectorstore=self.parent_interface.vectorstore,
                documents=self.parent_interface.text_chunks,
                llm=self.parent_interface.llm,
                k=self.k_value,
                retriever_type=self.retriever_type
            )

            # Create QA chain
            self.parent_interface.qa_chain = QAChainBuilder.create_qa_chain(
                retriever,
                self.parent_interface.llm
            )

            msg = f"✅ Successfully processed {len(self.pdf_paths)} PDF(s) with {len(self.parent_interface.text_chunks)} chunks using {self.retriever_type} retriever (k={self.k_value}). Ready for questions!"
            self.finished.emit(msg)

        except Exception as e:
            self.finished.emit(f"❌ Error during initialization: {str(e)}")


class QueryThread(QThread):
    """Thread for processing queries without blocking UI"""
    finished = pyqtSignal(str, str)

    def __init__(self, qa_chain, question, parent=None):
        super().__init__(parent)
        self.qa_chain = qa_chain
        self.question = question

    def run(self):
        try:
            print(f"\n🔍 Processing question: {self.question}")

            # Updated for LangChain 0.3.x LCEL pattern
            # Changed from: qa_chain({"query": question})
            # Changed to: qa_chain.invoke({"input": question})
            result = self.qa_chain.invoke({"input": self.question})

            # Updated key access for new LCEL return format
            # Changed from: result['result']
            # Changed to: result['answer']
            answer = result['answer']
            print(f'the raw answer is given as {answer}')
            answer = QAChainBuilder.post_process_answer(self.question, answer)
            print(f'the processed answer is {answer}')

            # Updated for new LCEL return format
            # Changed from: result.get('source_documents', [])
            # Changed to: result.get('context', [])
            sources = self._format_sources(result.get('context', []))

            self.finished.emit(answer, sources)

        except Exception as e:
            self.finished.emit(f"❌ Error: {str(e)}", "")

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


class ComparisonThread(QThread):
    """Thread for comparing retrievers"""
    finished = pyqtSignal(str)

    def __init__(self, vectorstore, text_chunks, llm, question, parent=None):
        super().__init__(parent)
        self.vectorstore = vectorstore
        self.text_chunks = text_chunks
        self.llm = llm
        self.question = question

    def run(self):
        try:
            print(f"\n🔬 Comparing retrievers for: {self.question}")

            results = VectorStoreManager.compare_retrievers(
                vectorstore=self.vectorstore,
                documents=self.text_chunks,
                llm=self.llm,
                query=self.question,
                k=5
            )

            # Format results
            output = f"# 🔬 Retriever Comparison Results\n\n**Query:** {self.question}\n\n---\n\n"

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

            self.finished.emit(output)

        except Exception as e:
            self.finished.emit(f"❌ Error during comparison: {str(e)}")


class QABotWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()

        # Initialize variables
        self.qa_chain = None
        self.vectorstore = None
        self.chat_history = []
        self.text_chunks = None
        self.llm = None
        self.pdf_paths = []
        self.processing_thread = None
        self.query_thread = None
        self.comparison_thread = None

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("🤖 PDF QA Bot - Advanced Retrieval Edition")
        self.setGeometry(100, 100, 1400, 900)

        # Set dark theme styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #000000;
            }
            QWidget {
                background-color: #000000;
                color: #ffffff;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #1e90ff;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #1a1a2e;
                color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #1e90ff;
            }
            QPushButton {
                background-color: #1e90ff;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4169e1;
            }
            QPushButton:pressed {
                background-color: #0066cc;
            }
            QPushButton:disabled {
                background-color: #333333;
                color: #666666;
            }
            QTextEdit {
                border: 2px solid #1e90ff;
                border-radius: 5px;
                padding: 5px;
                background-color: #1a1a2e;
                color: #ffffff;
            }
            QComboBox {
                padding: 5px;
                background-color: #1a1a2e;
                border: 2px solid #1e90ff;
                border-radius: 5px;
                color: #ffffff;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #1e90ff;
                margin-right: 5px;
            }
            QComboBox QAbstractItemView {
                background-color: #1a1a2e;
                color: #ffffff;
                selection-background-color: #1e90ff;
                border: 2px solid #1e90ff;
            }
            QSlider::groove:horizontal {
                background: #1a1a2e;
                height: 8px;
                border-radius: 4px;
                border: 1px solid #1e90ff;
            }
            QSlider::handle:horizontal {
                background: #1e90ff;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #4169e1;
            }
            QLabel {
                color: #ffffff;
                background-color: transparent;
            }
            QTabWidget::pane {
                border: 2px solid #1e90ff;
                background-color: #1a1a2e;
                border-radius: 5px;
            }
            QTabBar::tab {
                background-color: #1a1a2e;
                color: #ffffff;
                border: 2px solid #1e90ff;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            QTabBar::tab:selected {
                background-color: #1e90ff;
                color: #ffffff;
            }
            QTabBar::tab:hover {
                background-color: #4169e1;
            }
            QSplitter::handle {
                background-color: #1e90ff;
                width: 2px;
            }
        """)

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Add header
        header = self.create_header()
        main_layout.addWidget(header)

        # Create splitter for left and right panels
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel (upload and config)
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)

        # Right panel (Q&A interface)
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)

        # Set splitter sizes (1:2 ratio)
        splitter.setSizes([400, 800])

        main_layout.addWidget(splitter)

    def create_header(self):
        """Create the header section"""
        header_widget = QWidget()
        header_layout = QVBoxLayout(header_widget)

        title = QLabel("🤖 PDF Question Answering Bot - Advanced Retrieval Edition")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        subtitle = QLabel("Built with LangChain - Enhanced with Multiple Retrieval Strategies")
        subtitle_font = QFont()
        subtitle_font.setPointSize(12)
        subtitle.setFont(subtitle_font)
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #bbbbbb;")

        description = QLabel(
            "Upload PDF documents and ask questions using advanced retrieval techniques.\n"
            "🔍 Multiple Retriever Types | 📊 Retriever Comparison | ⚡ Optimized for Large Knowledge Bases"
        )
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description.setStyleSheet("color: #999999; margin: 10px;")

        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        header_layout.addWidget(description)

        return header_widget

    def create_left_panel(self):
        """Create the left panel with upload and configuration"""
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # Document Upload Group
        upload_group = QGroupBox("📤 Document Upload")
        upload_layout = QVBoxLayout()

        self.file_list_display = QTextEdit()
        self.file_list_display.setMaximumHeight(100)
        self.file_list_display.setPlaceholderText("No files selected...")
        self.file_list_display.setReadOnly(True)
        upload_layout.addWidget(self.file_list_display)

        upload_btn = QPushButton("📁 Select PDF Files")
        upload_btn.clicked.connect(self.select_files)
        upload_layout.addWidget(upload_btn)

        upload_group.setLayout(upload_layout)
        left_layout.addWidget(upload_group)

        # Retriever Configuration Group
        config_group = QGroupBox("⚙️ Retriever Configuration")
        config_layout = QVBoxLayout()

        # Retriever type dropdown
        retriever_label = QLabel("Retriever Type:")
        config_layout.addWidget(retriever_label)

        self.retriever_combo = QComboBox()
        self.retriever_combo.addItems([
            "simple",
            "mmr",
            "hybrid",
            "compressed",
            "multi_query",
            "ultimate"
        ])
        self.retriever_combo.setCurrentText("hybrid")
        config_layout.addWidget(self.retriever_combo)

        info_label = QLabel("Choose retrieval strategy (hybrid recommended for large datasets)")
        info_label.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        config_layout.addWidget(info_label)

        # K value slider
        k_label = QLabel("Number of Documents (k): 10")
        config_layout.addWidget(k_label)

        self.k_slider = QSlider(Qt.Orientation.Horizontal)
        self.k_slider.setMinimum(3)
        self.k_slider.setMaximum(20)
        self.k_slider.setValue(10)
        self.k_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.k_slider.setTickInterval(1)
        self.k_slider.valueChanged.connect(
            lambda v: k_label.setText(f"Number of Documents (k): {v}")
        )
        config_layout.addWidget(self.k_slider)

        k_info = QLabel("How many documents to retrieve")
        k_info.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        config_layout.addWidget(k_info)

        config_group.setLayout(config_layout)
        left_layout.addWidget(config_group)

        # Process button
        self.process_btn = QPushButton("🚀 Process PDFs")
        self.process_btn.setMinimumHeight(50)
        self.process_btn.setStyleSheet("""
            QPushButton {
                background-color: #1e90ff;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #4169e1;
            }
        """)
        self.process_btn.clicked.connect(self.process_pdfs)
        left_layout.addWidget(self.process_btn)

        # Status output
        status_label = QLabel("System Status:")
        left_layout.addWidget(status_label)

        self.status_output = QTextEdit()
        self.status_output.setMaximumHeight(120)
        self.status_output.setReadOnly(True)
        self.status_output.setPlaceholderText("Status messages will appear here...")
        left_layout.addWidget(self.status_output)

        # System Info Group (collapsible)
        info_group = QGroupBox("ℹ️ System Info")
        info_layout = QVBoxLayout()

        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setMaximumHeight(300)
        info_text.setHtml(f"""
        <b>Current Configuration:</b><br>
        • <b>LLM:</b> {Config.LLM_MODEL_ID.split('/')[-1]}<br>
        • <b>Embeddings:</b> {Config.EMBEDDING_MODEL.split('/')[-1]}<br>
        • <b>Chunk Size:</b> {Config.CHUNK_SIZE}<br>
        • <b>Default K:</b> {Config.RETRIEVAL_K}<br>
        • <b>Device:</b> {Config.DEVICE.upper()}<br><br>

        <b>Retriever Types:</b><br>
        • <b>Simple:</b> Basic similarity search<br>
        • <b>MMR:</b> Maximal Marginal Relevance (diversity)<br>
        • <b>Hybrid:</b> Vector + BM25 (recommended) ⭐<br>
        • <b>Compressed:</b> Fetch many, return best<br>
        • <b>Multi-Query:</b> Query variations<br>
        • <b>Ultimate:</b> All techniques combined
        """)
        info_layout.addWidget(info_text)

        info_group.setLayout(info_layout)
        left_layout.addWidget(info_group)

        left_layout.addStretch()

        return left_widget

    def create_right_panel(self):
        """Create the right panel with Q&A interface"""
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Question input group
        question_group = QGroupBox("💬 Ask Questions")
        question_layout = QVBoxLayout()

        self.question_input = QTextEdit()
        self.question_input.setMaximumHeight(100)
        self.question_input.setPlaceholderText("What would you like to know about the documents?")
        question_layout.addWidget(self.question_input)

        # Action buttons
        button_layout = QHBoxLayout()

        self.submit_btn = QPushButton("🎯 Get Answer")
        self.submit_btn.setMinimumHeight(40)
        self.submit_btn.clicked.connect(self.get_answer)
        button_layout.addWidget(self.submit_btn)

        clear_btn = QPushButton("🗑️ Clear")
        clear_btn.setMinimumHeight(40)
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc143c;
            }
            QPushButton:hover {
                background-color: #ff1744;
            }
        """)
        clear_btn.clicked.connect(self.clear_fields)
        button_layout.addWidget(clear_btn)

        self.compare_btn = QPushButton("🔬 Compare Retrievers")
        self.compare_btn.setMinimumHeight(40)
        self.compare_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff8c00;
            }
            QPushButton:hover {
                background-color: #ffa500;
            }
        """)
        self.compare_btn.clicked.connect(self.compare_retrievers)
        button_layout.addWidget(self.compare_btn)

        question_layout.addLayout(button_layout)
        question_group.setLayout(question_layout)
        right_layout.addWidget(question_group)

        # Create tabs for answer, sources, and comparison
        self.tabs = QTabWidget()

        # Answer tab
        answer_tab = QWidget()
        answer_layout = QVBoxLayout(answer_tab)
        answer_label = QLabel("Answer:")
        answer_layout.addWidget(answer_label)
        self.answer_output = QTextEdit()
        self.answer_output.setReadOnly(True)
        self.answer_output.setPlaceholderText("Your answer will appear here...")
        answer_layout.addWidget(self.answer_output)
        self.tabs.addTab(answer_tab, "📝 Answer")

        # Sources tab
        sources_tab = QWidget()
        sources_layout = QVBoxLayout(sources_tab)
        self.sources_output = QTextEdit()
        self.sources_output.setReadOnly(True)
        self.sources_output.setPlaceholderText("Sources will appear here...")
        sources_layout.addWidget(self.sources_output)
        self.tabs.addTab(sources_tab, "📚 Sources")

        # Comparison tab
        comparison_tab = QWidget()
        comparison_layout = QVBoxLayout(comparison_tab)
        self.comparison_output = QTextEdit()
        self.comparison_output.setReadOnly(True)
        self.comparison_output.setPlaceholderText("Run a comparison to see results here...")
        comparison_layout.addWidget(self.comparison_output)
        self.tabs.addTab(comparison_tab, "📊 Comparison")

        right_layout.addWidget(self.tabs)

        # Example questions group
        examples_group = QGroupBox("💡 Example Questions")
        examples_layout = QVBoxLayout()

        example_questions = [
            "What is the main topic of this document?",
            "Can you summarize the key points?",
            "What are the conclusions mentioned?",
            "What methodology was used?",
            "Are there any recommendations?",
        ]

        for question in example_questions:
            btn = QPushButton(question)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #2a2a3e;
                    color: #ffffff;
                    text-align: left;
                    padding: 8px;
                    border: 1px solid #1e90ff;
                }
                QPushButton:hover {
                    background-color: #1e90ff;
                }
            """)
            btn.clicked.connect(lambda checked, q=question: self.question_input.setText(q))
            examples_layout.addWidget(btn)

        examples_group.setLayout(examples_layout)
        right_layout.addWidget(examples_group)

        return right_widget

    def select_files(self):
        """Open file dialog to select PDF files"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select PDF Files",
            "",
            "PDF Files (*.pdf)"
        )

        if files:
            self.pdf_paths = files
            file_names = "\n".join([Path(f).name for f in files])
            self.file_list_display.setText(f"Selected {len(files)} file(s):\n{file_names}")

    def process_pdfs(self):
        """Process the uploaded PDFs"""
        if not self.pdf_paths:
            self.status_output.setText("⚠️  Please select at least one PDF file.")
            return

        self.process_btn.setEnabled(False)
        self.status_output.setText("⏳ Processing PDFs... Please wait...")

        # Get configuration
        retriever_type = self.retriever_combo.currentText()
        k_value = self.k_slider.value()

        # Start processing in separate thread
        self.processing_thread = ProcessingThread(
            self.pdf_paths,
            retriever_type,
            k_value,
            self
        )
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.start()

    def on_processing_finished(self, message):
        """Handle processing completion"""
        self.status_output.setText(message)
        self.process_btn.setEnabled(True)

        if message.startswith("✅"):
            self.submit_btn.setEnabled(True)
            self.compare_btn.setEnabled(True)

    def get_answer(self):
        """Get answer to user question"""
        if self.qa_chain is None:
            self.answer_output.setText("⚠️  Please upload and process PDFs first.")
            return

        question = self.question_input.toPlainText().strip()
        if not question:
            self.answer_output.setText("⚠️  Please enter a question.")
            return

        self.submit_btn.setEnabled(False)
        self.answer_output.setText("⏳ Searching for answer... Please wait...")
        self.tabs.setCurrentIndex(0)  # Switch to answer tab

        # Start query in separate thread
        self.query_thread = QueryThread(self.qa_chain, question)
        self.query_thread.finished.connect(self.on_query_finished)
        self.query_thread.start()

    def on_query_finished(self, answer, sources):
        """Handle query completion"""
        self.answer_output.setText(answer)
        self.sources_output.setText(sources)
        self.submit_btn.setEnabled(True)

        # Add to chat history
        self.chat_history.append({
            "question": self.question_input.toPlainText(),
            "answer": answer,
            "timestamp": time.strftime("%H:%M:%S")
        })

    def compare_retrievers(self):
        """Compare different retrievers"""
        if self.vectorstore is None or self.text_chunks is None:
            self.comparison_output.setText("⚠️  Please upload and process PDFs first.")
            return

        question = self.question_input.toPlainText().strip()
        if not question:
            self.comparison_output.setText("⚠️  Please enter a question.")
            return

        self.compare_btn.setEnabled(False)
        self.comparison_output.setText("⏳ Comparing retrievers... Please wait...")
        self.tabs.setCurrentIndex(2)  # Switch to comparison tab

        # Start comparison in separate thread
        self.comparison_thread = ComparisonThread(
            self.vectorstore,
            self.text_chunks,
            self.llm,
            question
        )
        self.comparison_thread.finished.connect(self.on_comparison_finished)
        self.comparison_thread.start()

    def on_comparison_finished(self, results):
        """Handle comparison completion"""
        self.comparison_output.setText(results)
        self.compare_btn.setEnabled(True)

    def clear_fields(self):
        """Clear input and output fields"""
        self.question_input.clear()
        self.answer_output.clear()
        self.sources_output.clear()


def main():
    """Main execution function"""
    print("\n" + "=" * 80)
    print("PDF QA BOT - PYQT6 DESKTOP EDITION")
    print("=" * 80)
    print("\nEnhanced desktop application with multiple retrieval strategies")
    print("Updated for LangChain 0.3.x with LCEL pattern")
    print("=" * 80 + "\n")

    app = QApplication(sys.argv)

    # Set application-wide font
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    window = QABotWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()