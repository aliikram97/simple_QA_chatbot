from langchain_community.document_loaders import PyPDFLoader
from typing import List
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from config import Config

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

            print(f" Successfully loaded {len(documents)} pages from {Path(pdf_path).name}")

            # Display metadata from first page
            if documents:
                print(f"   Metadata: {documents[0].metadata}")

            return documents

        except Exception as e:
            print(f" Error loading PDF {pdf_path}: {str(e)}")
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

        print(f"\n Loading {len(pdf_paths)} PDF file(s)...")

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

        print(f" Found {len(pdf_paths)} PDF files in {directory_path}")
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
        print(f"\n  Splitting documents...")
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

        print(f" Created {len(splits)} text chunks")

        # Show sample chunk
        if splits:
            print(f"\n Sample chunk (first 200 chars):")
            print(f"   {splits[0].page_content[:200]}...")

        return splits