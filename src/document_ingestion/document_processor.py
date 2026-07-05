"""Document processing module for loading and splitting documents"""

import tempfile
from typing import List
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.schema import Document
from langchain_core.documents import Document

from typing import List, Union
from pathlib import Path
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader,
    PyPDFDirectoryLoader
)
from google.cloud import storage

class DocumentProcessor:
    """Handles document loading and processing"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize document processor
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    def load_from_url(self, url: str) -> List[Document]:
        """Load document(s) from a URL"""
        loader = WebBaseLoader(url)
        return loader.load()

    def load_from_pdf_dir(self, directory: Union[str, Path]) -> List[Document]:
        """Load documents from all PDFs inside a directory"""
        loader = PyPDFDirectoryLoader(str(directory))
        return loader.load()

    def load_from_txt(self, file_path: Union[str, Path]) -> List[Document]:
        """Load document(s) from a TXT file"""
        loader = TextLoader(str(file_path), encoding="utf-8")
        return loader.load()

    def load_from_pdf(self, file_path: Union[str, Path]) -> List[Document]:
        """Load document(s) from a PDF file"""
        loader = PyPDFLoader(str(file_path))
        return loader.load()

    def load_from_gcs(self, gcs_uri: str) -> List[Document]:
        """
        Load a PDF document from a Google Cloud Storage bucket.

        Downloads the blob to a local temp file, then reuses the existing
        PyPDFLoader path. Only single PDF objects are supported (not prefixes
        / directories) since that is all this project needs today.

        Args:
            gcs_uri: URI in the form gs://bucket-name/path/to/file.pdf

        Returns:
            List of loaded documents
        """
        if not gcs_uri.startswith("gs://"):
            raise ValueError(f"Not a GCS URI: {gcs_uri}")

        bucket_name, _, blob_name = gcs_uri[len("gs://"):].partition("/")
        if not blob_name:
            raise ValueError(
                f"GCS URI must point to a specific object, got: {gcs_uri}"
            )
        if not blob_name.lower().endswith(".pdf"):
            raise ValueError(
                f"Only PDF objects are supported from GCS, got: {gcs_uri}"
            )

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_file:
            blob.download_to_filename(tmp_file.name)
            return self.load_from_pdf(tmp_file.name)

    def load_documents(self, sources: List[str]) -> List[Document]:
        """
        Load documents from URLs, GCS PDF objects, PDF directories, or TXT files

        Args:
            sources: List of URLs, gs:// PDF paths, PDF folder paths, or TXT file paths

        Returns:
            List of loaded documents
        """
        docs: List[Document] = []
        for src in sources:
            if src.startswith("http://") or src.startswith("https://"):
                docs.extend(self.load_from_url(src))
                continue

            if src.startswith("gs://"):
                docs.extend(self.load_from_gcs(src))
                continue

            path = Path(src)
            if path.is_dir():  # PDF directory
                docs.extend(self.load_from_pdf_dir(path))
            elif path.suffix.lower() == ".pdf":
                docs.extend(self.load_from_pdf(path))
            elif path.suffix.lower() == ".txt":
                docs.extend(self.load_from_txt(path))
            else:
                raise ValueError(
                    f"Unsupported source type: {src}. "
                    "Use URL, gs:// PDF path, .txt file, or PDF directory."
                )
        return docs
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of split documents
        """
        return self.splitter.split_documents(documents)
    
    def process_urls(self, urls: List[str]) -> List[Document]:
        """
        Complete pipeline to load and split documents
        
        Args:
            urls: List of URLs to process
            
        Returns:
            List of processed document chunks
        """
        docs = self.load_documents(urls)
        return self.split_documents(docs)