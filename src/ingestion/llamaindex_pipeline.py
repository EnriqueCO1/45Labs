"""
Ingestion Pipeline for 45Labs
Uses LlamaIndex for chunking, embedding, and storage in Supabase.
Run this offline/rarely to process PDFs.
"""

import os
from pathlib import Path
from typing import List, Dict, Any
import logging
from datetime import datetime

from llama_index.core import (
    SimpleDirectoryReader,
    Document,
    StorageContext,
    VectorStoreIndex,
    Settings
)
from llama_index.vector_stores.supabase import SupabaseVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LlamaIndexIngestion:
    """
    Ingestion pipeline using LlamaIndex.
    Handles: PDF → Chunks → Embeddings → Supabase
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Initialize the ingestion pipeline.
        
        Args:
            chunk_size: Size of each chunk in tokens
            chunk_overlap: Overlap between chunks in tokens
            embedding_model: OpenAI embedding model
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Configure LlamaIndex settings globally
        Settings.embed_model = OpenAIEmbedding(
            model=embedding_model,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        Settings.chunk_size = chunk_size
        Settings.chunk_overlap = chunk_overlap
        
        # Initialize Supabase vector store
        self.vector_store = SupabaseVectorStore(
            postgres_connection_string=self._get_connection_string(),
            collection_name="documents",
            dimension=1536  # text-embedding-3-small dimension
        )
        
        logger.info(f"Initialized ingestion pipeline with chunk_size={chunk_size}")
    
    def _get_connection_string(self) -> str:
        """
        Build PostgreSQL connection string using Supabase connection pooler.
        The pooler works better with psycopg2 on macOS than direct connection.
        """
        from urllib.parse import quote_plus
        
        supabase_url = os.getenv("SUPABASE_URL")
        
        if not supabase_url:
            raise ValueError("SUPABASE_URL not set")
        
        # Extract project ref from URL
        project_ref = supabase_url.replace("https://", "").replace("http://", "").split(".")[0]
        
        # Get database password
        db_password = os.getenv("SUPABASE_DB_PASSWORD")
        if not db_password:
            raise ValueError(
                "SUPABASE_DB_PASSWORD not set. "
                "Get it from: Supabase Dashboard > Project Settings > Database > Connection string"
            )
        
        # URL-encode password
        encoded_password = quote_plus(db_password)
        
        # Use connection pooler (port 6543) instead of direct connection (port 5432)
        # This avoids psycopg2 socket issues on macOS
        host = f"aws-0-us-west-1.pooler.supabase.com"
        connection_string = (
            f"postgresql://postgres.{project_ref}:{encoded_password}@{host}:6543/postgres"
        )
        
        logger.info(f"Using Supabase connection pooler: {host}:6543")
        
        return connection_string
    
    def load_pdf(self, pdf_path: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """
        Load a single PDF file.
        
        Args:
            pdf_path: Path to PDF
            metadata: Additional metadata to attach
            
        Returns:
            List of Document objects
        """
        logger.info(f"Loading PDF: {pdf_path}")
        
        # Use LlamaIndex's SimpleDirectoryReader
        reader = SimpleDirectoryReader(
            input_files=[pdf_path],
            filename_as_id=True
        )
        
        documents = reader.load_data()
        
        # Add custom metadata
        if metadata:
            for doc in documents:
                doc.metadata.update(metadata)
                doc.metadata["file_name"] = Path(pdf_path).name
                doc.metadata["ingested_at"] = datetime.now().isoformat()
        
        logger.info(f"Loaded {len(documents)} document(s) from PDF")
        return documents
    
    def load_directory(
        self,
        directory: str,
        metadata: Dict[str, Any] = None,
        recursive: bool = True
    ) -> List[Document]:
        """
        Load all PDFs from a directory.
        
        Args:
            directory: Directory path
            metadata: Common metadata for all files
            recursive: Search subdirectories
            
        Returns:
            List of Document objects
        """
        logger.info(f"Loading PDFs from: {directory}")
        
        reader = SimpleDirectoryReader(
            input_dir=directory,
            required_exts=[".pdf"],
            recursive=recursive,
            filename_as_id=True
        )
        
        documents = reader.load_data()
        
        # Add metadata
        if metadata:
            for doc in documents:
                doc.metadata.update(metadata)
                doc.metadata["ingested_at"] = datetime.now().isoformat()
        
        logger.info(f"Loaded {len(documents)} documents from directory")
        return documents
    
    def ingest_documents(
        self,
        documents: List[Document],
        show_progress: bool = True
    ) -> VectorStoreIndex:
        """
        Process documents and store in Supabase.
        
        LlamaIndex handles:
        - Chunking (using SentenceSplitter)
        - Embedding (using OpenAI)
        - Storage (in Supabase pgvector)
        
        Args:
            documents: List of Document objects
            show_progress: Show progress bar
            
        Returns:
            VectorStoreIndex
        """
        logger.info(f"Ingesting {len(documents)} documents...")
        
        # Create storage context with Supabase
        storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        
        # Create index - this does chunking, embedding, and storage
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=show_progress
        )
        
        logger.info("✓ Documents ingested successfully")
        return index
    
    def ingest_pdf(
        self,
        pdf_path: str,
        metadata: Dict[str, Any] = None
    ) -> VectorStoreIndex:
        """
        Complete pipeline: Load PDF → Chunk → Embed → Store.
        
        Args:
            pdf_path: Path to PDF
            metadata: Metadata to attach
            
        Returns:
            VectorStoreIndex
        """
        documents = self.load_pdf(pdf_path, metadata)
        return self.ingest_documents(documents)
    
    def ingest_directory_pipeline(
        self,
        directory: str,
        category: str,
        metadata: Dict[str, Any] = None
    ) -> VectorStoreIndex:
        """
        Complete pipeline for a directory.
        
        Args:
            directory: Directory path
            category: Document category (rubrics, guides, etc.)
            metadata: Additional metadata
            
        Returns:
            VectorStoreIndex
        """
        # Prepare metadata
        full_metadata = {"category": category}
        if metadata:
            full_metadata.update(metadata)
        
        # Load and ingest
        documents = self.load_directory(directory, full_metadata)
        return self.ingest_documents(documents)
    
    def ingest_organized_library(self, base_path: str = "data/documents"):
        """
        Ingest entire organized library.
        
        Expected structure:
        data/documents/
        ├── rubrics/
        ├── guides/
        ├── exemplars/
        └── resources/
        """
        base = Path(base_path)
        categories = ["rubrics", "guides", "exemplars", "resources", "past_papers"]
        
        total_docs = 0
        
        for category in categories:
            category_path = base / category
            
            if not category_path.exists():
                logger.warning(f"Category not found: {category_path}")
                continue
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing category: {category}")
            logger.info(f"{'='*60}")
            
            try:
                documents = self.load_directory(
                    str(category_path),
                    metadata={"category": category}
                )
                
                if documents:
                    self.ingest_documents(documents)
                    total_docs += len(documents)
                    logger.info(f"✓ {category}: {len(documents)} documents")
                else:
                    logger.info(f"No PDFs found in {category}")
                    
            except Exception as e:
                logger.error(f"✗ Error processing {category}: {e}")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"TOTAL: {total_docs} documents ingested")
        logger.info(f"{'='*60}")
        
        return total_docs


class IncrementalIngestion:
    """
    Track which files have been processed to avoid re-processing.
    """
    
    def __init__(self, registry_path: str = "data/ingestion_registry.json"):
        """Initialize registry tracker."""
        import json
        
        self.registry_path = Path(registry_path)
        self.registry = self._load_registry()
        self.pipeline = LlamaIndexIngestion()
    
    def _load_registry(self) -> Dict:
        """Load ingestion registry."""
        import json
        
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        return {"files": {}}
    
    def _save_registry(self):
        """Save ingestion registry."""
        import json
        
        self.registry["last_updated"] = datetime.now().isoformat()
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def _get_file_hash(self, file_path: str) -> str:
        """Get file hash (mtime + size)."""
        path = Path(file_path)
        return f"{path.stat().st_mtime}_{path.stat().st_size}"
    
    def ingest_new_files(self, directory: str, category: str):
        """Only ingest new or modified files."""
        dir_path = Path(directory)
        pdf_files = list(dir_path.glob("*.pdf"))
        
        new_files = []
        for pdf_file in pdf_files:
            file_key = str(pdf_file)
            current_hash = self._get_file_hash(file_key)
            
            if file_key not in self.registry["files"]:
                new_files.append(pdf_file)
            elif self.registry["files"][file_key]["hash"] != current_hash:
                new_files.append(pdf_file)
        
        logger.info(f"Found {len(new_files)} new/updated files out of {len(pdf_files)} total")
        
        # Process new files
        for pdf_file in new_files:
            try:
                self.pipeline.ingest_pdf(
                    str(pdf_file),
                    metadata={"category": category}
                )
                
                # Update registry
                self.registry["files"][str(pdf_file)] = {
                    "hash": self._get_file_hash(str(pdf_file)),
                    "ingested_at": datetime.now().isoformat()
                }
                
                logger.info(f"✓ {pdf_file.name}")
            except Exception as e:
                logger.error(f"✗ {pdf_file.name}: {e}")
        
        self._save_registry()
        return len(new_files)


# ============================================================================
# CLI for easy usage
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest PDFs into 45Labs")
    parser.add_argument("command", choices=["file", "dir", "library", "incremental"])
    parser.add_argument("--path", help="Path to file or directory")
    parser.add_argument("--category", help="Category (rubrics, guides, etc.)")
    parser.add_argument("--subject", help="Subject (Biology, Chemistry, etc.)")
    parser.add_argument("--year", type=int, help="Year (2024, etc.)")
    
    args = parser.parse_args()
    
    pipeline = LlamaIndexIngestion()
    
    if args.command == "file":
        # Ingest single PDF
        metadata = {}
        if args.category:
            metadata["category"] = args.category
        if args.subject:
            metadata["subject"] = args.subject
        if args.year:
            metadata["year"] = args.year
        
        pipeline.ingest_pdf(args.path, metadata)
    
    elif args.command == "dir":
        # Ingest directory
        metadata = {}
        if args.subject:
            metadata["subject"] = args.subject
        if args.year:
            metadata["year"] = args.year
        
        pipeline.ingest_directory_pipeline(
            args.path,
            args.category or "general",
            metadata
        )
    
    elif args.command == "library":
        # Ingest organized library
        pipeline.ingest_organized_library(args.path or "data/documents")
    
    elif args.command == "incremental":
        # Incremental ingestion
        incremental = IncrementalIngestion()
        incremental.ingest_new_files(args.path, args.category or "general")
    
    print("\n✅ Ingestion complete!")
