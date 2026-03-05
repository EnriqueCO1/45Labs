"""
Hybrid Ingestion Pipeline for 45Labs
Uses LlamaIndex for chunking and embedding, Supabase REST API for storage.
This approach avoids PostgreSQL connection issues on macOS.
"""

import os
from pathlib import Path
from typing import List, Dict, Any
import logging
from datetime import datetime

from llama_index.core import SimpleDirectoryReader, Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridIngestion:
    """
    Hybrid ingestion: LlamaIndex for processing, Supabase REST API for storage.
    
    Pipeline:
    1. Load PDFs with LlamaIndex
    2. Chunk with LlamaIndex SentenceSplitter
    3. Embed with OpenAI (via LlamaIndex)
    4. Store in Supabase (via REST API)
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Initialize the hybrid ingestion pipeline.
        
        Args:
            chunk_size: Size of each chunk in tokens
            chunk_overlap: Overlap between chunks in tokens
            embedding_model: OpenAI embedding model
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Configure LlamaIndex for chunking and embedding
        self.embed_model = OpenAIEmbedding(
            model=embedding_model,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        Settings.embed_model = self.embed_model
        
        # Initialize Supabase client (REST API - works perfectly!)
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")
        
        self.supabase: Client = create_client(supabase_url, supabase_key)
        
        # Initialize node parser (chunker)
        self.node_parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        logger.info(f"Initialized hybrid pipeline: chunk_size={chunk_size}, model={embedding_model}")
    
    def load_pdf(self, pdf_path: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """
        Load a single PDF file using LlamaIndex.
        
        Args:
            pdf_path: Path to PDF
            metadata: Additional metadata to attach
            
        Returns:
            List of Document objects
        """
        logger.info(f"Loading PDF: {pdf_path}")
        
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
    
    def chunk_and_embed(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Chunk documents and generate embeddings using LlamaIndex.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of dicts with text, embedding, and metadata
        """
        logger.info(f"Chunking and embedding {len(documents)} documents...")
        
        # Parse documents into nodes (chunks)
        nodes = self.node_parser.get_nodes_from_documents(documents, show_progress=True)
        
        logger.info(f"Created {len(nodes)} chunks")
        
        # Generate embeddings for all nodes
        logger.info("Generating embeddings with OpenAI...")
        
        chunks_with_embeddings = []
        
        # Process in batches to show progress
        batch_size = 10
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i + batch_size]
            
            for node in batch:
                # Get embedding
                embedding = self.embed_model.get_text_embedding(node.get_content())
                
                chunks_with_embeddings.append({
                    "content": node.get_content(),
                    "embedding": embedding,
                    "metadata": node.metadata,
                    "node_id": node.node_id
                })
            
            logger.info(f"Embedded {min(i + batch_size, len(nodes))}/{len(nodes)} chunks")
        
        return chunks_with_embeddings
    
    def store_in_supabase(self, chunks: List[Dict[str, Any]]):
        """
        Store chunks in Supabase using REST API.
        
        Args:
            chunks: List of dicts with content, embedding, metadata
        """
        logger.info(f"Storing {len(chunks)} chunks in Supabase...")
        
        # Prepare documents for insertion
        documents = []
        for chunk in chunks:
            documents.append({
                "content": chunk["content"],
                "embedding": chunk["embedding"],
                "metadata": chunk["metadata"],
                "node_id": chunk.get("node_id"),
                # Extract commonly filtered fields
                "category": chunk["metadata"].get("category"),
                "file_name": chunk["metadata"].get("file_name"),
                "subject": chunk["metadata"].get("subject"),
                "year": chunk["metadata"].get("year"),
            })
        
        # Insert in batches (Supabase has limits)
        batch_size = 100
        total_inserted = 0
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            try:
                result = self.supabase.table("documents").insert(batch).execute()
                total_inserted += len(batch)
                logger.info(f"Inserted {total_inserted}/{len(documents)} chunks")
            except Exception as e:
                logger.error(f"Error inserting batch: {e}")
                raise
        
        logger.info(f"✓ Successfully stored {total_inserted} chunks in Supabase")
    
    def ingest_pdf(
        self,
        pdf_path: str,
        metadata: Dict[str, Any] = None
    ):
        """
        Complete pipeline: Load PDF → Chunk → Embed → Store.
        
        Args:
            pdf_path: Path to PDF
            metadata: Metadata to attach
        """
        # Load
        documents = self.load_pdf(pdf_path, metadata)
        
        # Chunk and embed
        chunks = self.chunk_and_embed(documents)
        
        # Store
        self.store_in_supabase(chunks)
        
        logger.info(f"✓ Ingested {pdf_path}")
    
    def ingest_directory(
        self,
        directory: str,
        category: str,
        metadata: Dict[str, Any] = None
    ):
        """
        Complete pipeline for a directory.
        
        Args:
            directory: Directory path
            category: Document category (rubrics, guides, etc.)
            metadata: Additional metadata
        """
        # Prepare metadata
        full_metadata = {"category": category}
        if metadata:
            full_metadata.update(metadata)
        
        # Load
        documents = self.load_directory(directory, full_metadata)
        
        if not documents:
            logger.warning(f"No documents found in {directory}")
            return
        
        # Chunk and embed
        chunks = self.chunk_and_embed(documents)
        
        # Store
        self.store_in_supabase(chunks)
        
        logger.info(f"✓ Ingested directory: {directory}")
    
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
                    chunks = self.chunk_and_embed(documents)
                    self.store_in_supabase(chunks)
                    total_docs += len(documents)
                    logger.info(f"✓ {category}: {len(documents)} documents, {len(chunks)} chunks")
                else:
                    logger.info(f"No PDFs found in {category}")
                    
            except Exception as e:
                logger.error(f"✗ Error processing {category}: {e}")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"TOTAL: {total_docs} documents ingested")
        logger.info(f"{'='*60}")
        
        return total_docs


# ============================================================================
# CLI for easy usage
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest PDFs into 45Labs")
    parser.add_argument("command", choices=["file", "dir", "library"])
    parser.add_argument("--path", help="Path to file or directory")
    parser.add_argument("--category", help="Category (rubrics, guides, etc.)")
    parser.add_argument("--subject", help="Subject (Biology, Chemistry, etc.)")
    parser.add_argument("--year", type=int, help="Year (2024, etc.)")
    parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=128, help="Chunk overlap")
    
    args = parser.parse_args()
    
    pipeline = HybridIngestion(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
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
        
        pipeline.ingest_directory(
            args.path,
            args.category or "general",
            metadata
        )
    
    elif args.command == "library":
        # Ingest organized library
        pipeline.ingest_organized_library(args.path or "data/documents")
    
    print("\n✅ Ingestion complete!")
