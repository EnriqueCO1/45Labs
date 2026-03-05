"""
Cloud-Based Ingestion Pipeline for 45Labs
Reads PDFs from Supabase Storage and metadata from Supabase database.
Perfect for handling 500+ PDFs without local storage.
"""

import os
import io
from pathlib import Path
from typing import List, Dict, Any
import logging
from datetime import datetime
import tempfile

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


class CloudIngestion:
    """
    Cloud-based ingestion: PDFs in Supabase Storage, metadata in Supabase tables.
    
    Architecture:
    1. PDFs stored in Supabase Storage bucket
    2. Metadata tracked in pdf_metadata table
    3. Processing happens on-demand (download → process → upload chunks)
    4. No local storage needed!
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        embedding_model: str = "text-embedding-3-small"
    ):
        """Initialize cloud ingestion pipeline."""
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Configure LlamaIndex
        self.embed_model = OpenAIEmbedding(
            model=embedding_model,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        Settings.embed_model = self.embed_model
        
        # Initialize Supabase client
        self.supabase: Client = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY")
        )
        
        # Initialize node parser
        self.node_parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        logger.info(f"Initialized cloud pipeline: chunk_size={chunk_size}")
    
    def download_pdf_from_storage(self, storage_path: str) -> bytes:
        """
        Download PDF from Supabase Storage.
        
        Args:
            storage_path: Path in storage bucket (e.g., "rubrics/biology_hl.pdf")
            
        Returns:
            PDF file bytes
        """
        logger.info(f"Downloading: {storage_path}")
        
        try:
            # Download from Supabase Storage
            response = self.supabase.storage.from_("pdfs").download(storage_path)
            
            if not response:
                raise Exception(f"Failed to download {storage_path}")
            
            logger.info(f"Downloaded {len(response)} bytes")
            return response
            
        except Exception as e:
            logger.error(f"Error downloading {storage_path}: {e}")
            raise
    
    def process_pdf_from_bytes(self, pdf_bytes: bytes, metadata: Dict[str, Any]) -> List[Document]:
        """
        Process PDF from bytes without saving to disk.
        
        Args:
            pdf_bytes: PDF file content
            metadata: Metadata to attach
            
        Returns:
            List of Document objects
        """
        # Create temporary file (automatically cleaned up)
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(pdf_bytes)
            tmp_path = tmp_file.name
        
        try:
            # Load PDF
            reader = SimpleDirectoryReader(input_files=[tmp_path])
            documents = reader.load_data()
            
            # Add metadata
            for doc in documents:
                doc.metadata.update(metadata)
                doc.metadata["ingested_at"] = datetime.now().isoformat()
                
                # Clean up LlamaIndex metadata
                doc.metadata.pop("file_path", None)
                doc.metadata.pop("file_size", None)
                doc.metadata.pop("file_type", None)
                doc.metadata.pop("creation_date", None)
                doc.metadata.pop("last_modified_date", None)
            
            return documents
            
        finally:
            # Clean up temp file
            os.unlink(tmp_path)
    
    def chunk_and_embed(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Chunk documents and generate embeddings."""
        
        logger.info(f"Chunking {len(documents)} documents...")
        
        # Parse into nodes
        nodes = self.node_parser.get_nodes_from_documents(documents, show_progress=True)
        
        logger.info(f"Created {len(nodes)} chunks")
        
        # Track chunks per document
        doc_chunk_counts = {}
        for node in nodes:
            doc_id = node.ref_doc_id
            if doc_id not in doc_chunk_counts:
                doc_chunk_counts[doc_id] = {"count": 0, "nodes": []}
            doc_chunk_counts[doc_id]["nodes"].append(node)
            doc_chunk_counts[doc_id]["count"] += 1
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        
        chunks_with_embeddings = []
        batch_size = 10
        
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i + batch_size]
            
            for node in batch:
                embedding = self.embed_model.get_text_embedding(node.get_content())
                
                # Add chunk position metadata
                doc_id = node.ref_doc_id
                chunk_index = doc_chunk_counts[doc_id]["nodes"].index(node)
                total_chunks = doc_chunk_counts[doc_id]["count"]
                
                enhanced_metadata = node.metadata.copy()
                enhanced_metadata["chunk_index"] = chunk_index
                enhanced_metadata["total_chunks"] = total_chunks
                
                chunks_with_embeddings.append({
                    "content": node.get_content(),
                    "embedding": embedding,
                    "metadata": enhanced_metadata,
                    "node_id": node.node_id
                })
            
            logger.info(f"Embedded {min(i + batch_size, len(nodes))}/{len(nodes)} chunks")
        
        return chunks_with_embeddings
    
    def store_chunks(self, chunks: List[Dict[str, Any]]):
        """Store chunks in Supabase documents table."""
        
        logger.info(f"Storing {len(chunks)} chunks...")
        
        documents = []
        for chunk in chunks:
            documents.append({
                "content": chunk["content"],
                "embedding": chunk["embedding"],
                "metadata": chunk["metadata"],
                "node_id": chunk.get("node_id"),
                "category": chunk["metadata"].get("category"),
                "file_name": chunk["metadata"].get("file_name"),
                "subject": chunk["metadata"].get("subject"),
                "year": chunk["metadata"].get("year"),
                "level": chunk["metadata"].get("level"),
                "component": chunk["metadata"].get("component"),
            })
        
        # Insert in batches
        batch_size = 100
        total_inserted = 0
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            try:
                self.supabase.table("documents").insert(batch).execute()
                total_inserted += len(batch)
                logger.info(f"Inserted {total_inserted}/{len(documents)} chunks")
            except Exception as e:
                logger.error(f"Error inserting batch: {e}")
                raise
        
        logger.info(f"✓ Stored {total_inserted} chunks")
    
    def update_pdf_status(self, pdf_id: int, status: str, chunks: int = None, error: str = None):
        """Update PDF processing status in database."""
        
        try:
            self.supabase.rpc(
                "update_pdf_status",
                {
                    "pdf_id": pdf_id,
                    "new_status": status,
                    "num_chunks": chunks,
                    "error_msg": error
                }
            ).execute()
            
        except Exception as e:
            logger.error(f"Error updating status: {e}")
    
    def process_single_pdf(self, pdf_record: Dict[str, Any]):
        """
        Process a single PDF from metadata record.
        
        Args:
            pdf_record: Record from pdf_metadata table
        """
        pdf_id = pdf_record['id']
        storage_path = pdf_record['storage_path']
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing PDF ID: {pdf_id}")
        logger.info(f"Path: {storage_path}")
        logger.info(f"Subject: {pdf_record.get('subject')}, Level: {pdf_record.get('level')}")
        logger.info(f"{'='*60}")
        
        try:
            # Update status to processing
            self.update_pdf_status(pdf_id, "processing")
            
            # Download PDF
            pdf_bytes = self.download_pdf_from_storage(storage_path)
            
            # Prepare metadata (exclude internal fields)
            metadata = {
                "pdf_metadata_id": str(pdf_id),
                "category": pdf_record['category'],
                "subject": pdf_record['subject'],
                "level": pdf_record['level'],
                "component": pdf_record.get('component'),
                "year": pdf_record['year'],
                "language": pdf_record['language'],
                "file_name": pdf_record['file_name']
            }
            
            # Process PDF
            documents = self.process_pdf_from_bytes(pdf_bytes, metadata)
            
            # Chunk and embed
            chunks = self.chunk_and_embed(documents)
            
            # Store
            self.store_chunks(chunks)
            
            # Update status to ingested
            self.update_pdf_status(pdf_id, "ingested", chunks=len(chunks))
            
            logger.info(f"✓ Successfully processed: {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"✗ Error processing PDF {pdf_id}: {e}")
            self.update_pdf_status(pdf_id, "failed", error=str(e))
            raise
    
    def process_all_pending(self):
        """Process all PDFs with status='pending'."""
        
        logger.info("Fetching pending PDFs...")
        
        try:
            result = self.supabase.rpc("get_pending_pdfs").execute()
            pending_pdfs = result.data
            
            if not pending_pdfs:
                logger.info("No pending PDFs found")
                return 0
            
            logger.info(f"Found {len(pending_pdfs)} pending PDFs")
            
            total_chunks = 0
            
            for pdf_record in pending_pdfs:
                try:
                    self.process_single_pdf(pdf_record)
                    # Add chunks count if available
                    # (You'd need to track this in process_single_pdf)
                except Exception as e:
                    logger.error(f"Skipping PDF {pdf_record['id']} due to error")
                    continue
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Processed {len(pending_pdfs)} PDFs")
            logger.info(f"{'='*60}")
            
            return len(pending_pdfs)
            
        except Exception as e:
            logger.error(f"Error fetching pending PDFs: {e}")
            raise
    
    def process_by_category(self, category: str):
        """Process all pending PDFs in a specific category."""
        
        logger.info(f"Fetching pending PDFs in category: {category}")
        
        try:
            result = self.supabase.table("pdf_metadata").select("*").eq("status", "pending").eq("category", category).execute()
            
            pdfs = result.data
            
            if not pdfs:
                logger.info(f"No pending PDFs in category: {category}")
                return 0
            
            logger.info(f"Found {len(pdfs)} pending PDFs in {category}")
            
            for pdf_record in pdfs:
                try:
                    self.process_single_pdf(pdf_record)
                except Exception as e:
                    logger.error(f"Skipping PDF {pdf_record['id']}")
                    continue
            
            return len(pdfs)
            
        except Exception as e:
            logger.error(f"Error: {e}")
            raise


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Cloud-based PDF ingestion for 45Labs")
    parser.add_argument("command", choices=["all", "category", "single"])
    parser.add_argument("--category", help="Process specific category (rubrics, guides, etc.)")
    parser.add_argument("--pdf-id", type=int, help="Process specific PDF by ID")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=128)
    
    args = parser.parse_args()
    
    pipeline = CloudIngestion(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    if args.command == "all":
        # Process all pending
        pipeline.process_all_pending()
    
    elif args.command == "category":
        # Process specific category
        if not args.category:
            print("Error: --category required")
            exit(1)
        pipeline.process_by_category(args.category)
    
    elif args.command == "single":
        # Process single PDF
        if not args.pdf_id:
            print("Error: --pdf-id required")
            exit(1)
        
        # Fetch PDF record
        result = pipeline.supabase.table("pdf_metadata").select("*").eq("id", args.pdf_id).execute()
        if result.data:
            pipeline.process_single_pdf(result.data[0])
        else:
            print(f"PDF ID {args.pdf_id} not found")
    
    print("\n✅ Processing complete!")
