"""
Vector Store for 45Labs RAG System
Supabase + pgvector implementation with OpenAI embeddings.
"""

import os
import json
from typing import List, Dict, Any
import logging
from pathlib import Path
from supabase import create_client, Client
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """Supabase pgvector-based vector store for RAG retrieval."""
    
    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        """
        Initialize the vector store.
        
        Args:
            embedding_model: OpenAI embedding model to use
        """
        self.embedding_model = embedding_model
        self.embedding_dimension = 1536  # text-embedding-3-small dimension
        
        # Initialize Supabase client
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            raise ValueError(
                "Missing Supabase credentials. Please set SUPABASE_URL and SUPABASE_KEY environment variables."
            )
        
        self.supabase: Client = create_client(supabase_url, supabase_key)
        
        # Initialize OpenAI client
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("Missing OpenAI API key. Please set OPENAI_API_KEY environment variable.")
        
        self.openai_client = OpenAI(api_key=openai_api_key)
        
        logger.info(f"Initialized Supabase VectorStore with model: {embedding_model}")
    
    def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise
    
    def _get_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Get embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts per batch
            
        Returns:
            List of embedding vectors
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"Processing embeddings batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            try:
                response = self.openai_client.embeddings.create(
                    model=self.embedding_model,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error getting embeddings for batch: {e}")
                raise
        
        return all_embeddings
    
    def create_index(self, chunks: List[Dict[str, Any]]):
        """
        Create vector store from text chunks.
        
        Args:
            chunks: List of chunk dictionaries with 'text' and 'metadata'
        """
        logger.info(f"Creating vector store from {len(chunks)} chunks")
        
        # Extract texts
        texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings
        logger.info("Generating embeddings with OpenAI...")
        embeddings = self._get_embeddings_batch(texts)
        
        # Prepare documents for insertion
        documents = []
        for chunk, embedding in zip(chunks, embeddings):
            documents.append({
                "content": chunk["text"],
                "metadata": chunk["metadata"],
                "embedding": embedding
            })
        
        # Insert into Supabase in batches
        batch_size = 100
        logger.info(f"Inserting {len(documents)} documents into Supabase...")
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            try:
                self.supabase.table("documents").insert(batch).execute()
                logger.info(f"Inserted batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
            except Exception as e:
                logger.error(f"Error inserting batch: {e}")
                raise
        
        logger.info(f"Successfully created vector store with {len(documents)} documents")
    
    def load_index(self) -> bool:
        """
        Check if vector store has documents.
        
        Returns:
            True if documents exist, False otherwise
        """
        try:
            result = self.supabase.table("documents").select("id", count="exact").limit(1).execute()
            
            if result.count and result.count > 0:
                logger.info(f"Vector store loaded with {result.count} documents")
                return True
            else:
                logger.warning("No documents found in vector store")
                return False
        except Exception as e:
            logger.error(f"Error checking vector store: {e}")
            return False
    
    def search(self, query: str, k: int = 5, min_score: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for similar chunks.
        
        Args:
            query: Search query
            k: Number of results to return
            min_score: Minimum similarity score (0-1)
            
        Returns:
            List of similar chunks with scores
        """
        try:
            # Generate query embedding
            query_embedding = self._get_embedding(query)
            
            # Call the match_documents RPC function
            result = self.supabase.rpc(
                "match_documents",
                {
                    "query_embedding": query_embedding,
                    "match_count": k,
                    "match_threshold": min_score
                }
            ).execute()
            
            # Format results
            results = []
            for doc in result.data:
                results.append({
                    "text": doc["content"],
                    "metadata": doc["metadata"],
                    "score": float(doc["similarity"]),
                    "id": doc["id"]
                })
            
            logger.info(f"Found {len(results)} results for query")
            return results
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []
    
    def get_context(self, query: str, max_chunks: int = 3, min_score: float = 0.3) -> str:
        """
        Get context string for LLM from retrieved chunks.
        
        Args:
            query: Search query
            max_chunks: Maximum number of chunks to include
            min_score: Minimum similarity score to include chunk
            
        Returns:
            Formatted context string
        """
        results = self.search(query, k=max_chunks, min_score=min_score)
        
        if not results:
            logger.warning(f"No relevant chunks found for query: {query}")
            return ""
        
        # Format context
        context_parts = []
        for result in results:
            metadata = result["metadata"]
            text = result["text"]
            score = result["score"]
            
            context_parts.append(
                f"[Source: {metadata.get('source', 'Unknown')}, "
                f"Component: {metadata.get('component', 'Unknown')}, "
                f"Similarity: {score:.3f}]\n{text}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def save_index(self):
        """
        No-op for Supabase (data is already saved in cloud).
        Kept for compatibility with existing code.
        """
        logger.info("Documents are automatically saved in Supabase")
        pass
    
    def clear_documents(self):
        """Delete all documents from the vector store."""
        try:
            self.supabase.table("documents").delete().neq("id", 0).execute()
            logger.info("Cleared all documents from vector store")
        except Exception as e:
            logger.error(f"Error clearing documents: {e}")
            raise


def initialize_vectorstore(chunks_path: str = None, force_recreate: bool = False) -> VectorStore:
    """
    Initialize or load vector store.
    
    Args:
        chunks_path: Path to chunks JSON file
        force_recreate: Force recreation of index
        
    Returns:
        Initialized VectorStore
    """
    vectorstore = VectorStore()
    
    # Try to load existing index
    if not force_recreate:
        if vectorstore.load_index():
            return vectorstore
    
    # Create new index from chunks
    if chunks_path and os.path.exists(chunks_path):
        logger.info("Creating new vector store from chunks...")
        
        # Clear existing documents if force recreate
        if force_recreate:
            vectorstore.clear_documents()
        
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        vectorstore.create_index(chunks)
    else:
        raise ValueError(f"Chunks file not found: {chunks_path}")
    
    return vectorstore


if __name__ == "__main__":
    # Example usage
    from dotenv import load_dotenv
    load_dotenv()
    
    BASE_DIR = Path(__file__).resolve().parents[2]
    chunks_path = BASE_DIR / "data" / "rubrics" / "chunks.json"
    
    # Initialize vector store
    vectorstore = initialize_vectorstore(str(chunks_path), force_recreate=False)
    
    # Test search
    query = "What should be included in the conclusion of an Extended Essay?"
    results = vectorstore.search(query, k=3)
    
    print(f"\nSearch results for: '{query}'")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.3f}")
        print(f"   Text preview: {result['text'][:150]}...")
    
    # Test context generation
    context = vectorstore.get_context(query, max_chunks=2)
    print(f"\nGenerated context:\n{context[:500]}...")
