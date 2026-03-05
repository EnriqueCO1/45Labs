"""
Vector Store for 45Labs RAG System
Implements FAISS-based vector storage and retrieval.
"""

import os
import json
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple
#from sentence_transformers import SentenceTransformer
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS-based vector store for RAG retrieval."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", index_path: str = None):
        """
        Initialize the vector store.
        
        Args:
            model_name: Sentence transformer model for embeddings
            index_path: Path to save/load FAISS index
        """
        self.model_name = model_name
        BASE_DIR = Path(__file__).resolve().parents[2]  # dos niveles arriba de src/rag
        self.index_path = Path(index_path) if index_path else BASE_DIR / "data" / "vectorstore" / "faiss_index"
        self.model = None
        #self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
        self.dimension = None
        
        logger.info(f"Initialized VectorStore with model: {model_name}")

    def _load_model(self):
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            from sentence_transformers import SentenceTransformer
            logger.info("yeeeeedjnddjcn")
            self.model = SentenceTransformer(self.model_name, device="cpu")


    def create_index(self, chunks: List[Dict[str, Any]]):
        """
        Create FAISS index from text chunks.
        
        Args:
            chunks: List of chunk dictionaries with 'text' and 'metadata'
        """
        logger.info(f"Creating index from {len(chunks)} chunks")
        
        # Store chunks
        self.chunks = chunks
        
        # Generate embeddings
        texts = [chunk["text"] for chunk in chunks]
        logger.info("Generating embeddings...")
        #ADDED RECENTLY
        self._load_model()
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Convert to numpy array
        embeddings = np.array(embeddings).astype('float32')
        self.dimension = embeddings.shape[1]
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        logger.info(f"Created FAISS index with {self.index.ntotal} vectors")
    
    def save_index(self):
        """Save FAISS index and chunks to disk."""
        if self.index is None:
            raise ValueError("No index to save. Call create_index() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{self.index_path}.faiss")
        
        # Save chunks as JSON
        with open(f"{self.index_path}.json", 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved index to: {self.index_path}")
    
    def load_index(self) -> bool:
        """
        Load FAISS index and chunks from disk.
        
        Returns:
            True if successful, False otherwise
        """
        faiss_path = f"{self.index_path}.faiss"
        json_path = f"{self.index_path}.json"
        
        if not os.path.exists(faiss_path) or not os.path.exists(json_path):
            logger.warning(f"Index files not found at {self.index_path}")
            return False
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(faiss_path)
            self.dimension = self.index.d
            
            # Load chunks
            with open(json_path, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
            
            logger.info(f"Loaded index with {self.index.ntotal} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar chunks with scores
        """
        if self.index is None:
            raise ValueError("Index not created. Call create_index() or load_index() first.")
        
        # Generate query embedding
        self._load_model()
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        # Format results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.chunks):  # Valid index
                chunk = self.chunks[idx]
                results.append({
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                    "score": float(score),  # Cosine similarity
                    "index": int(idx)
                })
        
        return results
    
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
        results = self.search(query, k=max_chunks * 2)  # Get more to filter
        
        # Filter by score
        filtered_results = [r for r in results if r["score"] >= min_score]
        filtered_results = filtered_results[:max_chunks]
        
        if not filtered_results:
            logger.warning(f"No relevant chunks found for query: {query}")
            return ""
        
        # Format context
        context_parts = []
        for result in filtered_results:
            metadata = result["metadata"]
            text = result["text"]
            score = result["score"]
            
            context_parts.append(
                f"[Source: {metadata.get('source', 'Unknown')}, "
                f"Component: {metadata.get('component', 'Unknown')}, "
                f"Similarity: {score:.3f}]\n{text}\n"
            )
        
        return "\n---\n".join(context_parts)


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
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        vectorstore.create_index(chunks)
        vectorstore.save_index()
    else:
        raise ValueError(f"Chunks file not found: {chunks_path}")
    
    return vectorstore


if __name__ == "__main__":
    # Example usage
    BASE_DIR = Path(__file__).resolve().parents[2]
    chunks_path = BASE_DIR / "data" / "rubrics" / "chunks.json"
    
    # Initialize vector store
    vectorstore = initialize_vectorstore(chunks_path, force_recreate=True)
    
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
