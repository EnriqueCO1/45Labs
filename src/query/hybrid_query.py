"""
Hybrid Query Pipeline for 45Labs
Uses OpenAI for embeddings and Supabase REST API for retrieval.
"""

import os
from typing import List, Dict, Any, Optional
import logging
from dotenv import load_dotenv

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridQueryPipeline:
    """
    Hybrid query pipeline using OpenAI embeddings and Supabase REST API.
    """
    
    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4o-mini",
        top_k: int = 5,
        similarity_threshold: float = 0.3
    ):
        """
        Initialize the query pipeline.
        
        Args:
            embedding_model: OpenAI embedding model
            llm_model: LLM model to use
            top_k: Number of chunks to retrieve
            similarity_threshold: Minimum similarity score
        """
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        
        # Initialize embedding model
        self.embed_model = OpenAIEmbedding(
            model=embedding_model,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize LLM
        if "gpt" in llm_model:
            self.llm = OpenAI(
                model=llm_model,
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.3
            )
        elif "claude" in llm_model:
            self.llm = Anthropic(
                model=llm_model,
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                temperature=0.3
            )
        
        # Initialize Supabase client
        self.supabase: Client = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY")
        )
        
        logger.info(f"Initialized query pipeline: model={llm_model}")
    
    def get_query_embedding(self, query: str) -> List[float]:
        """
        Get embedding for a query string.
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector
        """
        return self.embed_model.get_text_embedding(query)
    
    def search(
        self,
        query: str,
        k: int = None,
        category: str = None,
        subject: str = None,
        year: int = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using Supabase RPC function.
        
        Args:
            query: Search query
            k: Number of results (default: self.top_k)
            category: Filter by category
            subject: Filter by subject
            year: Filter by year
            
        Returns:
            List of similar documents with scores
        """
        k = k or self.top_k
        
        # Get query embedding
        query_embedding = self.get_query_embedding(query)
        
        # Call Supabase RPC function
        try:
            result = self.supabase.rpc(
                "match_documents",
                {
                    "query_embedding": query_embedding,
                    "match_count": k,
                    "match_threshold": self.similarity_threshold,
                    "filter_category": category,
                    "filter_subject": subject,
                    "filter_year": year
                }
            ).execute()
            
            logger.info(f"Found {len(result.data)} results")
            return result.data
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def get_context(
        self,
        query: str,
        max_chunks: int = None,
        category: str = None
    ) -> str:
        """
        Get formatted context string from retrieved chunks.
        
        Args:
            query: Search query
            max_chunks: Maximum chunks to include
            category: Filter by category
            
        Returns:
            Formatted context string
        """
        max_chunks = max_chunks or self.top_k
        
        results = self.search(query, k=max_chunks, category=category)
        
        if not results:
            logger.warning(f"No results found for query: {query}")
            return ""
        
        # Format context
        context_parts = []
        for result in results:
            metadata = result.get("metadata", {})
            content = result.get("content", "")
            similarity = result.get("similarity", 0)
            
            context_parts.append(
                f"[Source: {metadata.get('file_name', 'Unknown')}, "
                f"Category: {result.get('category', 'Unknown')}, "
                f"Similarity: {similarity:.3f}]\n{content}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def query(
        self,
        query_text: str,
        system_prompt: str = None,
        category: str = None
    ) -> str:
        """
        Query the system and get an LLM response.
        
        Args:
            query_text: User's question
            system_prompt: Custom system prompt
            category: Filter by category
            
        Returns:
            LLM response string
        """
        logger.info(f"Processing query: {query_text[:50]}...")
        
        # Get context
        context = self.get_context(query_text, category=category)
        
        if not context:
            return "I couldn't find relevant information in the knowledge base to answer this question."
        
        # Build prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nCONTEXT:\n{context}\n\nQUESTION: {query_text}"
        else:
            full_prompt = f"Context:\n{context}\n\nQuestion: {query_text}\n\nAnswer based only on the context provided:"
        
        # Get LLM response
        response = self.llm.complete(full_prompt)
        
        return str(response)
    
    def query_with_sources(
        self,
        query_text: str,
        system_prompt: str = None,
        category: str = None
    ) -> Dict[str, Any]:
        """
        Query and return both response and source information.
        
        Args:
            query_text: User's question
            system_prompt: Custom system prompt
            category: Filter by category
            
        Returns:
            Dict with response, sources, and metadata
        """
        # Retrieve
        results = self.search(query_text, category=category)
        
        if not results:
            return {
                "response": "No relevant information found.",
                "sources": [],
                "num_sources": 0
            }
        
        # Format context
        context = self.get_context(query_text, category=category)
        
        # Build prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nCONTEXT:\n{context}\n\nQUESTION: {query_text}"
        else:
            full_prompt = f"Context:\n{context}\n\nQuestion: {query_text}\n\nAnswer:"
        
        # Get response
        response = self.llm.complete(full_prompt)
        
        # Format sources
        sources = []
        for result in results:
            sources.append({
                "file_name": result.get("metadata", {}).get("file_name", "Unknown"),
                "category": result.get("category"),
                "similarity": result.get("similarity"),
                "preview": result.get("content", "")[:200]
            })
        
        return {
            "response": str(response),
            "sources": sources,
            "num_sources": len(sources)
        }


class IBFeedbackEngine:
    """
    Specialized query engine for IB Extended Essay feedback.
    """
    
    def __init__(self):
        """Initialize IB feedback engine."""
        self.pipeline = HybridQueryPipeline(
            llm_model="claude-3-5-sonnet-20241022",
            top_k=5
        )
        
        # IB-specific prompts
        self.rubric_prompt = """You are an academic advisor aligned with the International Baccalaureate (IB) curriculum.
Your role is to provide guidance based STRICTLY on the official IB documentation provided in the context.

IMPORTANT RULES:
- Use ONLY the information from the context provided
- Do NOT invent criteria or make up information
- If information is not in the context, explicitly state "This is not specified in the official IB guide"
- Maintain an academic and supportive tone
- Always cite the specific part of the rubric you're referencing

Answer the student's question using the context provided."""
        
        self.essay_prompt = """You are an IB Extended Essay examiner providing feedback based on official rubrics.

Analyze the essay CRITERION BY CRITERION according to the official IB rubric provided in the context.

For each criterion (A-E), provide:
- STRENGTHS: What the student has done well
- WEAKNESSES: Areas needing improvement  
- HOW TO IMPROVE: Specific suggestions based on the rubric

Structure your feedback with clear headings for each criterion.
Use ONLY information from the official rubric context.
Do NOT rewrite or edit the student's text.
Be constructive and specific."""
    
    def answer_rubric_question(self, question: str) -> Dict[str, Any]:
        """Answer a question about the IB rubric."""
        return self.pipeline.query_with_sources(
            query_text=question,
            system_prompt=self.rubric_prompt,
            category="rubrics"
        )
    
    def provide_essay_feedback(self, essay_text: str) -> Dict[str, Any]:
        """Provide feedback on an Extended Essay."""
        # Create query from essay
        query_text = f"Extended Essay assessment criteria for analyzing: {essay_text[:500]}"
        
        return self.pipeline.query_with_sources(
            query_text=query_text,
            system_prompt=self.essay_prompt,
            category="rubrics"
        )


# ============================================================================
# CLI for testing
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Query 45Labs knowledge base")
    parser.add_argument("query", help="Your question")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model")
    parser.add_argument("--category", help="Filter by category")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument("--sources", action="store_true", help="Show sources")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = HybridQueryPipeline(
        llm_model=args.model,
        top_k=args.top_k
    )
    
    if args.sources:
        # Query with sources
        result = pipeline.query_with_sources(args.query, category=args.category)
        
        print(f"\nResponse:\n{result['response']}\n")
        print(f"\nSources ({result['num_sources']}):")
        for i, source in enumerate(result['sources'], 1):
            print(f"{i}. {source['file_name']} (similarity: {source['similarity']:.3f})")
            print(f"   {source['preview']}...\n")
    else:
        # Just get response
        response = pipeline.query(args.query, category=args.category)
        print(f"\nResponse:\n{response}\n")
