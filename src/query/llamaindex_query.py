"""
Query Pipeline for 45Labs
Uses LlamaIndex for retrieval and LLM generation.
Run this at runtime for every user prompt.
"""

import os
from typing import List, Dict, Any, Optional
import logging
from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    get_response_synthesizer
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.vector_stores.supabase import SupabaseVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryPipeline:
    """
    Query pipeline using LlamaIndex.
    Handles: Query → Embedding → Retrieval → LLM Response
    """
    
    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4o-mini",
        similarity_threshold: float = 0.3,
        top_k: int = 5
    ):
        """
        Initialize the query pipeline.
        
        Args:
            embedding_model: OpenAI embedding model
            llm_model: LLM model to use
            similarity_threshold: Minimum similarity score
            top_k: Number of chunks to retrieve
        """
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        
        # Configure LlamaIndex settings
        Settings.embed_model = OpenAIEmbedding(
            model=embedding_model,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Configure LLM
        if "gpt" in llm_model:
            Settings.llm = OpenAI(
                model=llm_model,
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.3
            )
        elif "claude" in llm_model:
            Settings.llm = Anthropic(
                model=llm_model,
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                temperature=0.3
            )
        
        # Initialize Supabase vector store
        self.vector_store = SupabaseVectorStore(
            postgres_connection_string=self._get_connection_string(),
            collection_name="documents",
            dimension=1536
        )
        
        # Create index from existing vector store
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store
        )
        
        logger.info(f"Initialized query pipeline with model={llm_model}")
    
    def _get_connection_string(self) -> str:
        """Build PostgreSQL connection string using Supabase connection pooler."""
        from urllib.parse import quote_plus
        
        supabase_url = os.getenv("SUPABASE_URL")
        
        if not supabase_url:
            raise ValueError("SUPABASE_URL not set")
        
        project_ref = supabase_url.replace("https://", "").replace("http://", "").split(".")[0]
        
        db_password = os.getenv("SUPABASE_DB_PASSWORD")
        if not db_password:
            raise ValueError("SUPABASE_DB_PASSWORD required")
        
        encoded_password = quote_plus(db_password)
        
        # Use connection pooler (better compatibility with macOS)
        return (
            f"postgresql://postgres.{project_ref}:{encoded_password}@"
            f"aws-0-us-west-1.pooler.supabase.com:6543/postgres"
        )
    
    def query(
        self,
        query_text: str,
        custom_prompt: str = None,
        metadata_filters: Dict[str, Any] = None
    ) -> str:
        """
        Query the system and get an LLM response.
        
        Args:
            query_text: User's question
            custom_prompt: Custom system prompt (optional)
            metadata_filters: Filter by metadata (category, subject, etc.)
            
        Returns:
            LLM response string
        """
        logger.info(f"Processing query: {query_text[:50]}...")
        
        # Create retriever
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.top_k,
            # filters=metadata_filters  # TODO: Add metadata filtering
        )
        
        # Create response synthesizer with custom prompt if provided
        response_synthesizer = get_response_synthesizer(
            response_mode="compact",
            text_qa_template=custom_prompt
        )
        
        # Create query engine
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=self.similarity_threshold)
            ]
        )
        
        # Execute query
        response = query_engine.query(query_text)
        
        return str(response)
    
    def retrieve_only(
        self,
        query_text: str,
        metadata_filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks without LLM generation.
        
        Args:
            query_text: User's question
            metadata_filters: Filter by metadata
            
        Returns:
            List of retrieved nodes with scores
        """
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.top_k
        )
        
        nodes = retriever.retrieve(query_text)
        
        # Format results
        results = []
        for node in nodes:
            if node.score >= self.similarity_threshold:
                results.append({
                    "text": node.text,
                    "score": node.score,
                    "metadata": node.metadata,
                    "node_id": node.node_id
                })
        
        return results
    
    def query_with_context(
        self,
        query_text: str,
        system_prompt: str,
        metadata_filters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Query and return both context and response.
        
        Args:
            query_text: User's question
            system_prompt: System prompt for LLM
            metadata_filters: Filter by metadata
            
        Returns:
            Dictionary with context and response
        """
        # Retrieve context
        context_nodes = self.retrieve_only(query_text, metadata_filters)
        
        if not context_nodes:
            return {
                "response": "No relevant information found in the knowledge base.",
                "context": [],
                "num_sources": 0
            }
        
        # Format context for LLM
        context_text = "\n\n---\n\n".join([
            f"[Source: {node['metadata'].get('file_name', 'Unknown')}, "
            f"Score: {node['score']:.3f}]\n{node['text']}"
            for node in context_nodes
        ])
        
        # Create full prompt
        full_prompt = f"{system_prompt}\n\nCONTEXT:\n{context_text}\n\nQUESTION: {query_text}"
        
        # Get LLM response
        response = Settings.llm.complete(full_prompt)
        
        return {
            "response": str(response),
            "context": context_nodes,
            "num_sources": len(context_nodes)
        }


class IBFeedbackEngine:
    """
    Specialized query engine for IB Extended Essay feedback.
    Wraps QueryPipeline with IB-specific prompts.
    """
    
    def __init__(self):
        """Initialize IB feedback engine."""
        self.pipeline = QueryPipeline(
            llm_model="claude-3-5-sonnet-20241022",  # Better for long feedback
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
    
    def answer_rubric_question(self, question: str) -> str:
        """
        Answer a question about the IB rubric.
        
        Args:
            question: Student's question
            
        Returns:
            Answer based on rubric
        """
        return self.pipeline.query_with_context(
            query_text=question,
            system_prompt=self.rubric_prompt,
            metadata_filters={"category": "rubrics"}
        )
    
    def provide_essay_feedback(self, essay_text: str) -> str:
        """
        Provide feedback on an Extended Essay.
        
        Args:
            essay_text: Student's essay
            
        Returns:
            Detailed feedback
        """
        # For essay feedback, we query for rubric context
        query_text = f"Extended Essay assessment criteria for: {essay_text[:500]}"
        
        return self.pipeline.query_with_context(
            query_text=query_text,
            system_prompt=self.essay_prompt,
            metadata_filters={"category": "rubrics"}
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
    parser.add_argument("--retrieve-only", action="store_true", help="Only retrieve, don't generate")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = QueryPipeline(
        llm_model=args.model,
        top_k=args.top_k
    )
    
    # Prepare filters
    filters = {}
    if args.category:
        filters["category"] = args.category
    
    if args.retrieve_only:
        # Just retrieve context
        results = pipeline.retrieve_only(args.query, filters)
        
        print(f"\nFound {len(results)} relevant chunks:\n")
        for i, result in enumerate(results, 1):
            print(f"{i}. Score: {result['score']:.3f}")
            print(f"   Source: {result['metadata'].get('file_name', 'Unknown')}")
            print(f"   Text: {result['text'][:150]}...\n")
    else:
        # Full query with LLM
        response = pipeline.query(args.query, metadata_filters=filters)
        
        print(f"\nResponse:\n{response}\n")
