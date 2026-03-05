"""
Model Router for 45Labs
Routes requests to appropriate LLM based on input type and size.
"""

import os
from typing import Tuple, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelRouter:
    """Routes requests to appropriate LLM models."""
    
    # Model configurations
    QUESTION_MODEL = "gpt-4o-mini"  # For short rubric questions
    ESSAY_MODEL = "claude-3-5-sonnet-20241022"  # For essay feedback
    
    # Word limits (approximate)
    QUESTION_WORD_LIMIT = 100  # Threshold for routing to question model
    
    def __init__(self):
        """Initialize model clients."""
        # Initialize OpenAI client
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=openai_api_key)
            logger.info("OpenAI client initialized")
        else:
            self.openai_client = None
            logger.warning("OPENAI_API_KEY not found in environment variables")
        
        # Initialize Anthropic client
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_api_key:
            import anthropic
            self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
            logger.info("Anthropic client initialized")
        else:
            self.anthropic_client = None
            logger.warning("ANTHROPIC_API_KEY not found in environment variables")
    
    def count_words(self, text: str) -> int:
        """
        Count words in text (approximation for token counting).
        
        Args:
            text: Input text
            
        Returns:
            Number of words
        """
        return len(text.split())
    
    def classify_input(self, text: str, input_type: str = None) -> Tuple[str, int]:
        """
        Classify input and determine which model to use.
        
        Args:
            text: Input text
            input_type: Explicit input type ('question' or 'essay'). If None, auto-detect.
            
        Returns:
            Tuple of (model_name, word_count)
        """
        word_count = self.count_words(text)
        
        if input_type == "question":
            model = self.QUESTION_MODEL
        elif input_type == "essay":
            model = self.ESSAY_MODEL
        else:
            # Auto-detect based on word count
            if word_count <= self.QUESTION_WORD_LIMIT:
                model = self.QUESTION_MODEL
            else:
                model = self.ESSAY_MODEL
        
        logger.info(f"Classified input as {input_type or 'auto-detected'} with {word_count} words. Using {model}")
        return model, word_count
    
    def call_openai(self, prompt: str, model: str = "gpt-4o-mini", max_tokens: int = 1000) -> str:
        """
        Call OpenAI API.
        
        Args:
            prompt: Input prompt
            model: Model name
            max_tokens: Maximum tokens in response
            
        Returns:
            Generated response
        """
        if not self.openai_client:
            return "OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."
        
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.3  # Low temperature for factual responses
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return f"Error generating response: {str(e)}"
    
    def call_claude(self, prompt: str, model: str = "claude-3-5-sonnet-20241022", max_tokens: int = 2000) -> str:
        """
        Call Claude API.
        
        Args:
            prompt: Input prompt
            model: Model name
            max_tokens: Maximum tokens in response
            
        Returns:
            Generated response
        """
        if not self.anthropic_client:
            return "Anthropic API key not configured. Please set ANTHROPIC_API_KEY environment variable."
        
        try:
            response = self.anthropic_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3  # Low temperature for factual responses
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error calling Claude API: {e}")
            return f"Error generating response: {str(e)}"
    
    def generate_response(self, prompt: str, input_type: str = None) -> Dict[str, Any]:
        """
        Generate response using appropriate model.
        
        Args:
            prompt: Input prompt
            input_type: Type of input ('question' or 'essay')
            
        Returns:
            Dictionary with response and metadata
        """
        # Classify input
        model, word_count = self.classify_input(prompt, input_type)
        
        # Set response limits based on model
        if model == self.QUESTION_MODEL:
            max_tokens = 1000
        else:
            max_tokens = 2000
        
        # Generate response
        if "gpt" in model:
            response = self.call_openai(prompt, model, max_tokens)
        else:
            response = self.call_claude(prompt, model, max_tokens)
        
        return {
            "response": response,
            "model_used": model,
            "input_words": word_count,
            "max_tokens": max_tokens
        }
    def count_tokens(self, text: str) -> int:
        # crude token approximation
        return len(text.split())


class SimpleRouter:
    """Simplified router for basic model selection."""
    
    @staticmethod
    def select_model(text: str, input_type: str = None) -> str:
        """
        Simple model selection based on input characteristics.
        
        Args:
            text: Input text
            input_type: Explicit type ('question' or 'essay')
            
        Returns:
            Model name
        """
        if input_type == "question":
            return ModelRouter.QUESTION_MODEL
        elif input_type == "essay":
            return ModelRouter.ESSAY_MODEL
        else:
            # Simple heuristic: if it's a single sentence or question, use GPT-4o-mini
            if len(text.strip().split('.')) <= 2 and len(text) < 300:
                return ModelRouter.QUESTION_MODEL
            else:
                return ModelRouter.ESSAY_MODEL


def check_api_keys() -> Dict[str, bool]:
    """
    Check if required API keys are available.
    
    Returns:
        Dictionary with availability status
    """
    openai_key = bool(os.getenv("OPENAI_API_KEY"))
    anthropic_key = bool(os.getenv("ANTHROPIC_API_KEY"))
    
    return {
        "openai": openai_key,
        "anthropic": anthropic_key,
        "all_available": openai_key and anthropic_key
    }


def count_words(text: str) -> int:
    """Simple word counting function."""
    return len(text.split())


if __name__ == "__main__":
    # Test the router
    router = ModelRouter()
    
    # Test cases
    test_cases = [
        ("What should be in the conclusion?", "question"),
        ("A" * 1000, "essay"),
        ("Short question?", None),  # Auto-detect
    ]
    
    for text, input_type in test_cases:
        model, words = router.classify_input(text, input_type)
        print(f"Input: '{text[:50]}...' | Type: {input_type} | Words: {words} | Model: {model}")
    
    # Check API keys
    keys = check_api_keys()
    print(f"\nAPI Keys available: {keys}")
