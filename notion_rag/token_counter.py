"""
Token counting utilities for the Notion RAG system.
Provides accurate token counting for different models.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    tiktoken = None

logger = logging.getLogger(__name__)


@dataclass
class TokenCount:
    """Represents token count information."""
    
    count: int
    method: str  # "exact", "approximate", "tiktoken"
    model: str
    confidence: float  # 0.0 to 1.0, how confident we are in the count


class TokenCounter:
    """Utility for counting tokens in text."""
    
    # Approximate token-to-character ratios for different languages
    TOKEN_RATIOS = {
        "english": 4.0,      # ~4 characters per token
        "code": 3.5,         # ~3.5 characters per token (more punctuation)
        "mixed": 3.8,        # ~3.8 characters per token (general purpose)
    }
    
    # Common token counts for special tokens
    SPECIAL_TOKENS = {
        "system_prompt": 50,     # Approximate tokens for system prompts
        "user_prefix": 10,       # Tokens for "User: " prefix
        "assistant_prefix": 10,  # Tokens for "Assistant: " prefix
        "context_separator": 5,  # Tokens for context separators
    }
    
    def __init__(self, model: str = "gemini-2.5-flash-lite-preview"):
        """
        Initialize the token counter.
        
        Args:
            model: Model name for token counting
        """
        self.model = model
        self.tiktoken_encoder = None
        
        # Try to initialize tiktoken for better accuracy
        if TIKTOKEN_AVAILABLE:
            try:
                # Use cl100k_base encoding (closest to modern models)
                self.tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
                logger.info("Initialized tiktoken encoder for accurate token counting")
            except Exception as e:
                logger.warning(f"Failed to initialize tiktoken: {e}")
    
    def count_tokens(self, text: str, method: str = "auto") -> TokenCount:
        """
        Count tokens in text using specified method.
        
        Args:
            text: Text to count tokens in
            method: Counting method ("auto", "exact", "approximate", "tiktoken")
            
        Returns:
            TokenCount object with count and metadata
        """
        if not text:
            return TokenCount(count=0, method="empty", model=self.model, confidence=1.0)
        
        if method == "auto":
            # Use tiktoken if available, otherwise approximate
            if self.tiktoken_encoder:
                method = "tiktoken"
            else:
                method = "approximate"
        
        if method == "tiktoken" and self.tiktoken_encoder:
            return self._count_with_tiktoken(text)
        elif method == "approximate":
            return self._count_approximate(text)
        else:
            raise ValueError(f"Unknown counting method: {method}")
    
    def _count_with_tiktoken(self, text: str) -> TokenCount:
        """Count tokens using tiktoken library."""
        try:
            tokens = self.tiktoken_encoder.encode(text)
            count = len(tokens)
            
            # Calculate confidence based on model similarity
            # tiktoken is very accurate for OpenAI models, good for others
            confidence = 0.9 if "gpt" in self.model.lower() else 0.8
            
            return TokenCount(
                count=count,
                method="tiktoken",
                model=self.model,
                confidence=confidence
            )
        except Exception as e:
            logger.warning(f"Tiktoken counting failed: {e}, falling back to approximate")
            return self._count_approximate(text)
    
    def _count_approximate(self, text: str) -> TokenCount:
        """Count tokens using approximate character-based method."""
        # Determine text type for better approximation
        text_type = self._classify_text_type(text)
        ratio = self.TOKEN_RATIOS[text_type]
        
        # Count characters (excluding whitespace for better accuracy)
        char_count = len(text.strip())
        
        # Apply ratio to get approximate token count
        approximate_tokens = int(char_count / ratio)
        
        # Adjust for common patterns
        approximate_tokens = self._adjust_for_patterns(text, approximate_tokens)
        
        return TokenCount(
            count=approximate_tokens,
            method="approximate",
            model=self.model,
            confidence=0.7  # Lower confidence for approximate method
        )
    
    def _classify_text_type(self, text: str) -> str:
        """Classify text type for better token approximation."""
        # Check for code patterns
        code_patterns = [
            r'def\s+\w+\s*\(',
            r'import\s+',
            r'class\s+\w+',
            r'function\s+\w+',
            r'var\s+\w+',
            r'const\s+\w+',
            r'if\s*\(',
            r'for\s*\(',
            r'while\s*\(',
            r'print\s*\(',
            r'return\s+',
            r'#\s*',  # Comments
            r'//\s*',  # Comments
            r'/\*.*?\*/',  # Block comments
        ]
        
        code_score = sum(1 for pattern in code_patterns if re.search(pattern, text, re.IGNORECASE))
        
        if code_score >= 2:
            return "code"
        elif re.search(r'[a-zA-Z]', text) and re.search(r'[\u4e00-\u9fff]', text):
            return "mixed"  # Mixed language
        else:
            return "english"
    
    def _adjust_for_patterns(self, text: str, base_count: int) -> int:
        """Adjust token count based on common patterns."""
        adjustments = 0
        
        # Punctuation and special characters
        punctuation_count = len(re.findall(r'[^\w\s]', text))
        adjustments += int(punctuation_count * 0.1)  # Punctuation often creates tokens
        
        # Numbers
        number_count = len(re.findall(r'\d+', text))
        adjustments += int(number_count * 0.2)  # Numbers often create tokens
        
        # URLs
        url_count = len(re.findall(r'https?://\S+', text))
        adjustments += url_count * 2  # URLs create multiple tokens
        
        # Email addresses
        email_count = len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
        adjustments += email_count * 3  # Email addresses create multiple tokens
        
        return max(0, base_count + adjustments)
    
    def count_message_tokens(self, messages: List[Dict[str, Any]]) -> TokenCount:
        """
        Count tokens in a list of messages (for chat completions).
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            TokenCount object
        """
        total_tokens = 0
        
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            # Count content tokens
            content_count = self.count_tokens(content)
            total_tokens += content_count.count
            
            # Add role-specific overhead
            if role == 'user':
                total_tokens += self.SPECIAL_TOKENS['user_prefix']
            elif role == 'assistant':
                total_tokens += self.SPECIAL_TOKENS['assistant_prefix']
            elif role == 'system':
                total_tokens += self.SPECIAL_TOKENS['system_prompt']
        
        return TokenCount(
            count=total_tokens,
            method=content_count.method,
            model=self.model,
            confidence=content_count.confidence
        )
    
    def estimate_rag_tokens(
        self,
        query: str,
        context_documents: List[Dict[str, Any]],
        system_prompt: Optional[str] = None
    ) -> Dict[str, TokenCount]:
        """
        Estimate token counts for RAG operations.
        
        Args:
            query: User query
            context_documents: List of context documents
            system_prompt: Optional system prompt
            
        Returns:
            Dictionary with token counts for different components
        """
        counts = {}
        
        # Count query tokens
        counts['query'] = self.count_tokens(query)
        
        # Count system prompt tokens
        if system_prompt:
            counts['system_prompt'] = self.count_tokens(system_prompt)
        else:
            counts['system_prompt'] = TokenCount(
                count=self.SPECIAL_TOKENS['system_prompt'],
                method="estimated",
                model=self.model,
                confidence=0.5
            )
        
        # Count context tokens
        context_text = self._build_context_text(context_documents)
        counts['context'] = self.count_tokens(context_text)
        
        # Count total input tokens
        total_input = (
            counts['query'].count +
            counts['system_prompt'].count +
            counts['context'].count
        )
        counts['total_input'] = TokenCount(
            count=total_input,
            method="calculated",
            model=self.model,
            confidence=0.8
        )
        
        return counts
    
    def _build_context_text(self, documents: List[Dict[str, Any]]) -> str:
        """Build context text from documents for token counting."""
        if not documents:
            return ""
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            
            # Add document header
            title = metadata.get("title", f"Document {i}")
            source = metadata.get("source_id", "Unknown source")
            
            context_parts.append(f"--- Document {i}: {title} (Source: {source}) ---")
            context_parts.append(content)
            context_parts.append("")  # Empty line for separation
        
        return "\n".join(context_parts)
    
    def format_token_info(self, token_count: TokenCount) -> str:
        """Format token count information for display."""
        method_names = {
            "tiktoken": "Tiktoken (accurate)",
            "approximate": "Approximate",
            "calculated": "Calculated",
            "estimated": "Estimated",
            "empty": "Empty"
        }
        
        method_display = method_names.get(token_count.method, token_count.method)
        confidence_pct = int(token_count.confidence * 100)
        
        return f"{token_count.count:,} tokens ({method_display}, {confidence_pct}% confidence)"


# Global token counter instance
_token_counter: Optional[TokenCounter] = None


def get_token_counter(model: str = "gemini-2.5-flash-lite-preview") -> TokenCounter:
    """Get the global token counter instance."""
    global _token_counter
    if _token_counter is None or _token_counter.model != model:
        _token_counter = TokenCounter(model)
    return _token_counter


def count_tokens(text: str, model: str = "gemini-2.5-flash-lite-preview") -> TokenCount:
    """Convenience function to count tokens in text."""
    return get_token_counter(model).count_tokens(text)


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "gemini-2.5-flash-lite-preview"
) -> Dict[str, float]:
    """
    Estimate cost for token usage.
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model: Model name
        
    Returns:
        Dictionary with cost breakdown
    """
    # Gemini 2.5 Flash-Lite Preview pricing
    if "gemini" in model.lower():
        input_cost_per_1m = 0.10
        output_cost_per_1m = 0.40
    else:
        # Default pricing (can be updated for other models)
        input_cost_per_1m = 0.10
        output_cost_per_1m = 0.40
    
    input_cost = (input_tokens / 1_000_000) * input_cost_per_1m
    output_cost = (output_tokens / 1_000_000) * output_cost_per_1m
    total_cost = input_cost + output_cost
    
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens
    } 