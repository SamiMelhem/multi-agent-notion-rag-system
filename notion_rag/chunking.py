"""
Text chunking utility for the Notion RAG system.
Splits text into 512-token chunks with 50-token overlap.
"""
from typing import List, Callable

try:
    import tiktoken
    def default_tokenizer(text: str) -> List[int]:
        enc = tiktoken.get_encoding("cl100k_base")
        return enc.encode(text)
    def detokenize(tokens: List[int]) -> str:
        enc = tiktoken.get_encoding("cl100k_base")
        return enc.decode(tokens)
    def count_tokens(text: str) -> int:
        return len(default_tokenizer(text))
except ImportError:
    def default_tokenizer(text: str) -> List[str]:
        # Fallback: whitespace tokenization
        return text.split()
    def detokenize(tokens: List[str]) -> str:
        return ' '.join(tokens)
    def count_tokens(text: str) -> int:
        return len(default_tokenizer(text))

def chunk_text(
    text: str,
    tokenizer: Callable[[str], List] = default_tokenizer,
    chunk_size: int = 512,
    overlap: int = 50
) -> List[str]:
    """
    Split text into overlapping chunks of tokens.
    Args:
        text: The input text to chunk.
        tokenizer: Function to tokenize text into tokens.
        chunk_size: Number of tokens per chunk.
        overlap: Number of tokens to overlap between chunks.
    Returns:
        List of text chunks (as strings).
    """
    tokens = tokenizer(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text_str = detokenize(chunk_tokens)
        chunks.append(chunk_text_str)
        if end == len(tokens):
            break
        start += chunk_size - overlap
    return chunks 