"""
Google Gemini API client for the Notion RAG system.
Provides authentication, chat completion, and RAG-specific functionality.
Uses Google GenAI SDK with Vertex AI for Gemini 2.5 Flash-Lite Preview.
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from collections import deque
from threading import Lock

try:
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig
except ImportError:
    print("❌ google-generativeai package not found. Install with: pip install google-generativeai")
    raise

from .config import Config
from .security import InputValidator, SecureKeyManager
from .cost_tracker import track_gemini_cost
from .token_counter import get_token_counter, estimate_cost

logger = logging.getLogger(__name__)


@dataclass
class GeminiMessage:
    """Represents a message in the Gemini conversation."""
    
    role: str  # "user", "model"
    content: str
    
    def __post_init__(self):
        """Validate and sanitize the message."""
        if self.role not in ["user", "model"]:
            raise ValueError(f"Invalid role: {self.role}")
        
        self.content = InputValidator.sanitize_text(self.content, max_length=32000)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for API."""
        return {
            "role": self.role,
            "parts": [{"text": self.content}]
        }


@dataclass
class GeminiResponse:
    """Represents a response from Gemini API."""
    
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str
    response_id: str
    created_at: int
    
    @classmethod
    def from_api_response(cls, response_data: Dict[str, Any]) -> 'GeminiResponse':
        """Create GeminiResponse from API response data."""
        candidates = response_data.get("candidates", [{}])
        candidate = candidates[0] if candidates else {}
        
        # Extract content from parts
        content = ""
        if "content" in candidate:
            parts = candidate["content"].get("parts", [])
            content = "".join([part.get("text", "") for part in parts])
        
        # Extract usage information
        usage = response_data.get("usageMetadata", {})
        
        return cls(
            content=content,
            model=response_data.get("model", "unknown"),
            usage={
                "prompt_token_count": usage.get("promptTokenCount", 0),
                "candidates_token_count": usage.get("candidatesTokenCount", 0),
                "total_token_count": usage.get("totalTokenCount", 0)
            },
            finish_reason=candidate.get("finishReason", "unknown"),
            response_id=response_data.get("responseId", ""),
            created_at=int(time.time())
        )


class RateLimiter:
    """Rate limiter for API requests (60 requests per minute)."""
    
    def __init__(self, max_requests: int = 60, time_window: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests allowed in time window
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
        self.lock = Lock()
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded."""
        with self.lock:
            now = time.time()
            
            # Remove old requests outside the time window
            while self.requests and now - self.requests[0] > self.time_window:
                self.requests.popleft()
            
            # If at rate limit, wait until oldest request expires
            if len(self.requests) >= self.max_requests:
                wait_time = self.time_window - (now - self.requests[0])
                if wait_time > 0:
                    logger.info(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                    # Clean up expired requests after waiting
                    now = time.time()
                    while self.requests and now - self.requests[0] > self.time_window:
                        self.requests.popleft()
            
            # Add current request
            self.requests.append(now)
    
    def get_remaining_requests(self) -> int:
        """Get number of remaining requests in current time window."""
        with self.lock:
            now = time.time()
            # Remove old requests
            while self.requests and now - self.requests[0] > self.time_window:
                self.requests.popleft()
            return self.max_requests - len(self.requests)


class ContextManager:
    """Manages context for Gemini 2.5 Flash-Lite Preview (1M token context window)."""
    
    def __init__(self, max_tokens: int = 1_000_000):
        """
        Initialize context manager.
        
        Args:
            max_tokens: Maximum tokens allowed in context
        """
        self.max_tokens = max_tokens
        self.current_tokens = 0
        self.messages: List[GeminiMessage] = []
        self.lock = Lock()
    
    def add_message(self, message: GeminiMessage) -> bool:
        """
        Add a message to context if it fits.
        
        Args:
            message: Message to add
            
        Returns:
            bool: True if message was added, False if it would exceed limit
        """
        with self.lock:
            # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
            estimated_tokens = len(message.content) // 4
            
            if self.current_tokens + estimated_tokens > self.max_tokens:
                logger.warning(f"Message would exceed context limit. "
                             f"Current: {self.current_tokens}, "
                             f"Adding: {estimated_tokens}, "
                             f"Limit: {self.max_tokens}")
                return False
            
            self.messages.append(message)
            self.current_tokens += estimated_tokens
            logger.debug(f"Added message. Current tokens: {self.current_tokens}")
            return True
    
    def add_messages(self, messages: List[GeminiMessage]) -> List[GeminiMessage]:
        """
        Add multiple messages, returning only those that fit.
        
        Args:
            messages: Messages to add
            
        Returns:
            List of messages that were successfully added
        """
        added_messages = []
        
        for message in messages:
            if self.add_message(message):
                added_messages.append(message)
            else:
                break
        
        return added_messages
    
    def get_messages(self) -> List[GeminiMessage]:
        """Get all messages in context."""
        with self.lock:
            return self.messages.copy()
    
    def clear(self):
        """Clear all messages from context."""
        with self.lock:
            self.messages.clear()
            self.current_tokens = 0
    
    def get_token_count(self) -> int:
        """Get current token count."""
        with self.lock:
            return self.current_tokens
    
    def get_remaining_tokens(self) -> int:
        """Get remaining tokens available."""
        with self.lock:
            return self.max_tokens - self.current_tokens
    
    def trim_context(self, target_tokens: int):
        """
        Trim context to target token count by removing oldest messages.
        
        Args:
            target_tokens: Target token count to trim to
        """
        with self.lock:
            while self.current_tokens > target_tokens and self.messages:
                # Remove oldest message
                removed_message = self.messages.pop(0)
                estimated_tokens = len(removed_message.content) // 4
                self.current_tokens -= estimated_tokens
                logger.debug(f"Removed message. Current tokens: {self.current_tokens}")


class GeminiClient:
    """Client for Google Gemini API with RAG-specific functionality."""
    
    # Model Configuration
    DEFAULT_MODEL = "gemini-2.5-flash-lite-preview-06-17"
    DEFAULT_TIMEOUT = 60
    MAX_RETRIES = 3
    
    def __init__(self, config: Config, api_key: Optional[str] = None, prompt_library: Optional[Any] = None):
        """
        Initialize the Gemini client.
        
        Args:
            config: Configuration object
            api_key: Optional API key (will be retrieved from environment if not provided)
            prompt_library: Optional prompt library for enhanced prompting
        """
        self.config = config
        self.api_key = api_key or self._get_api_key()
        self.project_id = config.get_google_cloud_project()
        
        if not self.api_key:
            raise ValueError("Gemini API key is required")
        
        # Initialize Gemini
        self._initialize_gemini()
        
        # Initialize rate limiter and context manager
        self.rate_limiter = RateLimiter(max_requests=60, time_window=60)
        self.context_manager = ContextManager(max_tokens=1_000_000)
        
        # Initialize prompt library
        self.prompt_library = prompt_library
        
        logger.info("Initialized Gemini client with rate limiting, context management, and prompt library")
    
    def _get_api_key(self) -> Optional[str]:
        """Retrieve API key from config or environment."""
        # Try config first
        if hasattr(self.config, 'get_gemini_api_key'):
            key = self.config.get_gemini_api_key()
            if key:
                return key
        
        # Fallback to environment variable
        return os.getenv("GEMINI_API_KEY")
    
    def _initialize_gemini(self):
        """Initialize the Gemini client with the Google GenAI SDK."""
        try:
            # Configure the API key
            genai.configure(api_key=self.api_key)
            
            # Create the model
            self.model = genai.GenerativeModel(self.DEFAULT_MODEL)
            
            logger.info(f"Initialized Gemini model: {self.DEFAULT_MODEL}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            raise
    
    def _create_generation_config(self, temperature: float, max_tokens: Optional[int], top_p: float, top_k: int) -> GenerationConfig:
        """Create a generation config for the model."""
        config_params = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }
        
        if max_tokens:
            config_params["max_output_tokens"] = max_tokens
            
        return GenerationConfig(**config_params)
    
    def chat_completion(
        self,
        messages: List[GeminiMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        top_k: int = 40,
        stream: bool = False,
        use_context_manager: bool = True,
        tools: Optional[List[Dict[str, Any]]] = None,
        structured_output: Optional[Dict[str, Any]] = None
    ) -> GeminiResponse:
        """
        Generate a chat completion using Gemini 2.5 Flash-Lite Preview.
        
        Args:
            messages: List of conversation messages
            model: Model to use (defaults to DEFAULT_MODEL)
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            stream: Whether to stream the response
            use_context_manager: Whether to use context manager
            tools: Optional list of function/tool definitions for function calling
            structured_output: Optional structured output schema
            
        Returns:
            GeminiResponse object
            
        Raises:
            ValueError: For invalid parameters
            requests.RequestException: For API errors
        """
        # Validate parameters
        if not messages:
            raise ValueError("At least one message is required")
        
        if not (0.0 <= temperature <= 2.0):
            raise ValueError("Temperature must be between 0.0 and 2.0")
        
        if not (0.0 <= top_p <= 1.0):
            raise ValueError("top_p must be between 0.0 and 1.0")
        
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        
        # Use context manager if requested
        if use_context_manager:
            # Add messages to context manager
            added_messages = self.context_manager.add_messages(messages)
            if not added_messages:
                raise ValueError("Messages would exceed context limit")
            messages = added_messages
        
        # Apply rate limiting
        self.rate_limiter.wait_if_needed()
        
        # Estimate tokens and cost before making the request
        token_counter = get_token_counter(self.DEFAULT_MODEL)
        
        # Convert messages to format for token counting
        message_list = []
        for msg in messages:
            message_list.append({
                'role': msg.role,
                'content': msg.content
            })
        
        # Estimate input tokens
        input_token_count = token_counter.count_message_tokens(message_list)
        
        # Estimate output tokens (rough estimate based on max_tokens or default)
        estimated_output_tokens = max_tokens or 1000
        
        # Calculate estimated cost
        cost_estimate = estimate_cost(
            input_tokens=input_token_count.count,
            output_tokens=estimated_output_tokens,
            model=self.DEFAULT_MODEL
        )
        
        # Display token and cost information
        logger.info(f"Token estimation: Input {token_counter.format_token_info(input_token_count)}, "
                   f"Estimated output: {estimated_output_tokens:,} tokens")
        logger.info(f"Cost estimate: ${cost_estimate['total_cost']:.6f} "
                   f"(Input: ${cost_estimate['input_cost']:.6f}, Output: ${cost_estimate['output_cost']:.6f})")
        
        # Create generation config
        generation_config = self._create_generation_config(temperature, max_tokens, top_p, top_k)
        
        # Prepare content for the model
        if len(messages) == 1:
            # Single message - use generate_content
            content = messages[0].content
            logger.info(f"Making single message completion request")
            
            try:
                response = self.model.generate_content(
                    content,
                    generation_config=generation_config
                )
                
                # Get actual token usage from response if available, otherwise use estimates
                actual_input_tokens = getattr(response, 'prompt_token_count', None)
                actual_output_tokens = getattr(response, 'candidates_token_count', None)
                
                # If actual tokens not available, use our estimates
                if actual_input_tokens is None:
                    actual_input_tokens = input_token_count.count
                if actual_output_tokens is None:
                    # Estimate output tokens based on response length
                    actual_output_tokens = len(response.text) // 4  # Rough estimate
                
                # Create response object
                gemini_response = GeminiResponse(
                    content=response.text,
                    model=self.DEFAULT_MODEL,
                    usage={
                        "prompt_token_count": actual_input_tokens,
                        "candidates_token_count": actual_output_tokens,
                        "total_token_count": actual_input_tokens + actual_output_tokens
                    },
                    finish_reason=getattr(response, 'finish_reason', 'stop'),
                    response_id=getattr(response, 'response_id', ''),
                    created_at=int(time.time())
                )
                
                # Track cost with actual token counts
                track_gemini_cost(
                    operation="single_completion",
                    input_tokens=actual_input_tokens,
                    output_tokens=actual_output_tokens,
                    metadata={"model": self.DEFAULT_MODEL, "temperature": temperature}
                )
                
                return gemini_response
                
            except Exception as e:
                logger.error(f"Failed to generate content: {e}")
                raise
        else:
            # Multiple messages - use chat
            logger.info(f"Making chat completion request with {len(messages)} messages")
            
            try:
                chat = self.model.start_chat(history=[])
                
                # Add all messages except the last one to history
                for msg in messages[:-1]:
                    if msg.role == "user":
                        chat.send_message(msg.content)
                    elif msg.role == "model":
                        # For model messages, we'd need to handle this differently
                        # For now, we'll skip model messages in history
                        pass
                
                # Send the last message and get response
                response = chat.send_message(messages[-1].content, generation_config=generation_config)
                
                # Get actual token usage from response if available, otherwise use estimates
                actual_input_tokens = getattr(response, 'prompt_token_count', None)
                actual_output_tokens = getattr(response, 'candidates_token_count', None)
                
                # If actual tokens not available, use our estimates
                if actual_input_tokens is None:
                    actual_input_tokens = input_token_count.count
                if actual_output_tokens is None:
                    # Estimate output tokens based on response length
                    actual_output_tokens = len(response.text) // 4  # Rough estimate
                
                # Create response object
                gemini_response = GeminiResponse(
                    content=response.text,
            model=self.DEFAULT_MODEL,
                    usage={
                        "prompt_token_count": actual_input_tokens,
                        "candidates_token_count": actual_output_tokens,
                        "total_token_count": actual_input_tokens + actual_output_tokens
                    },
                    finish_reason=getattr(response, 'finish_reason', 'stop'),
                    response_id=getattr(response, 'response_id', ''),
            created_at=int(time.time())
                )
                
                # Track cost with actual token counts
                track_gemini_cost(
                    operation="chat_completion",
                    input_tokens=actual_input_tokens,
                    output_tokens=actual_output_tokens,
                    metadata={"model": self.DEFAULT_MODEL, "temperature": temperature}
                )
                
                return gemini_response
                
            except Exception as e:
                logger.error(f"Failed to generate chat content: {e}")
                raise
    

    
    def function_call(
        self,
        messages: List[GeminiMessage],
        tools: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        use_context_manager: bool = True
    ) -> GeminiResponse:
        """
        Make a function call using Gemini 2.5 Flash-Lite Preview's function calling capability.
        
        Args:
            messages: List of conversation messages
            tools: List of function definitions
            model: Model to use (defaults to DEFAULT_MODEL)
            temperature: Sampling temperature (lower for function calling)
            max_tokens: Maximum tokens to generate
            use_context_manager: Whether to use context manager
            
        Returns:
            GeminiResponse object with function call information
        """
        return self.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            use_context_manager=use_context_manager
        )
    
    def structured_completion(
        self,
        messages: List[GeminiMessage],
        schema: Dict[str, Any],
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        use_context_manager: bool = True
    ) -> GeminiResponse:
        """
        Generate structured output using Gemini 2.5 Flash-Lite Preview's structured output capability.
        
        Args:
            messages: List of conversation messages
            schema: JSON schema for structured output
            model: Model to use (defaults to DEFAULT_MODEL)
            temperature: Sampling temperature (lower for structured output)
            max_tokens: Maximum tokens to generate
            use_context_manager: Whether to use context manager
            
        Returns:
            GeminiResponse object with structured output
        """
        return self.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            structured_output=schema,
            use_context_manager=use_context_manager
        )
    
    def rag_completion(
        self,
        query: str,
        context_documents: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        prompt_template: str = "rag_qa",
        temperature: float = 0.7,
        max_tokens: Optional[int] = 1000,
        use_context_manager: bool = True
    ) -> GeminiResponse:
        """
        Generate a RAG completion using retrieved documents as context.
        
        Args:
            query: User query
            context_documents: List of retrieved documents with content and metadata
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            use_context_manager: Whether to use context manager
            
        Returns:
            GeminiResponse object
        """
        # Validate inputs
        if not query.strip():
            raise ValueError("Query cannot be empty")
        
        if not context_documents:
            logger.warning("No context documents provided for RAG completion")
        
        # Build context from documents
        context_text = self._build_context_from_documents(context_documents)
        
        # Create system prompt using prompt library if available
        if not system_prompt:
            if self.prompt_library:
                try:
                    template = self.prompt_library.get_template(prompt_template)
                    if template:
                        user_content = template.format(
                            context=context_text,
                            question=query
                        )
                        messages = [GeminiMessage("user", user_content)]
                        logger.info(f"Using prompt template: {prompt_template}")
                    else:
                        logger.warning(f"Template '{prompt_template}' not found, using default")
                        system_prompt = self._get_default_rag_system_prompt()
                        user_content = f"{system_prompt}\n\nContext:\n{context_text}\n\nQuery: {query}"
                        messages = [GeminiMessage("user", user_content)]
                except Exception as e:
                    logger.warning(f"Failed to use prompt template '{prompt_template}': {e}, using default")
                    system_prompt = self._get_default_rag_system_prompt()
                    user_content = f"{system_prompt}\n\nContext:\n{context_text}\n\nQuery: {query}"
                    messages = [GeminiMessage("user", user_content)]
            else:
                system_prompt = self._get_default_rag_system_prompt()
                user_content = f"{system_prompt}\n\nContext:\n{context_text}\n\nQuery: {query}"
                messages = [GeminiMessage("user", user_content)]
        else:
            # Use provided system prompt
            user_content = f"{system_prompt}\n\nContext:\n{context_text}\n\nQuery: {query}"
            messages = [GeminiMessage("user", user_content)]
        
        # Estimate tokens for RAG operation
        token_counter = get_token_counter(self.DEFAULT_MODEL)
        rag_token_breakdown = token_counter.estimate_rag_tokens(
            query=query,
            context_documents=context_documents,
            system_prompt=system_prompt
        )
        
        # Display detailed token breakdown
        logger.info("RAG Token Breakdown:")
        logger.info(f"  Query: {token_counter.format_token_info(rag_token_breakdown['query'])}")
        logger.info(f"  System Prompt: {token_counter.format_token_info(rag_token_breakdown['system_prompt'])}")
        logger.info(f"  Context: {token_counter.format_token_info(rag_token_breakdown['context'])}")
        logger.info(f"  Total Input: {token_counter.format_token_info(rag_token_breakdown['total_input'])}")
        
        # Estimate output tokens
        estimated_output_tokens = max_tokens or 1000
        
        # Calculate estimated cost
        cost_estimate = estimate_cost(
            input_tokens=rag_token_breakdown['total_input'].count,
            output_tokens=estimated_output_tokens,
            model=self.DEFAULT_MODEL
        )
        
        logger.info(f"Estimated output: {estimated_output_tokens:,} tokens")
        logger.info(f"Cost estimate: ${cost_estimate['total_cost']:.6f} "
                   f"(Input: ${cost_estimate['input_cost']:.6f}, Output: ${cost_estimate['output_cost']:.6f})")
        
        logger.info(f"Making RAG completion request with {len(context_documents)} context documents")
        
        # Make the completion request
        response = self.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            use_context_manager=use_context_manager
        )
        
        # Update cost tracking with RAG-specific information
        # The chat_completion method already tracks the cost, but we can add RAG-specific metadata
        if hasattr(response, 'usage') and response.usage:
            # Log RAG-specific cost information
            logger.info(f"RAG completion completed. "
                       f"Input: {response.usage.get('prompt_token_count', 0):,} tokens, "
                       f"Output: {response.usage.get('candidates_token_count', 0):,} tokens")
        
        return response
    
    def _build_context_from_documents(self, documents: List[Dict[str, Any]]) -> str:
        """Build context text from retrieved documents."""
        if not documents:
            return "No relevant documents found."
        
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
    
    def _get_default_rag_system_prompt(self) -> str:
        """Get the default system prompt for RAG completions."""
        return """You are a helpful AI assistant that answers questions based on the provided context documents. 

Guidelines:
1. Use only the information provided in the context documents to answer questions
2. If the context doesn't contain enough information to answer the question, say so clearly
3. Cite specific documents or sources when possible
4. Be concise but thorough in your responses
5. If you're unsure about something, acknowledge the uncertainty
6. Maintain a helpful and professional tone
7. You can create plans and summaries based on the context

Format your responses clearly and structure them logically."""
    
    def test_connection(self) -> bool:
        """
        Test the connection to Gemini API.
        
        Returns:
            bool: True if connection is successful
        """
        try:
            # Make a simple test request
            response = self.model.generate_content("Hello")
            return response.text is not None
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def store_api_key(self, api_key: str) -> bool:
        """
        Store API key in environment (for compatibility).
        
        Args:
            api_key: Gemini API key
            
        Returns:
            bool: True if successful
        """
        try:
            os.environ["GEMINI_API_KEY"] = api_key
            return True
        except Exception as e:
            logger.error(f"Failed to store API key: {e}")
            return False
    
    # Context management methods
    def add_to_context(self, message: GeminiMessage) -> bool:
        """Add a message to the conversation context."""
        return self.context_manager.add_message(message)
    
    def get_context_messages(self) -> List[GeminiMessage]:
        """Get all messages in the conversation context."""
        return self.context_manager.get_messages()
    
    def clear_context(self):
        """Clear the conversation context."""
        self.context_manager.clear()
    
    def get_context_token_count(self) -> int:
        """Get current token count in context."""
        return self.context_manager.get_token_count()
    
    def get_remaining_context_tokens(self) -> int:
        """Get remaining tokens available in context."""
        return self.context_manager.get_remaining_tokens()
    
    def trim_context(self, target_tokens: int):
        """Trim context to target token count."""
        self.context_manager.trim_context(target_tokens)
    
    # Rate limiting methods
    def get_remaining_requests(self) -> int:
        """Get number of remaining requests in current time window."""
        return self.rate_limiter.get_remaining_requests()
    
    def close(self):
        """Close the client and clean up resources."""
        logger.info("Gemini client closed")


def create_gemini_client(config: Config, api_key: Optional[str] = None, prompt_library: Optional[Any] = None) -> GeminiClient:
    """
    Factory function to create a Gemini client.
    
    Args:
        config: Configuration object
        api_key: Optional API key
        prompt_library: Optional prompt library for enhanced prompting
        
    Returns:
        Configured GeminiClient
    """
    return GeminiClient(config, api_key, prompt_library) 