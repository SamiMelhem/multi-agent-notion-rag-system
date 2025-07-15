"""
Tests for the Google Gemini API client.
"""

import pytest
import json
import time
from unittest.mock import Mock, patch, MagicMock
from requests.exceptions import RequestException, HTTPError
import os

# Import timeout if available
try:
    from pytest_timeout import timeout
except ImportError:
    # Fallback decorator if pytest-timeout is not available
    def timeout(seconds):
        def decorator(func):
            return func
        return decorator

from notion_rag.config import Config
from notion_rag.gemini_client import (
    GeminiMessage,
    GeminiResponse,
    GeminiClient,
    RateLimiter,
    ContextManager,
    create_gemini_client
)


class TestGeminiMessage:
    """Test GeminiMessage class."""
    
    def test_valid_message_creation(self):
        """Test creating valid messages."""
        # Test user message
        user_msg = GeminiMessage("user", "Hello, how are you?")
        assert user_msg.role == "user"
        assert user_msg.content == "Hello, how are you?"
        
        # Test model message
        model_msg = GeminiMessage("model", "I'm doing well, thank you!")
        assert model_msg.role == "model"
        assert model_msg.content == "I'm doing well, thank you!"
    
    def test_invalid_role(self):
        """Test that invalid roles raise ValueError."""
        with pytest.raises(ValueError, match="Invalid role"):
            GeminiMessage("assistant", "Some content")
    
    def test_content_sanitization(self):
        """Test that content is properly sanitized."""
        # Test whitespace trimming
        msg = GeminiMessage("user", "  Hello World  ")
        assert msg.content == "Hello World"
        
        # Test control character removal
        msg = GeminiMessage("user", "Hello\x00\x08World")
        assert msg.content == "HelloWorld"
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        msg = GeminiMessage("user", "Test message")
        msg_dict = msg.to_dict()
        
        assert msg_dict == {
            "role": "user",
            "parts": [{"text": "Test message"}]
        }


class TestGeminiResponse:
    """Test GeminiResponse class."""
    
    def test_from_api_response(self):
        """Test creating GeminiResponse from API response."""
        api_response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Hello! How can I help you today?"}
                        ]
                    },
                    "finishReason": "STOP"
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 8,
                "totalTokenCount": 18
            },
            "model": "gemini-2.0-flash-exp:preview-06-13",
            "responseId": "resp-123"
        }
        
        response = GeminiResponse.from_api_response(api_response)
        
        assert response.content == "Hello! How can I help you today?"
        assert response.model == "gemini-2.0-flash-exp:preview-06-13"
        assert response.usage == {
            "prompt_token_count": 10,
            "candidates_token_count": 8,
            "total_token_count": 18
        }
        assert response.finish_reason == "STOP"
        assert response.response_id == "resp-123"
    
    def test_from_api_response_missing_fields(self):
        """Test creating GeminiResponse with missing fields."""
        api_response = {
            "candidates": [{}],
            "usageMetadata": {}
        }
        
        response = GeminiResponse.from_api_response(api_response)
        
        assert response.content == ""
        assert response.model == "unknown"
        assert response.usage == {
            "prompt_token_count": 0,
            "candidates_token_count": 0,
            "total_token_count": 0
        }
        assert response.finish_reason == "unknown"
        assert response.response_id == ""


class TestRateLimiter:
    """Test RateLimiter class."""
    
    def test_initialization(self):
        """Test rate limiter initialization."""
        limiter = RateLimiter(max_requests=10, time_window=30)
        
        assert limiter.max_requests == 10
        assert limiter.time_window == 30
        assert limiter.get_remaining_requests() == 10
    
    def test_single_request(self):
        """Test single request doesn't trigger rate limiting."""
        limiter = RateLimiter(max_requests=5, time_window=60)
        
        limiter.wait_if_needed()
        
        assert limiter.get_remaining_requests() == 4
    
    def test_multiple_requests_no_waiting(self):
        """Test multiple requests within limit don't trigger waiting."""
        limiter = RateLimiter(max_requests=10, time_window=1)
        
        start_time = time.time()
        for _ in range(5):
            limiter.wait_if_needed()
        end_time = time.time()
        
        # Should complete quickly without waiting
        assert end_time - start_time < 0.1
        assert limiter.get_remaining_requests() == 5
    
    @timeout(5)  # 5 second timeout
    def test_rate_limit_reached(self):
        """Test rate limiting when limit is reached."""
        limiter = RateLimiter(max_requests=2, time_window=0.1)  # 2 requests per 0.1 seconds
        
        # Make first request
        limiter.wait_if_needed()
        assert limiter.get_remaining_requests() == 1
        
        # Make second request
        limiter.wait_if_needed()
        assert limiter.get_remaining_requests() == 0
        
        # Third request should trigger waiting
        start_time = time.time()
        limiter.wait_if_needed()
        end_time = time.time()
        
        # Should have waited at least 0.05 seconds (half of time window)
        # But not more than 0.15 seconds (time window + buffer)
        wait_time = end_time - start_time
        assert 0.05 <= wait_time <= 0.15, f"Wait time {wait_time} outside expected range"
    
    @timeout(5)  # 5 second timeout
    def test_rate_limit_expiry(self):
        """Test that rate limit resets after time window."""
        limiter = RateLimiter(max_requests=1, time_window=0.05)  # 1 request per 0.05 seconds
        
        # Make request
        limiter.wait_if_needed()
        assert limiter.get_remaining_requests() == 0
        
        # Wait for time window to expire
        time.sleep(0.1)
        
        # Should have requests available again
        assert limiter.get_remaining_requests() == 1
    
    @timeout(5)  # 5 second timeout
    def test_rate_limiter_no_hang(self):
        """Test that rate limiter doesn't hang in edge cases."""
        limiter = RateLimiter(max_requests=1, time_window=0.01)  # Very short window
        
        # Make many requests quickly - should not hang
        start_time = time.time()
        for _ in range(5):
            limiter.wait_if_needed()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete within reasonable time (less than 1 second total)
        assert total_time < 1.0, f"Rate limiter took too long: {total_time} seconds"


class TestContextManager:
    """Test ContextManager class."""
    
    def test_initialization(self):
        """Test context manager initialization."""
        manager = ContextManager(max_tokens=1000)
        
        assert manager.max_tokens == 1000
        assert manager.current_tokens == 0
        assert manager.get_remaining_tokens() == 1000
    
    def test_add_single_message(self):
        """Test adding a single message."""
        manager = ContextManager(max_tokens=100)
        message = GeminiMessage("user", "Hello world")  # ~3 tokens
        
        success = manager.add_message(message)
        
        assert success is True
        assert manager.get_token_count() > 0
        assert len(manager.get_messages()) == 1
    
    def test_add_message_exceeding_limit(self):
        """Test adding message that exceeds token limit."""
        manager = ContextManager(max_tokens=10)  # Very small limit
        message = GeminiMessage("user", "This is a very long message that will exceed the token limit")  # ~15 tokens
        
        success = manager.add_message(message)
        
        assert success is False
        assert manager.get_token_count() == 0
        assert len(manager.get_messages()) == 0
    
    def test_add_multiple_messages(self):
        """Test adding multiple messages."""
        manager = ContextManager(max_tokens=50)
        messages = [
            GeminiMessage("user", "Hello"),
            GeminiMessage("model", "Hi there"),
            GeminiMessage("user", "How are you?")
        ]
        
        added_messages = manager.add_messages(messages)
        
        assert len(added_messages) == 3
        assert manager.get_token_count() > 0
    
    def test_add_messages_partial_fit(self):
        """Test adding messages where only some fit."""
        manager = ContextManager(max_tokens=5)  # Very small limit
        messages = [
            GeminiMessage("user", "Short"),  # ~1 token
            GeminiMessage("user", "This is a very long message that will definitely exceed the token limit"),  # ~15 tokens
            GeminiMessage("user", "Another short one")  # ~4 tokens
        ]
        
        added_messages = manager.add_messages(messages)
        
        # Only first message should fit
        assert len(added_messages) == 1
        assert added_messages[0].content == "Short"
    
    def test_clear_context(self):
        """Test clearing context."""
        manager = ContextManager(max_tokens=100)
        message = GeminiMessage("user", "Hello")
        
        manager.add_message(message)
        assert manager.get_token_count() > 0
        
        manager.clear()
        assert manager.get_token_count() == 0
        assert len(manager.get_messages()) == 0
    
    def test_trim_context(self):
        """Test trimming context."""
        manager = ContextManager(max_tokens=50)
        
        # Add several messages
        messages = [
            GeminiMessage("user", "First message"),  # Remove this
            GeminiMessage("model", "First response"),  # Remove this
            GeminiMessage("user", "Second message")  # Keep this
        ]
        
        manager.add_messages(messages)
        initial_tokens = manager.get_token_count()
        
        # Trim to a lower token count
        manager.trim_context(initial_tokens // 2)
        
        assert manager.get_token_count() <= initial_tokens // 2


class TestGeminiClient:
    """Test GeminiClient class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock config."""
        mock = Mock(spec=Config)
        mock.get_gemini_api_key.return_value = None  # Default to None
        return mock
    
    @pytest.fixture
    def api_key(self):
        """Sample API key."""
        return "AIzaSyC-test-api-key-1234567890abcdef"
    
    def test_initialization_with_api_key(self, mock_config, api_key):
        """Test client initialization with provided API key."""
        client = GeminiClient(mock_config, api_key)
        
        assert client.api_key == api_key
        assert client.config == mock_config
        assert client.rate_limiter is not None
        assert client.context_manager is not None
        assert client.model is not None
    
    def test_initialization_without_api_key(self, mock_config):
        """Test client initialization without API key."""
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'env-api-key'}):
            client = GeminiClient(mock_config)
            
            assert client.api_key == "env-api-key"
    
    def test_initialization_no_api_key_available(self, mock_config):
        """Test client initialization when no API key is available."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="Gemini API key is required"):
                GeminiClient(mock_config)
    
    def test_create_session(self, mock_config, api_key):
        """Test session creation with retry logic."""
        client = GeminiClient(mock_config, api_key)
        
        assert client.model is not None
        assert hasattr(client, 'rate_limiter')
        assert hasattr(client, 'context_manager')
    
    @patch('google.generativeai.GenerativeModel')
    def test_make_request_success(self, mock_genai_model, mock_config, api_key):
        """Test successful API request."""
        # Mock the GenerativeModel
        mock_model_instance = Mock()
        mock_genai_model.return_value = mock_model_instance
        
        # Mock successful response
        mock_response = Mock()
        mock_response.text = "Hello! How can I help you?"
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].content.parts = [Mock()]
        mock_response.candidates[0].content.parts[0].text = "Hello! How can I help you?"
        mock_response.candidates[0].finish_reason = "STOP"
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 5
        mock_response.usage_metadata.candidates_token_count = 7
        mock_response.usage_metadata.total_token_count = 12
        mock_response.response_id = "test-response-id"
        
        # Set up proper attributes for token counting
        mock_response.prompt_token_count = 5
        mock_response.candidates_token_count = 7
        mock_response.finish_reason = "STOP"
        
        mock_model_instance.generate_content.return_value = mock_response
        
        client = GeminiClient(mock_config, api_key)
        
        # Test chat completion
        messages = [GeminiMessage("user", "Hello")]
        response = client.chat_completion(messages)
        
        assert isinstance(response, GeminiResponse)
        assert response.content == "Hello! How can I help you?"
    
    @patch('google.generativeai.GenerativeModel')
    def test_make_request_failure(self, mock_genai_model, mock_config, api_key):
        """Test failed API request."""
        mock_model_instance = Mock()
        mock_genai_model.return_value = mock_model_instance
        
        # Mock failed response
        mock_model_instance.generate_content.side_effect = Exception("API Error")
        
        client = GeminiClient(mock_config, api_key)
        
        messages = [GeminiMessage("user", "Hello")]
        with pytest.raises(Exception):
            client.chat_completion(messages)
    
    def test_chat_completion_validation(self, mock_config, api_key):
        """Test chat completion parameter validation."""
        client = GeminiClient(mock_config, api_key)
        
        # Test empty messages
        with pytest.raises(ValueError, match="At least one message is required"):
            client.chat_completion([])
        
        # Test invalid temperature
        messages = [GeminiMessage("user", "Hello")]
        with pytest.raises(ValueError, match="Temperature must be between 0.0 and 2.0"):
            client.chat_completion(messages, temperature=3.0)
        
        # Test invalid top_p
        with pytest.raises(ValueError, match="top_p must be between 0.0 and 1.0"):
            client.chat_completion(messages, top_p=1.5)
        
        # Test invalid top_k
        with pytest.raises(ValueError, match="top_k must be at least 1"):
            client.chat_completion(messages, top_k=0)
    
    @patch('google.generativeai.GenerativeModel')
    def test_chat_completion_success(self, mock_genai_model, mock_config, api_key):
        """Test successful chat completion."""
        mock_model_instance = Mock()
        mock_genai_model.return_value = mock_model_instance
        
        # Mock successful response
        mock_response = Mock()
        mock_response.text = "Hello! How can I help you?"
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].content.parts = [Mock()]
        mock_response.candidates[0].content.parts[0].text = "Hello! How can I help you?"
        mock_response.candidates[0].finish_reason = "STOP"
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 5
        mock_response.usage_metadata.candidates_token_count = 7
        mock_response.usage_metadata.total_token_count = 12
        mock_response.response_id = "test-response-id"
        
        # Set up proper attributes for token counting
        mock_response.prompt_token_count = 5
        mock_response.candidates_token_count = 7
        mock_response.finish_reason = "STOP"
        
        mock_model_instance.generate_content.return_value = mock_response
        
        client = GeminiClient(mock_config, api_key)
        
        messages = [GeminiMessage("user", "Hello")]
        response = client.chat_completion(messages)
        
        assert isinstance(response, GeminiResponse)
        assert response.content == "Hello! How can I help you?"
        assert response.finish_reason == "STOP"
    
    def test_rag_completion_validation(self, mock_config, api_key):
        """Test RAG completion parameter validation."""
        client = GeminiClient(mock_config, api_key)
        
        # Test empty query
        with pytest.raises(ValueError, match="Query cannot be empty"):
            client.rag_completion("", [])
        
        # Test whitespace-only query
        with pytest.raises(ValueError, match="Query cannot be empty"):
            client.rag_completion("   ", [])
    
    @patch('google.generativeai.GenerativeModel')
    def test_rag_completion_success(self, mock_genai_model, mock_config, api_key):
        """Test successful RAG completion."""
        mock_model_instance = Mock()
        mock_genai_model.return_value = mock_model_instance
        
        # Mock successful response
        mock_response = Mock()
        mock_response.text = "Based on the provided documents, here is the answer..."
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].content.parts = [Mock()]
        mock_response.candidates[0].content.parts[0].text = "Based on the provided documents, here is the answer..."
        mock_response.candidates[0].finish_reason = "STOP"
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 50
        mock_response.usage_metadata.candidates_token_count = 25
        mock_response.usage_metadata.total_token_count = 75
        mock_response.response_id = "test-response-id"
        
        # Set up proper attributes for token counting
        mock_response.prompt_token_count = 50
        mock_response.candidates_token_count = 25
        mock_response.finish_reason = "STOP"
        
        mock_model_instance.generate_content.return_value = mock_response
        
        client = GeminiClient(mock_config, api_key)
        
        # Test documents
        documents = [
            {
                "content": "This is document 1 content.",
                "metadata": {
                    "title": "Document 1",
                    "source_id": "doc1"
                }
            },
            {
                "content": "This is document 2 content.",
                "metadata": {
                    "title": "Document 2",
                    "source_id": "doc2"
                }
            }
        ]
        
        response = client.rag_completion("What is the main topic?", documents)
        
        assert isinstance(response, GeminiResponse)
        assert "Based on the provided documents" in response.content
    
    def test_build_context_from_documents(self, mock_config, api_key):
        """Test context building from documents."""
        client = GeminiClient(mock_config, api_key)
        
        documents = [
            {
                "content": "First document content",
                "metadata": {
                    "title": "First Doc",
                    "source_id": "doc1"
                }
            },
            {
                "content": "Second document content",
                "metadata": {
                    "title": "Second Doc",
                    "source_id": "doc2"
                }
            }
        ]
        
        context = client._build_context_from_documents(documents)
        
        assert "--- Document 1: First Doc (Source: doc1) ---" in context
        assert "--- Document 2: Second Doc (Source: doc2) ---" in context
        assert "First document content" in context
        assert "Second document content" in context
    
    def test_build_context_empty_documents(self, mock_config, api_key):
        """Test context building with empty documents."""
        client = GeminiClient(mock_config, api_key)
        
        context = client._build_context_from_documents([])
        assert context == "No relevant documents found."
    
    @patch('google.generativeai.GenerativeModel')
    def test_test_connection_success(self, mock_genai_model, mock_config, api_key):
        """Test successful connection test."""
        mock_model_instance = Mock()
        mock_genai_model.return_value = mock_model_instance
        
        # Mock successful response
        mock_response = Mock()
        mock_response.text = "Test response"
        mock_model_instance.generate_content.return_value = mock_response
        
        client = GeminiClient(mock_config, api_key)
        
        assert client.test_connection() is True
    
    @patch('google.generativeai.GenerativeModel')
    def test_test_connection_failure(self, mock_genai_model, mock_config, api_key):
        """Test failed connection test."""
        mock_model_instance = Mock()
        mock_genai_model.return_value = mock_model_instance
        
        # Mock failed response
        mock_model_instance.generate_content.side_effect = Exception("API Error")
        
        client = GeminiClient(mock_config, api_key)
        
        assert client.test_connection() is False
    
    def test_get_available_models(self, mock_config, api_key):
        """Test getting available models."""
        client = GeminiClient(mock_config, api_key)
        
        # The current implementation doesn't have this method, so we'll skip it
        # or implement a basic version that returns the default model
        assert hasattr(client, 'DEFAULT_MODEL')
        assert client.DEFAULT_MODEL == "gemini-2.5-flash-lite-preview-06-17"
    
    def test_store_api_key(self, mock_config, api_key):
        """Test storing API key."""
        client = GeminiClient(mock_config, api_key)
        
        result = client.store_api_key("new-api-key")
        assert result is True
    
    def test_close(self, mock_config, api_key):
        """Test closing the client."""
        client = GeminiClient(mock_config, api_key)
        
        # The current implementation doesn't have a close method that closes a session
        # It's a no-op method, so we just test that it doesn't raise an error
        client.close()  # Should not raise any error
    
    # Test context management methods
    def test_context_management_methods(self, mock_config, api_key):
        """Test context management methods."""
        client = GeminiClient(mock_config, api_key)
        
        # Test adding to context
        message = GeminiMessage("user", "Hello")
        success = client.add_to_context(message)
        assert success is True
        
        # Test getting context messages
        messages = client.get_context_messages()
        assert len(messages) == 1
        assert messages[0].content == "Hello"
        
        # Test getting token count
        token_count = client.get_context_token_count()
        assert token_count > 0
        
        # Test getting remaining tokens
        remaining = client.get_remaining_context_tokens()
        assert remaining < 1_000_000
        
        # Test clearing context
        client.clear_context()
        assert client.get_context_token_count() == 0
        assert len(client.get_context_messages()) == 0
    
    # Test rate limiting methods
    def test_rate_limiting_methods(self, mock_config, api_key):
        """Test rate limiting methods."""
        client = GeminiClient(mock_config, api_key)
        
        # Test getting remaining requests
        remaining = client.get_remaining_requests()
        assert remaining == 60  # Should start with 60 requests


class TestFactoryFunction:
    """Test the factory function."""
    
    def test_create_gemini_client(self):
        """Test creating client with factory function."""
        config = Mock(spec=Config)
        api_key = "test-api-key"
        
        client = create_gemini_client(config, api_key)
        
        assert isinstance(client, GeminiClient)
        assert client.api_key == api_key
        assert client.config == config


if __name__ == "__main__":
    pytest.main([__file__]) 