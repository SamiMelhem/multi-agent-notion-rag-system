"""
Tests for the Notion API client.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from notion_client.errors import HTTPResponseError, RequestTimeoutError, APIResponseError

from notion_rag.notion_client import (
    NotionClient,
    NotionAPIResponse,
    NotionErrorType,
    RateLimiter
)
from notion_rag.config import Config


class TestRateLimiter:
    """Test cases for RateLimiter."""
    
    def test_rate_limiter_initialization(self):
        """Test rate limiter initialization."""
        limiter = RateLimiter(max_requests=5, time_window=2.0)
        assert limiter.max_requests == 5
        assert limiter.time_window == 2.0
        assert limiter.requests == []
    
    @pytest.mark.asyncio
    async def test_rate_limiter_allows_requests_within_limit(self):
        """Test that rate limiter allows requests within the limit."""
        limiter = RateLimiter(max_requests=3, time_window=1.0)
        
        # First 3 requests should be allowed immediately
        start_time = time.time()
        await limiter.acquire()
        await limiter.acquire()
        await limiter.acquire()
        elapsed = time.time() - start_time
        
        # Should be very fast since no rate limiting occurred
        assert elapsed < 0.1
    
    @pytest.mark.asyncio
    async def test_rate_limiter_delays_excess_requests(self):
        """Test that rate limiter delays requests exceeding the limit."""
        limiter = RateLimiter(max_requests=2, time_window=1.0)
        
        # First 2 requests should be immediate
        await limiter.acquire()
        await limiter.acquire()
        
        # Third request should be delayed
        start_time = time.time()
        await limiter.acquire()
        elapsed = time.time() - start_time
        
        # Should have waited close to 1 second
        assert elapsed >= 0.9
    
    def test_rate_limiter_reset(self):
        """Test rate limiter reset functionality."""
        limiter = RateLimiter(max_requests=3, time_window=1.0)
        limiter.requests = [time.time() - 0.5, time.time() - 0.3, time.time() - 0.1]
        
        limiter.reset()
        assert limiter.requests == []


class TestNotionClient:
    """Test cases for NotionClient."""
    
    @patch('notion_rag.notion_client.Client')
    def test_client_initialization_success(self, mock_client_class):
        """Test successful client initialization."""
        mock_config = Mock()
        mock_config.get_notion_api_key.return_value = "test_api_key_123"
        
        client = NotionClient(config=mock_config)
        
        assert client._authenticated is True
        mock_client_class.assert_called_once_with(auth="test_api_key_123")
    
    @patch('notion_rag.notion_client.Client')
    def test_client_initialization_failure(self, mock_client_class):
        """Test client initialization failure."""
        mock_config = Mock()
        mock_config.get_notion_api_key.side_effect = ValueError("API key not found")
        
        with pytest.raises(ValueError):
            NotionClient(config=mock_config)
    
    def test_handle_notion_error_authentication(self):
        """Test handling of authentication errors."""
        client = NotionClient.__new__(NotionClient)
        
        mock_response = Mock()
        mock_response.status_code = 401
        error = HTTPResponseError(mock_response, "Unauthorized")
        
        response = client._handle_notion_error(error)
        
        assert response.success is False
        assert response.error_type == NotionErrorType.AUTHENTICATION
        assert "Invalid or expired API key" in response.error_message
    
    def test_handle_notion_error_rate_limit(self):
        """Test handling of rate limit errors."""
        client = NotionClient.__new__(NotionClient)
        
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {"retry-after": "30"}
        error = HTTPResponseError(mock_response, "Rate limited")
        
        response = client._handle_notion_error(error)
        
        assert response.success is False
        assert response.error_type == NotionErrorType.RATE_LIMIT
        assert response.retry_after == 30
    
    def test_handle_notion_error_not_found(self):
        """Test handling of not found errors."""
        client = NotionClient.__new__(NotionClient)
        
        mock_response = Mock()
        mock_response.status_code = 404
        error = HTTPResponseError(mock_response, "Not found")
        
        response = client._handle_notion_error(error)
        
        assert response.success is False
        assert response.error_type == NotionErrorType.OBJECT_NOT_FOUND
    
    def test_handle_notion_error_bad_request(self):
        """Test handling of bad request errors."""
        client = NotionClient.__new__(NotionClient)
        
        mock_response = Mock()
        mock_response.status_code = 400
        error = HTTPResponseError(mock_response, "Bad request")
        
        response = client._handle_notion_error(error)
        
        assert response.success is False
        assert response.error_type == NotionErrorType.INVALID_REQUEST
    
    def test_handle_notion_error_timeout(self):
        """Test handling of timeout errors."""
        client = NotionClient.__new__(NotionClient)
        
        error = RequestTimeoutError("Request timed out")
        
        response = client._handle_notion_error(error)
        
        assert response.success is False
        assert response.error_type == NotionErrorType.TIMEOUT
    
    def test_handle_notion_error_api_response(self):
        """Test handling of API response errors."""
        client = NotionClient.__new__(NotionClient)
        
        mock_response = Mock()
        from notion_client.errors import APIErrorCode
        error = APIResponseError(response=mock_response, message="API error", code=APIErrorCode.ValidationError)
        
        response = client._handle_notion_error(error)
        
        assert response.success is False
        assert response.error_type == NotionErrorType.UNKNOWN  # APIResponseError is handled as unknown
    
    def test_handle_notion_error_unknown(self):
        """Test handling of unknown errors."""
        client = NotionClient.__new__(NotionClient)
        
        error = Exception("Unknown error")
        
        response = client._handle_notion_error(error)
        
        assert response.success is False
        assert response.error_type == NotionErrorType.UNKNOWN
    
    @pytest.mark.asyncio
    @patch('notion_rag.notion_client.Client')
    async def test_make_request_success(self, mock_client_class):
        """Test successful API request."""
        mock_client_instance = Mock()
        mock_client_instance.test_method.return_value = {"result": "success"}
        mock_client_class.return_value = mock_client_instance
        
        mock_config = Mock()
        mock_config.get_notion_api_key.return_value = "test_api_key_123"
        
        client = NotionClient(config=mock_config)
        response = await client._make_request("test_method", arg1="value1")
        
        assert response.success is True
        assert response.data == {"result": "success"}
        mock_client_instance.test_method.assert_called_once_with(arg1="value1")
    
    @pytest.mark.asyncio
    @patch('notion_rag.notion_client.Client')
    async def test_make_request_not_authenticated(self, mock_client_class):
        """Test request when not authenticated."""
        client = NotionClient.__new__(NotionClient)
        client._authenticated = False
        
        response = await client._make_request("test_method")
        
        assert response.success is False
        assert response.error_type == NotionErrorType.AUTHENTICATION
    
    @pytest.mark.asyncio
    @patch('notion_rag.notion_client.Client')
    async def test_get_database_success(self, mock_client_class):
        """Test successful database retrieval."""
        mock_client_instance = Mock()
        mock_client_instance.databases = Mock()
        mock_client_instance.databases.retrieve = Mock(return_value={"id": "test_db"})
        mock_client_class.return_value = mock_client_instance
        
        mock_config = Mock()
        mock_config.get_notion_api_key.return_value = "test_api_key_123"
        
        client = NotionClient(config=mock_config)
        
        # Mock the _make_request method to return proper structure
        with patch.object(client, '_make_request') as mock_make_request:
            mock_make_request.return_value = NotionAPIResponse(success=True, data={"id": "test_db"})
            response = await client.get_database("123e4567e89b12d3a456426614174000")
        
        assert response.success is True
        assert response.data == {"id": "test_db"}
    
    @pytest.mark.asyncio
    @patch('notion_rag.notion_client.Client')
    async def test_get_database_invalid_id(self, mock_client_class):
        """Test database retrieval with invalid ID."""
        mock_config = Mock()
        mock_config.get_notion_api_key.return_value = "test_api_key_123"
        
        client = NotionClient(config=mock_config)
        response = await client.get_database("invalid_id")
        
        assert response.success is False
        assert response.error_type == NotionErrorType.VALIDATION_ERROR
    
    @pytest.mark.asyncio
    @patch('notion_rag.notion_client.Client')
    async def test_query_database_success(self, mock_client_class):
        """Test successful database query."""
        mock_client_instance = Mock()
        mock_client_instance.databases.query.return_value = {"results": []}
        mock_client_class.return_value = mock_client_instance
        
        mock_config = Mock()
        mock_config.get_notion_api_key.return_value = "test_api_key_123"
        
        client = NotionClient(config=mock_config)
        
        # Mock the _make_request method to return proper structure
        with patch.object(client, '_make_request') as mock_make_request:
            mock_make_request.return_value = NotionAPIResponse(success=True, data={"results": []})
            response = await client.query_database("123e4567e89b12d3a456426614174000")
        
        assert response.success is True
        assert response.data == {"results": []}
    
    @pytest.mark.asyncio
    @patch('notion_rag.notion_client.Client')
    async def test_get_page_success(self, mock_client_class):
        """Test successful page retrieval."""
        mock_client_instance = Mock()
        mock_client_instance.pages.retrieve.return_value = {"id": "test_page"}
        mock_client_class.return_value = mock_client_instance
        
        mock_config = Mock()
        mock_config.get_notion_api_key.return_value = "test_api_key_123"
        
        client = NotionClient(config=mock_config)
        
        # Mock the _make_request method to return proper structure
        with patch.object(client, '_make_request') as mock_make_request:
            mock_make_request.return_value = NotionAPIResponse(success=True, data={"id": "test_page"})
            response = await client.get_page("123e4567e89b12d3a456426614174000")
        
        assert response.success is True
        assert response.data == {"id": "test_page"}
    
    @pytest.mark.asyncio
    @patch('notion_rag.notion_client.Client')
    async def test_get_page_invalid_id(self, mock_client_class):
        """Test page retrieval with invalid ID."""
        mock_config = Mock()
        mock_config.get_notion_api_key.return_value = "test_api_key_123"
        
        client = NotionClient(config=mock_config)
        response = await client.get_page("invalid_id")
        
        assert response.success is False
        assert response.error_type == NotionErrorType.VALIDATION_ERROR
    
    @pytest.mark.asyncio
    @patch('notion_rag.notion_client.Client')
    async def test_get_block_children_success(self, mock_client_class):
        """Test successful block children retrieval."""
        mock_client_instance = Mock()
        mock_client_instance.blocks.children.list.return_value = {"results": []}
        mock_client_class.return_value = mock_client_instance
        
        mock_config = Mock()
        mock_config.get_notion_api_key.return_value = "test_api_key_123"
        
        client = NotionClient(config=mock_config)
        
        # Mock the _make_request method to return proper structure
        with patch.object(client, '_make_request') as mock_make_request:
            mock_make_request.return_value = NotionAPIResponse(success=True, data={"results": []})
            response = await client.get_block_children("123e4567e89b12d3a456426614174000")
        
        assert response.success is True
        assert response.data == {"results": []}
    
    @pytest.mark.asyncio
    @patch('notion_rag.notion_client.Client')
    async def test_search_success(self, mock_client_class):
        """Test successful search."""
        mock_client_instance = Mock()
        mock_client_instance.search.return_value = {"results": []}
        mock_client_class.return_value = mock_client_instance
        
        mock_config = Mock()
        mock_config.get_notion_api_key.return_value = "test_api_key_123"
        
        client = NotionClient(config=mock_config)
        response = await client.search("test query")
        
        assert response.success is True
        assert response.data == {"results": []}
    
    @pytest.mark.asyncio
    @patch('notion_rag.notion_client.Client')
    async def test_search_invalid_query(self, mock_client_class):
        """Test search with invalid query."""
        mock_config = Mock()
        mock_config.get_notion_api_key.return_value = "test_api_key_123"
        
        client = NotionClient(config=mock_config)
        response = await client.search("")  # Empty query
        
        assert response.success is False
        assert response.error_type == NotionErrorType.VALIDATION_ERROR
    
    @pytest.mark.asyncio
    @patch('notion_rag.notion_client.Client')
    async def test_get_all_pages_from_database_success(self, mock_client_class):
        """Test successful retrieval of all pages from database."""
        mock_client_instance = Mock()
        # Mock paginated response
        mock_client_instance.databases.query.side_effect = [
            {"results": [{"id": "page1"}, {"id": "page2"}], "has_more": True, "next_cursor": "cursor1"},
            {"results": [{"id": "page3"}], "has_more": False, "next_cursor": None}
        ]
        mock_client_class.return_value = mock_client_instance
        
        mock_config = Mock()
        mock_config.get_notion_api_key.return_value = "test_api_key_123"
        
        client = NotionClient(config=mock_config)
        
        # Mock the query_database method to return proper structure
        with patch.object(client, 'query_database') as mock_query_database:
            mock_query_database.side_effect = [
                NotionAPIResponse(success=True, data={"results": [{"id": "page1"}, {"id": "page2"}], "has_more": True, "next_cursor": "cursor1"}),
                NotionAPIResponse(success=True, data={"results": [{"id": "page3"}], "has_more": False, "next_cursor": None})
            ]
            response = await client.get_all_pages_from_database("123e4567e89b12d3a456426614174000")
        
        assert response.success is True
        assert len(response.data["results"]) == 3
        assert response.data["total_count"] == 3
    
    @pytest.mark.asyncio
    @patch('notion_rag.notion_client.Client')
    async def test_test_connection_success(self, mock_client_class):
        """Test successful connection test."""
        mock_client_instance = Mock()
        mock_client_instance.users.me.return_value = {"id": "user123"}
        mock_client_class.return_value = mock_client_instance
        
        mock_config = Mock()
        mock_config.get_notion_api_key.return_value = "test_api_key_123"
        
        client = NotionClient(config=mock_config)
        
        # Mock the _make_request method to return proper structure
        with patch.object(client, '_make_request') as mock_make_request:
            mock_make_request.return_value = NotionAPIResponse(success=True, data={"id": "user123"})
            response = await client.test_connection()
        
        assert response.success is True
        assert response.data == {"id": "user123"}
    
    @patch('notion_rag.notion_client.Client')
    def test_is_authenticated(self, mock_client_class):
        """Test authentication status check."""
        mock_config = Mock()
        mock_config.get_notion_api_key.return_value = "test_api_key_123"
        
        client = NotionClient(config=mock_config)
        assert client.is_authenticated() is True
    
    @patch('notion_rag.notion_client.Client')
    def test_reset_rate_limiter(self, mock_client_class):
        """Test rate limiter reset."""
        mock_config = Mock()
        mock_config.get_notion_api_key.return_value = "test_api_key_123"
        
        client = NotionClient(config=mock_config)
        client.rate_limiter.requests = [time.time(), time.time()]
        
        client.reset_rate_limiter()
        assert client.rate_limiter.requests == []
    
    @pytest.mark.asyncio
    @patch('notion_rag.notion_client.Client')
    async def test_rate_limiting_integration(self, mock_client_class):
        """Test rate limiting integration with actual requests."""
        mock_client_instance = Mock()
        mock_client_instance.users.me.return_value = {"id": "user123"}
        mock_client_class.return_value = mock_client_instance
        
        mock_config = Mock()
        mock_config.get_notion_api_key.return_value = "test_api_key_123"
        
        client = NotionClient(config=mock_config)
        client.rate_limiter = RateLimiter(max_requests=2, time_window=1.0)
        
        # First two requests should be fast
        start_time = time.time()
        await client.test_connection()
        await client.test_connection()
        elapsed = time.time() - start_time
        assert elapsed < 0.1
        
        # Third request should be delayed
        start_time = time.time()
        await client.test_connection()
        elapsed = time.time() - start_time
        assert elapsed >= 0.9  # Should wait about 1 second


class TestNotionAPIResponse:
    """Test cases for NotionAPIResponse."""
    
    def test_successful_response(self):
        """Test successful response creation."""
        response = NotionAPIResponse(
            success=True,
            data={"result": "success"}
        )
        
        assert response.success is True
        assert response.data == {"result": "success"}
        assert response.error_type is None
        assert response.error_message is None
    
    def test_error_response(self):
        """Test error response creation."""
        response = NotionAPIResponse(
            success=False,
            error_type=NotionErrorType.AUTHENTICATION,
            error_message="Invalid API key"
        )
        
        assert response.success is False
        assert response.error_type == NotionErrorType.AUTHENTICATION
        assert response.error_message == "Invalid API key"
        assert response.data is None 