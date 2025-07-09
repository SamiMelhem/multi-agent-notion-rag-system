"""
Notion API client with authentication, error handling, and rate limiting.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

from notion_client import Client
from notion_client.errors import (
    RequestTimeoutError,
    HTTPResponseError,
    APIResponseError
)

from .config import Config
from .security import InputValidator, SecureQueryInput

logger = logging.getLogger(__name__)


class NotionErrorType(Enum):
    """Types of Notion API errors."""
    AUTHENTICATION = "authentication_error"
    RATE_LIMIT = "rate_limit_exceeded"
    TIMEOUT = "request_timeout"
    INVALID_REQUEST = "invalid_request"
    OBJECT_NOT_FOUND = "object_not_found"
    VALIDATION_ERROR = "validation_error"
    UNKNOWN = "unknown_error"


@dataclass
class NotionAPIResponse:
    """Standardized response from Notion API operations."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error_type: Optional[NotionErrorType] = None
    error_message: Optional[str] = None
    retry_after: Optional[int] = None


class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, max_requests: int = 3, time_window: float = 1.0):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum number of requests allowed
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """Acquire permission to make a request."""
        async with self._lock:
            now = time.time()
            
            # Remove old requests outside the time window
            self.requests = [req_time for req_time in self.requests 
                           if now - req_time < self.time_window]
            
            # Check if we've exceeded the limit
            if len(self.requests) >= self.max_requests:
                sleep_time = self.time_window - (now - self.requests[0])
                if sleep_time > 0:
                    logger.info(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                    await asyncio.sleep(sleep_time)
                    # Remove the oldest request after sleeping
                    self.requests.pop(0)
            
            # Record this request
            self.requests.append(now)
    
    def reset(self) -> None:
        """Reset the rate limiter."""
        self.requests.clear()


class NotionClient:
    """
    Authenticated Notion API client with error handling and rate limiting.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the Notion client.
        
        Args:
            config: Configuration object (creates default if None)
        """
        self.config = config or Config()
        self._client: Optional[Any] = None
        self.rate_limiter = RateLimiter(max_requests=3, time_window=1.0)
        self._authenticated = False
        
        # Initialize client
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize the Notion client with authentication."""
        try:
            api_key = self.config.get_notion_api_key()
            self._client = Client(auth=api_key)
            self._authenticated = True
            logger.info("Notion client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Notion client: {str(e)}")
            self._authenticated = False
            raise
    
    def _handle_notion_error(self, error: Exception) -> NotionAPIResponse:
        """
        Handle and categorize Notion API errors.
        
        Args:
            error: The exception that occurred
            
        Returns:
            NotionAPIResponse with error details
        """
        if isinstance(error, HTTPResponseError):
            status_code = error.status
            
            if status_code == 401:
                logger.error("Notion API authentication failed")
                return NotionAPIResponse(
                    success=False,
                    error_type=NotionErrorType.AUTHENTICATION,
                    error_message="Invalid or expired API key"
                )
            elif status_code == 429:
                # Extract retry-after header if available
                retry_after = getattr(error, 'headers', {}).get('retry-after')
                retry_after = int(retry_after) if retry_after else 60
                
                logger.warning(f"Rate limit exceeded, retry after {retry_after} seconds")
                return NotionAPIResponse(
                    success=False,
                    error_type=NotionErrorType.RATE_LIMIT,
                    error_message="Rate limit exceeded",
                    retry_after=retry_after
                )
            elif status_code == 404:
                logger.error("Notion object not found")
                return NotionAPIResponse(
                    success=False,
                    error_type=NotionErrorType.OBJECT_NOT_FOUND,
                    error_message="The requested object was not found"
                )
            elif status_code == 400:
                logger.error("Invalid request to Notion API")
                return NotionAPIResponse(
                    success=False,
                    error_type=NotionErrorType.INVALID_REQUEST,
                    error_message="Invalid request parameters"
                )
        
        elif isinstance(error, RequestTimeoutError):
            logger.error("Notion API request timed out")
            return NotionAPIResponse(
                success=False,
                error_type=NotionErrorType.TIMEOUT,
                error_message="Request timed out"
            )
        
        elif isinstance(error, APIResponseError):
            logger.error(f"Notion API response error: {str(error)}")
            return NotionAPIResponse(
                success=False,
                error_type=NotionErrorType.VALIDATION_ERROR,
                error_message=str(error)
            )
        
        # Generic error handling
        logger.error(f"Unknown Notion API error: {str(error)}")
        return NotionAPIResponse(
            success=False,
            error_type=NotionErrorType.UNKNOWN,
            error_message=str(error)
        )
    
    async def _make_request(self, operation: str, *args, **kwargs) -> NotionAPIResponse:
        """
        Make a rate-limited request to the Notion API.
        
        Args:
            operation: The operation name (method name on the client)
            *args: Positional arguments for the operation
            **kwargs: Keyword arguments for the operation
            
        Returns:
            NotionAPIResponse with the result
        """
        if not self._authenticated:
            return NotionAPIResponse(
                success=False,
                error_type=NotionErrorType.AUTHENTICATION,
                error_message="Client not authenticated"
            )
        
        await self.rate_limiter.acquire()
        
        try:
            # Get the operation method from the client
            client_method = getattr(self._client, operation)
            result = client_method(*args, **kwargs)
            
            logger.debug(f"Notion API {operation} successful")
            return NotionAPIResponse(success=True, data=result)
            
        except Exception as e:
            return self._handle_notion_error(e)
    
    async def get_database(self, database_id: str) -> NotionAPIResponse:
        """
        Retrieve a database by ID.
        
        Args:
            database_id: The database ID to retrieve
            
        Returns:
            NotionAPIResponse with database data
        """
        if not InputValidator.validate_database_id(database_id):
            return NotionAPIResponse(
                success=False,
                error_type=NotionErrorType.VALIDATION_ERROR,
                error_message="Invalid database ID format"
            )
        
        logger.info(f"Getting database: {database_id}")
        return await self._make_request("databases.retrieve", database_id=database_id)
    
    async def query_database(self, database_id: str, query: Optional[Dict[str, Any]] = None) -> NotionAPIResponse:
        """
        Query a database.
        
        Args:
            database_id: The database ID to query
            query: Optional query parameters
            
        Returns:
            NotionAPIResponse with query results
        """
        if not InputValidator.validate_database_id(database_id):
            return NotionAPIResponse(
                success=False,
                error_type=NotionErrorType.VALIDATION_ERROR,
                error_message="Invalid database ID format"
            )
        
        query = query or {}
        logger.info(f"Querying database: {database_id}")
        return await self._make_request("databases.query", database_id=database_id, **query)
    
    async def get_page(self, page_id: str) -> NotionAPIResponse:
        """
        Retrieve a page by ID.
        
        Args:
            page_id: The page ID to retrieve
            
        Returns:
            NotionAPIResponse with page data
        """
        if not InputValidator.validate_notion_page_id(page_id):
            return NotionAPIResponse(
                success=False,
                error_type=NotionErrorType.VALIDATION_ERROR,
                error_message="Invalid page ID format"
            )
        
        logger.info(f"Getting page: {page_id}")
        return await self._make_request("pages.retrieve", page_id=page_id)
    
    async def get_block_children(self, block_id: str, page_size: int = 100) -> NotionAPIResponse:
        """
        Get children of a block.
        
        Args:
            block_id: The block ID to get children for
            page_size: Number of results per page (max 100)
            
        Returns:
            NotionAPIResponse with block children
        """
        if not InputValidator.validate_notion_page_id(block_id):
            return NotionAPIResponse(
                success=False,
                error_type=NotionErrorType.VALIDATION_ERROR,
                error_message="Invalid block ID format"
            )
        
        if page_size > 100:
            page_size = 100
        
        logger.info(f"Getting block children: {block_id}")
        return await self._make_request(
            "blocks.children.list", 
            block_id=block_id, 
            page_size=page_size
        )
    
    async def search(self, query: str, page_size: int = 10) -> NotionAPIResponse:
        """
        Search for pages and databases.
        
        Args:
            query: Search query
            page_size: Number of results per page (max 100)
            
        Returns:
            NotionAPIResponse with search results
        """
        try:
            # Validate and sanitize the query
            secure_query = SecureQueryInput(query=query, limit=min(page_size, 100))
            
            search_params = {
                "query": secure_query.query,
                "page_size": secure_query.limit
            }
            
            logger.info(f"Searching Notion: {secure_query.query}")
            return await self._make_request("search", **search_params)
            
        except Exception as e:
            return NotionAPIResponse(
                success=False,
                error_type=NotionErrorType.VALIDATION_ERROR,
                error_message=f"Invalid search parameters: {str(e)}"
            )
    
    async def get_all_pages_from_database(self, database_id: str) -> NotionAPIResponse:
        """
        Get all pages from a database with pagination.
        
        Args:
            database_id: The database ID to get pages from
            
        Returns:
            NotionAPIResponse with all pages
        """
        if not InputValidator.validate_database_id(database_id):
            return NotionAPIResponse(
                success=False,
                error_type=NotionErrorType.VALIDATION_ERROR,
                error_message="Invalid database ID format"
            )
        
        all_pages = []
        has_more = True
        next_cursor = None
        
        logger.info(f"Getting all pages from database: {database_id}")
        
        while has_more:
            query_params = {"page_size": 100}
            if next_cursor:
                query_params["start_cursor"] = next_cursor
            
            response = await self.query_database(database_id, query_params)
            
            if not response.success:
                return response
            
            results = response.data.get("results", [])
            all_pages.extend(results)
            
            has_more = response.data.get("has_more", False)
            next_cursor = response.data.get("next_cursor")
        
        logger.info(f"Retrieved {len(all_pages)} pages from database")
        return NotionAPIResponse(
            success=True,
            data={"results": all_pages, "total_count": len(all_pages)}
        )
    
    def is_authenticated(self) -> bool:
        """Check if the client is authenticated."""
        return self._authenticated
    
    def reset_rate_limiter(self) -> None:
        """Reset the rate limiter (useful for testing)."""
        self.rate_limiter.reset()
    
    async def test_connection(self) -> NotionAPIResponse:
        """
        Test the connection to Notion API.
        
        Returns:
            NotionAPIResponse indicating connection status
        """
        logger.info("Testing Notion API connection")
        return await self._make_request("users.me") 