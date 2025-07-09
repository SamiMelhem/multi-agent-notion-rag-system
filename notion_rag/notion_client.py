"""
Notion API client with authentication, error handling, and rate limiting.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import os
from notion_client import Client
from dotenv import load_dotenv

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
    """A client for interacting with the Notion API."""
    
    def __init__(self):
        """Initialize the Notion client with API credentials."""
        load_dotenv()
        
        self.api_key = os.getenv("NOTION_API_KEY")
        self.home_page_id = os.getenv("NOTION_HOME_PAGE_ID")
        
        if not self.api_key:
            raise ValueError("NOTION_API_KEY environment variable is not set")
        if not self.home_page_id:
            raise ValueError("NOTION_HOME_PAGE_ID environment variable is not set")
            
        self.client = Client(auth=self.api_key)
    
    def get_page_content(self, page_id: str) -> Dict[str, Any]:
        """
        Fetch the content of a specific page.
        
        Args:
            page_id (str): The ID of the page to fetch
            
        Returns:
            Dict[str, Any]: The page content
        """
        try:
            return self.client.pages.retrieve(page_id=page_id)
        except Exception as e:
            raise Exception(f"Failed to fetch page content: {str(e)}")
    
    def get_block_children(self, block_id: str) -> List[Dict[str, Any]]:
        """
        Fetch all children blocks of a given block (page or block).
        
        Args:
            block_id (str): The ID of the block whose children to fetch
            
        Returns:
            List[Dict[str, Any]]: List of child blocks
        """
        try:
            results = []
            has_more = True
            start_cursor = None
            
            while has_more:
                response = self.client.blocks.children.list(
                    block_id=block_id,
                    start_cursor=start_cursor
                )
                
                results.extend(response["results"])
                has_more = response["has_more"]
                start_cursor = response["next_cursor"]
            
            return results
        except Exception as e:
            raise Exception(f"Failed to fetch block children: {str(e)}")
    
    def get_database_content(self, database_id: str, filter_params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Fetch all pages from a database with optional filtering.
        
        Args:
            database_id (str): The ID of the database to query
            filter_params (Optional[Dict]): Optional filter parameters for the database query
            
        Returns:
            List[Dict[str, Any]]: List of database pages
        """
        try:
            results = []
            has_more = True
            start_cursor = None
            
            while has_more:
                query_params = {
                    "database_id": database_id,
                    "start_cursor": start_cursor
                }
                
                if filter_params:
                    query_params["filter"] = filter_params
                
                response = self.client.databases.query(**query_params)
                
                results.extend(response["results"])
                has_more = response["has_more"]
                start_cursor = response["next_cursor"]
            
            return results
        except Exception as e:
            raise Exception(f"Failed to fetch database content: {str(e)}")
    
    def get_all_child_pages(self, parent_page_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Recursively fetch all child pages under a given parent page.
        If no parent_page_id is provided, uses the home page ID.
        
        Args:
            parent_page_id (Optional[str]): The ID of the parent page
            
        Returns:
            List[Dict[str, Any]]: List of all child pages
        """
        page_id = parent_page_id or self.home_page_id
        try:
            # Get all blocks in the current page
            blocks = self.get_block_children(page_id)
            pages = []
            
            for block in blocks:
                # If the block is a child page or child database, add it to results
                if block["type"] in ["child_page", "child_database"]:
                    pages.append(block)
                    
                    # If it's a page, recursively get its children
                    if block["type"] == "child_page":
                        child_pages = self.get_all_child_pages(block["id"])
                        pages.extend(child_pages)
            
            return pages
        except Exception as e:
            raise Exception(f"Failed to fetch child pages: {str(e)}")
    
    def search_pages(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for pages in the workspace.
        
        Args:
            query (str): The search query
            
        Returns:
            List[Dict[str, Any]]: List of matching pages
        """
        try:
            results = []
            has_more = True
            start_cursor = None
            
            while has_more:
                response = self.client.search(
                    query=query,
                    start_cursor=start_cursor,
                    filter={
                        "property": "object",
                        "value": "page"
                    }
                )
                
                results.extend(response["results"])
                has_more = response["has_more"]
                start_cursor = response["next_cursor"]
            
            return results
        except Exception as e:
            raise Exception(f"Failed to search pages: {str(e)}") 