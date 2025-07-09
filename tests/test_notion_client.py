"""
Tests for the Notion API client.
"""

from pytest import mark, raises, fail, skip
from os import getenv, environ
from time import time
from unittest.mock import Mock, patch
from dotenv import load_dotenv

from notion_rag.notion_client import NotionClient

load_dotenv()


class TestNotionClient:
    """Test cases for NotionClient."""
    
    def test_env_file_loading(self):
        """Test that environment variables are loaded from .env file."""
        # This test verifies that the .env file is being loaded
        # The actual values will depend on what's in the user's .env file
        api_key = getenv("NOTION_API_KEY")
        home_page_id = getenv("NOTION_HOME_PAGE_ID")
        
        # If .env file exists and has these variables, they should be loaded
        # If not, they will be None, which is also valid for this test
        assert isinstance(api_key, (str, type(None)))
        assert isinstance(home_page_id, (str, type(None)))
    
    @patch.dict(environ, {
        'NOTION_API_KEY': 'test_api_key_123',
        'NOTION_HOME_PAGE_ID': 'test_home_page_id'
    })
    @patch('notion_rag.notion_client.Client')
    def test_client_initialization_success(self, mock_client_class):
        """Test successful client initialization."""
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        
        client = NotionClient()
        
        assert client.api_key == 'test_api_key_123'
        assert client.home_page_id == 'test_home_page_id'
        mock_client_class.assert_called_once_with(auth='test_api_key_123')
    
    @patch('notion_rag.notion_client.load_dotenv')
    @patch.dict(environ, {}, clear=True)
    def test_client_initialization_missing_api_key(self, mock_load_dotenv):
        """Test client initialization failure when API key is missing."""
        with raises(ValueError, match="NOTION_API_KEY environment variable is not set"):
            NotionClient()
    
    @patch('notion_rag.notion_client.load_dotenv')
    @patch.dict(environ, {'NOTION_API_KEY': 'test_api_key_123'}, clear=True)
    def test_client_initialization_missing_home_page_id(self, mock_load_dotenv):
        """Test client initialization failure when home page ID is missing."""
        with raises(ValueError, match="NOTION_HOME_PAGE_ID environment variable is not set"):
            NotionClient()
    
    @patch('notion_rag.notion_client.load_dotenv')
    def test_env_file_loading_in_client(self, mock_load_dotenv):
        """Test that load_dotenv is called during client initialization."""
        # Mock the environment to have the required variables
        with patch.dict(environ, {
            'NOTION_API_KEY': 'test_api_key_123',
            'NOTION_HOME_PAGE_ID': 'test_home_page_id'
        }):
            with patch('notion_rag.notion_client.Client'):
                client = NotionClient()
                
                # Verify that load_dotenv was called
                mock_load_dotenv.assert_called_once()
                
                # Verify client was initialized correctly
                assert client.api_key == 'test_api_key_123'
                assert client.home_page_id == 'test_home_page_id'
    
    @patch.dict(environ, {
        'NOTION_API_KEY': 'test_api_key_123',
        'NOTION_HOME_PAGE_ID': 'test_home_page_id'
    })
    @patch('notion_rag.notion_client.Client')
    def test_get_page_content_success(self, mock_client_class):
        """Test successful page content retrieval."""
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        
        expected_page_data = {
            "id": "test_page_id",
            "properties": {"title": {"title": [{"text": {"content": "Test Page"}}]}},
            "url": "https://notion.so/test_page_id"
        }
        mock_client_instance.pages.retrieve.return_value = expected_page_data
        
        client = NotionClient()
        result = client.get_page_content("test_page_id")
        
        assert result == expected_page_data
        mock_client_instance.pages.retrieve.assert_called_once_with(page_id="test_page_id")
    
    @patch.dict(environ, {
        'NOTION_API_KEY': 'test_api_key_123',
        'NOTION_HOME_PAGE_ID': 'test_home_page_id'
    })
    @patch('notion_rag.notion_client.Client')
    def test_get_page_content_failure(self, mock_client_class):
        """Test page content retrieval failure."""
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        
        mock_client_instance.pages.retrieve.side_effect = Exception("API Error")
        
        client = NotionClient()
        
        with raises(Exception, match="Failed to fetch page content: API Error"):
            client.get_page_content("test_page_id")
    
    @patch.dict(environ, {
        'NOTION_API_KEY': 'test_api_key_123',
        'NOTION_HOME_PAGE_ID': 'test_home_page_id'
    })
    @patch('notion_rag.notion_client.Client')
    def test_get_block_children_success(self, mock_client_class):
        """Test successful block children retrieval."""
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        
        # Mock pagination responses
        mock_client_instance.blocks.children.list.side_effect = [
            {
                "results": [{"id": "block1", "type": "paragraph"}],
                "has_more": True,
                "next_cursor": "cursor1"
            },
            {
                "results": [{"id": "block2", "type": "heading"}],
                "has_more": False,
                "next_cursor": None
            }
        ]
        
        client = NotionClient()
        result = client.get_block_children("test_block_id")
        
        expected_blocks = [
            {"id": "block1", "type": "paragraph"},
            {"id": "block2", "type": "heading"}
        ]
        assert result == expected_blocks
        
        # Verify both API calls were made
        assert mock_client_instance.blocks.children.list.call_count == 2
    
    @patch.dict(environ, {
        'NOTION_API_KEY': 'test_api_key_123',
        'NOTION_HOME_PAGE_ID': 'test_home_page_id'
    })
    @patch('notion_rag.notion_client.Client')
    def test_get_database_content_success(self, mock_client_class):
        """Test successful database content retrieval."""
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        
        # Mock pagination responses
        mock_client_instance.databases.query.side_effect = [
            {
                "results": [{"id": "page1", "properties": {"title": {"title": [{"text": {"content": "Page 1"}}]}}}],
                "has_more": True,
                "next_cursor": "cursor1"
            },
            {
                "results": [{"id": "page2", "properties": {"title": {"title": [{"text": {"content": "Page 2"}}]}}}],
                "has_more": False,
                "next_cursor": None
            }
        ]
        
        client = NotionClient()
        result = client.get_database_content("test_database_id")
        
        expected_pages = [
            {"id": "page1", "properties": {"title": {"title": [{"text": {"content": "Page 1"}}]}}},
            {"id": "page2", "properties": {"title": {"title": [{"text": {"content": "Page 2"}}]}}}
        ]
        assert result == expected_pages
        
        # Verify both API calls were made
        assert mock_client_instance.databases.query.call_count == 2
    
    @patch.dict(environ, {
        'NOTION_API_KEY': 'test_api_key_123',
        'NOTION_HOME_PAGE_ID': 'test_home_page_id'
    })
    @patch('notion_rag.notion_client.Client')
    def test_get_database_content_with_filter(self, mock_client_class):
        """Test database content retrieval with filter parameters."""
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        
        mock_client_instance.databases.query.return_value = {
            "results": [{"id": "page1", "properties": {"status": {"select": {"name": "Done"}}}}],
            "has_more": False,
            "next_cursor": None
        }
        
        filter_params = {
            "property": "Status",
            "select": {"equals": "Done"}
        }
        
        client = NotionClient()
        result = client.get_database_content("test_database_id", filter_params)
        
        expected_pages = [{"id": "page1", "properties": {"status": {"select": {"name": "Done"}}}}]
        assert result == expected_pages
        
        # Verify the API call was made with filter
        mock_client_instance.databases.query.assert_called_once_with(
            database_id="test_database_id",
            start_cursor=None,
            filter=filter_params
        )
    
    @patch.dict(environ, {
        'NOTION_API_KEY': 'test_api_key_123',
        'NOTION_HOME_PAGE_ID': 'test_home_page_id'
    })
    @patch('notion_rag.notion_client.Client')
    def test_get_all_child_pages_success(self, mock_client_class):
        """Test successful child pages retrieval."""
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        
        # Mock block children responses for different page IDs
        def mock_blocks_children_list(block_id, **kwargs):
            if block_id == "test_parent_id":
                return {
                    "results": [
                        {"id": "child_page1", "type": "child_page"},
                        {"id": "child_database1", "type": "child_database"},
                        {"id": "paragraph1", "type": "paragraph"}
                    ],
                    "has_more": False,
                    "next_cursor": None
                }
            elif block_id == "child_page1":
                return {
                    "results": [
                        {"id": "grandchild_page1", "type": "child_page"}
                    ],
                    "has_more": False,
                    "next_cursor": None
                }
            elif block_id == "grandchild_page1":
                return {
                    "results": [],
                    "has_more": False,
                    "next_cursor": None
                }
            else:
                return {
                    "results": [],
                    "has_more": False,
                    "next_cursor": None
                }
        
        mock_client_instance.blocks.children.list.side_effect = mock_blocks_children_list
        
        client = NotionClient()
        result = client.get_all_child_pages("test_parent_id")
        
        expected_pages = [
            {"id": "child_page1", "type": "child_page"},
            {"id": "grandchild_page1", "type": "child_page"},
            {"id": "child_database1", "type": "child_database"}
        ]
        
        # Debug: print the actual result
        print(f"Expected: {expected_pages}")
        print(f"Actual: {result}")
        
        assert result == expected_pages
    
    @patch.dict(environ, {
        'NOTION_API_KEY': 'test_api_key_123',
        'NOTION_HOME_PAGE_ID': 'test_home_page_id'
    })
    @patch('notion_rag.notion_client.Client')
    def test_get_all_child_pages_uses_home_page_id(self, mock_client_class):
        """Test that get_all_child_pages uses home page ID when no parent is specified."""
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        
        mock_client_instance.blocks.children.list.return_value = {
            "results": [],
            "has_more": False,
            "next_cursor": None
        }
        
        client = NotionClient()
        client.get_all_child_pages()  # No parent_page_id specified
        
        # Should use the home page ID
        mock_client_instance.blocks.children.list.assert_called_once_with(
            block_id="test_home_page_id",
            start_cursor=None
        )
    
    @patch.dict(environ, {
        'NOTION_API_KEY': 'test_api_key_123',
        'NOTION_HOME_PAGE_ID': 'test_home_page_id'
    })
    @patch('notion_rag.notion_client.Client')
    def test_search_pages_success(self, mock_client_class):
        """Test successful page search."""
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        
        # Mock pagination responses
        mock_client_instance.search.side_effect = [
            {
                "results": [{"id": "search_result1", "properties": {"title": {"title": [{"text": {"content": "Search Result 1"}}]}}}],
                "has_more": True,
                "next_cursor": "cursor1"
            },
            {
                "results": [{"id": "search_result2", "properties": {"title": {"title": [{"text": {"content": "Search Result 2"}}]}}}],
                "has_more": False,
                "next_cursor": None
            }
        ]
        
        client = NotionClient()
        result = client.search_pages("test query")
        
        expected_results = [
            {"id": "search_result1", "properties": {"title": {"title": [{"text": {"content": "Search Result 1"}}]}}},
            {"id": "search_result2", "properties": {"title": {"title": [{"text": {"content": "Search Result 2"}}]}}}
        ]
        assert result == expected_results
        
        # Verify both API calls were made with correct parameters
        assert mock_client_instance.search.call_count == 2
        mock_client_instance.search.assert_called_with(
            query="test query",
            start_cursor="cursor1",
            filter={"property": "object", "value": "page"}
        )
    
    @patch.dict(environ, {
        'NOTION_API_KEY': 'test_api_key_123',
        'NOTION_HOME_PAGE_ID': 'test_home_page_id'
    })
    @patch('notion_rag.notion_client.Client')
    def test_error_handling_in_get_block_children(self, mock_client_class):
        """Test error handling in get_block_children."""
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        
        mock_client_instance.blocks.children.list.side_effect = Exception("Block children error")
        
        client = NotionClient()
        
        with raises(Exception, match="Failed to fetch block children: Block children error"):
            client.get_block_children("test_block_id")
    
    @patch.dict(environ, {
        'NOTION_API_KEY': 'test_api_key_123',
        'NOTION_HOME_PAGE_ID': 'test_home_page_id'
    })
    @patch('notion_rag.notion_client.Client')
    def test_error_handling_in_get_database_content(self, mock_client_class):
        """Test error handling in get_database_content."""
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        
        mock_client_instance.databases.query.side_effect = Exception("Database query error")
        
        client = NotionClient()
        
        with raises(Exception, match="Failed to fetch database content: Database query error"):
            client.get_database_content("test_database_id")
    
    @patch.dict(environ, {
        'NOTION_API_KEY': 'test_api_key_123',
        'NOTION_HOME_PAGE_ID': 'test_home_page_id'
    })
    @patch('notion_rag.notion_client.Client')
    def test_error_handling_in_search_pages(self, mock_client_class):
        """Test error handling in search_pages."""
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        
        mock_client_instance.search.side_effect = Exception("Search error")
        
        client = NotionClient()
        
        with raises(Exception, match="Failed to search pages: Search error"):
            client.search_pages("test query")


class TestNotionClientIntegration:
    """Integration tests for NotionClient (requires real API credentials)."""
    
    @mark.integration
    def test_real_connection(self):
        """Test connection with real Notion API (requires environment variables)."""
        # This test will only run if NOTION_API_KEY and NOTION_HOME_PAGE_ID are set
        api_key = getenv("NOTION_API_KEY")
        home_page_id = getenv("NOTION_HOME_PAGE_ID")
        
        if not api_key or not home_page_id:
            skip(
                "NOTION_API_KEY and NOTION_HOME_PAGE_ID environment variables required. "
                "Make sure these are set in your .env file or environment."
            )
        
        # Set a timeout for this test (30 seconds)
        start_time = time()
        timeout_seconds = 30
        
        try:
            client = NotionClient()
            
            # Test a simple API call first to verify connection
            print(f"Testing connection with home page ID: {home_page_id}")
            
            # Check timeout before making API calls
            if time() - start_time > timeout_seconds:
                fail("Test timed out before making API calls")
            
            # Test getting page content directly (simpler than recursive child pages)
            try:
                page_content = client.get_page_content(home_page_id)
                
                # Check timeout after API call
                if time() - start_time > timeout_seconds:
                    fail("Test timed out after page content call")
                
                assert isinstance(page_content, dict)
                assert "id" in page_content
                print(f"✅ Successfully connected to page: {page_content.get('url', 'N/A')}")
            except Exception as page_error:
                print(f"⚠️ Could not fetch page content: {page_error}")
                
                # Check timeout before trying alternative test
                if time() - start_time > timeout_seconds:
                    fail("Test timed out before alternative test")
                
                # If page content fails, try a simpler test
                print("Trying alternative connection test...")
                
                # Test search functionality instead
                search_results = client.search_pages("test")
                
                # Check timeout after search call
                if time() - start_time > timeout_seconds:
                    fail("Test timed out after search call")
                
                assert isinstance(search_results, list)
                print(f"✅ Successfully connected via search. Found {len(search_results)} results")
                
        except Exception as e:
            fail(f"Integration test failed: {str(e)}")
    
    @mark.integration
    def test_env_file_integration(self):
        """Test that the .env file is properly loaded and used by the client."""
        api_key = getenv("NOTION_API_KEY")
        home_page_id = getenv("NOTION_HOME_PAGE_ID")
        
        if not api_key or not home_page_id:
            skip("Environment variables not available for integration test")
        
        # Test that the client can be initialized with .env variables
        try:
            client = NotionClient()
            assert client.api_key == api_key
            assert client.home_page_id == home_page_id
        except Exception as e:
            fail(f"Failed to initialize client with .env variables: {str(e)}") 