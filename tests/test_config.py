"""
Tests for the configuration module.
"""

from pathlib import Path
from unittest.mock import patch
from pytest import raises

from notion_rag.config import Config, ChromaDBConfig
from notion_rag.security import SecureNotionConfig


def test_chroma_config_defaults():
    """Test ChromaDB configuration defaults."""
    config = ChromaDBConfig()
    assert config.db_path == "./chroma_db"
    assert config.collection_name == "notion_documents"


def test_secure_notion_config_defaults():
    """Test SecureNotionConfig defaults."""
    config = SecureNotionConfig()
    assert config.api_key is None
    assert config.database_id is None


def test_main_config_defaults():
    """Test main configuration defaults."""
    config = Config()
    assert config.log_level == "INFO"
    assert config.max_retries == 3
    assert config.retry_delay == 1.0
    assert isinstance(config.notion, SecureNotionConfig)
    assert isinstance(config.chroma, ChromaDBConfig)


def test_get_project_root():
    """Test getting project root directory."""
    config = Config()
    root = config.get_project_root()
    assert isinstance(root, Path)
    assert root.exists()


def test_get_chroma_db_path():
    """Test getting ChromaDB path."""
    config = Config()
    db_path = config.get_chroma_db_path()
    assert isinstance(db_path, Path)
    assert "chroma_db" in str(db_path)


@patch('notion_rag.config.SecureKeyManager')
def test_get_notion_api_key_from_config(mock_key_manager):
    """Test getting Notion API key from config."""
    config = Config()
    config.notion.api_key = "test_api_key_123"
    
    result = config.get_notion_api_key()
    
    assert result == "test_api_key_123"
    mock_key_manager.retrieve_api_key.assert_not_called()


@patch('notion_rag.config.SecureKeyManager')
def test_get_notion_api_key_from_keyring(mock_key_manager):
    """Test getting Notion API key from keyring."""
    mock_key_manager.retrieve_api_key.return_value = "keyring_api_key_123"
    config = Config()
    
    result = config.get_notion_api_key()
    
    assert result == "keyring_api_key_123"
    mock_key_manager.retrieve_api_key.assert_called_once_with("notion_api_key")


@patch('notion_rag.config.SecureKeyManager')
def test_get_notion_api_key_not_found(mock_key_manager):
    """Test getting Notion API key when not found."""
    mock_key_manager.retrieve_api_key.return_value = None
    config = Config()
    
    with raises(ValueError, match="Notion API key not found"):
        config.get_notion_api_key()


@patch('notion_rag.config.SecureKeyManager')
def test_set_notion_api_key(mock_key_manager):
    """Test setting Notion API key."""
    mock_key_manager.store_api_key.return_value = True
    config = Config()
    
    result = config.set_notion_api_key("new_api_key_123")
    
    assert result is True
    mock_key_manager.store_api_key.assert_called_once_with("notion_api_key", "new_api_key_123")


@patch('notion_rag.config.SecureKeyManager')
def test_get_openai_api_key_from_config(mock_key_manager):
    """Test getting OpenAI API key from config."""
    config = Config()
    config.openai_api_key = "openai_test_key_123"
    
    result = config.get_openai_api_key()
    
    assert result == "openai_test_key_123"
    mock_key_manager.retrieve_api_key.assert_not_called()


# @patch('notion_rag.config.SecureKeyManager')
# def test_get_openai_api_key_from_keyring(mock_key_manager):
#     """Test getting OpenAI API key from keyring."""
#     mock_key_manager.retrieve_api_key.return_value = "keyring_openai_key_123"
#     config = Config()
    
#     result = config.get_openai_api_key()
    
#     assert result == "keyring_openai_key_123"
#     mock_key_manager.retrieve_api_key.assert_called_once_with("openai_api_key")


@patch('notion_rag.config.SecureKeyManager')
def test_set_openai_api_key(mock_key_manager):
    """Test setting OpenAI API key."""
    mock_key_manager.store_api_key.return_value = True
    config = Config()
    
    result = config.set_openai_api_key("new_openai_key_123")
    
    assert result is True
    mock_key_manager.store_api_key.assert_called_once_with("openai_api_key", "new_openai_key_123")


@patch('notion_rag.config.SecureKeyManager')
def test_delete_api_key(mock_key_manager):
    """Test deleting API key."""
    mock_key_manager.delete_api_key.return_value = True
    config = Config()
    
    result = config.delete_api_key("test_key")
    
    assert result is True
    mock_key_manager.delete_api_key.assert_called_once_with("test_key") 