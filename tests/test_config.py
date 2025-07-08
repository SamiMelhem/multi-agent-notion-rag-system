"""
Tests for the configuration module.
"""
from pathlib import Path

from notion_rag.config import Config, ChromaDBConfig, NotionConfig


def test_chroma_config_defaults():
    """Test ChromaDB configuration defaults."""
    config = ChromaDBConfig()
    assert config.db_path == "./chroma_db"
    assert config.collection_name == "notion_documents"


def test_notion_config_defaults():
    """Test Notion configuration defaults."""
    config = NotionConfig()
    assert config.api_key is None
    assert config.database_id is None


def test_main_config_defaults():
    """Test main configuration defaults."""
    config = Config()
    assert config.log_level == "INFO"
    assert config.max_retries == 3
    assert config.retry_delay == 1.0
    assert isinstance(config.notion, NotionConfig)
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