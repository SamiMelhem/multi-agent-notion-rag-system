"""
Configuration management for the Notion RAG system.
"""

import keyring
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict, BaseSettings


class ChromaDBConfig(BaseModel):
    """ChromaDB configuration settings."""
    
    db_path: str = Field(default="./chroma_db", description="Path to ChromaDB storage")
    collection_name: str = Field(default="notion_documents", description="Collection name")


class NotionConfig(BaseModel):
    """Notion API configuration settings."""
    
    api_key: Optional[str] = Field(default=None, description="Notion API key")
    database_id: Optional[str] = Field(default=None, description="Default database ID")


class Config(BaseSettings):
    """Main configuration class."""
    
    # Application settings
    log_level: str = Field(default="INFO", description="Logging level")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, description="Retry delay in seconds")
    
    # Component configs
    notion: NotionConfig = Field(default_factory=NotionConfig)
    chroma: ChromaDBConfig = Field(default_factory=ChromaDBConfig)
    
    # Optional API keys
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    huggingface_api_key: Optional[str] = Field(default=None, description="HuggingFace API key")
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_nested_delimiter="__"
    )
    
    def get_notion_api_key(self) -> str:
        """Get Notion API key from config or keyring."""
        if self.notion.api_key:
            return self.notion.api_key
        
        # Try to get from keyring
        try:
            key = keyring.get_password("notion-rag-cli", "notion_api_key")
            if key:
                return key
        except ImportError:
            pass
        
        raise ValueError("Notion API key not found in config or keyring")
    
    def set_notion_api_key(self, api_key: str) -> None:
        """Set Notion API key in keyring."""
        try:
            keyring.set_password("notion-rag-cli", "notion_api_key", api_key)
        except ImportError:
            raise ValueError("Keyring not available for storing API key")
    
    def get_project_root(self) -> Path:
        """Get the project root directory."""
        return Path.cwd()
    
    def get_chroma_db_path(self) -> Path:
        """Get the full path to ChromaDB storage."""
        return self.get_project_root() / self.chroma.db_path 