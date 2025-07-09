"""
Configuration management for the Notion RAG system.
"""

from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import BaseSettings
from .security import SecureKeyManager, SecureNotionConfig


class ChromaDBConfig(BaseModel):
    """ChromaDB configuration settings."""
    
    db_path: str = Field(default="./chroma_db", description="Path to ChromaDB storage")
    collection_name: str = Field(default="notion_documents", description="Collection name")


class Config(BaseSettings):
    """Main configuration class with enhanced security."""
    
    # Application settings
    log_level: str = Field(default="INFO", description="Logging level")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, description="Retry delay in seconds")
    
    # Component configs
    notion: SecureNotionConfig = Field(default_factory=SecureNotionConfig)
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
        """Get Notion API key from config or secure keyring."""
        if self.notion.api_key:
            return self.notion.api_key
        
        # Try to get from secure keyring
        key = SecureKeyManager.retrieve_api_key("notion_api_key")
        if key:
            return key
        
        raise ValueError("Notion API key not found in config or keyring")
    
    def set_notion_api_key(self, api_key: str) -> bool:
        """Set Notion API key in secure keyring."""
        return SecureKeyManager.store_api_key("notion_api_key", api_key)
    
    def get_openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key from config or secure keyring."""
        if self.openai_api_key:
            return self.openai_api_key
        
        return SecureKeyManager.retrieve_api_key("openai_api_key")
    
    def set_openai_api_key(self, api_key: str) -> bool:
        """Set OpenAI API key in secure keyring."""
        return SecureKeyManager.store_api_key("openai_api_key", api_key)
    
    def get_huggingface_api_key(self) -> Optional[str]:
        """Get HuggingFace API key from config or secure keyring."""
        if self.huggingface_api_key:
            return self.huggingface_api_key
        
        return SecureKeyManager.retrieve_api_key("huggingface_api_key")
    
    def set_huggingface_api_key(self, api_key: str) -> bool:
        """Set HuggingFace API key in secure keyring."""
        return SecureKeyManager.store_api_key("huggingface_api_key", api_key)
    
    def delete_api_key(self, key_name: str) -> bool:
        """Delete an API key from secure keyring."""
        return SecureKeyManager.delete_api_key(key_name)
    
    def get_project_root(self) -> Path:
        """Get the project root directory."""
        return Path.cwd()
    
    def get_chroma_db_path(self) -> Path:
        """Get the full path to ChromaDB storage."""
        return self.get_project_root() / self.chroma.db_path 