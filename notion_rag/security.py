"""
Security utilities for the Notion RAG system.
Provides secure API key management, input validation, and sanitization.
"""

import keyring
from re import match, sub
from string import ascii_letters, digits, hexdigits
from typing import Optional
from pydantic import BaseModel, Field, field_validator
from logging import getLogger

logger = getLogger(__name__)


class SecureKeyManager:
    """Secure API key management using system keyring."""
    
    SERVICE_NAME = "notion-rag-cli"
    
    @classmethod
    def store_api_key(cls, key_name: str, api_key: str) -> bool:
        """
        Securely store an API key in the system keyring.
        
        Args:
            key_name: Name identifier for the key
            api_key: The API key to store
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not cls._validate_key_name(key_name):
            raise ValueError(f"Invalid key name: {key_name}")
        
        if not cls._validate_api_key(api_key):
            raise ValueError("Invalid API key format")
        
        try:
            keyring.set_password(cls.SERVICE_NAME, key_name, api_key)
            logger.info(f"API key '{key_name}' stored successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to store API key '{key_name}': {str(e)}")
            return False
    
    @classmethod
    def retrieve_api_key(cls, key_name: str) -> Optional[str]:
        """
        Retrieve an API key from the system keyring.
        
        Args:
            key_name: Name identifier for the key
            
        Returns:
            Optional[str]: The API key if found, None otherwise
        """
        if not cls._validate_key_name(key_name):
            raise ValueError(f"Invalid key name: {key_name}")
        
        try:
            key = keyring.get_password(cls.SERVICE_NAME, key_name)
            if key:
                logger.info(f"API key '{key_name}' retrieved successfully")
                return key
            else:
                logger.warning(f"API key '{key_name}' not found in keyring")
                return None
        except Exception as e:
            logger.error(f"Failed to retrieve API key '{key_name}': {str(e)}")
            return None
    
    @classmethod
    def delete_api_key(cls, key_name: str) -> bool:
        """
        Delete an API key from the system keyring.
        
        Args:
            key_name: Name identifier for the key
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not cls._validate_key_name(key_name):
            raise ValueError(f"Invalid key name: {key_name}")
        
        try:
            keyring.delete_password(cls.SERVICE_NAME, key_name)
            logger.info(f"API key '{key_name}' deleted successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to delete API key '{key_name}': {str(e)}")
            return False
    
    @classmethod
    def list_stored_keys(cls) -> list[str]:
        """
        List all stored API key names.
        
        Returns:
            list[str]: List of stored key names
        """
        try:
            # This is a simplified approach - actual implementation may vary by keyring backend
            return []  # Most keyring backends don't support listing keys
        except Exception as e:
            logger.error(f"Failed to list stored keys: {str(e)}")
            return []
    
    @staticmethod
    def _validate_key_name(key_name: str) -> bool:
        """Validate key name format."""
        if not key_name or not isinstance(key_name, str):
            return False
        return bool(match(r'^[a-zA-Z0-9_-]+$', key_name))
    
    @staticmethod
    def _validate_api_key(api_key: str) -> bool:
        """Basic API key format validation."""
        if not api_key or not isinstance(api_key, str):
            return False
        
        # Basic validation - at least 10 characters, alphanumeric with some special chars
        if len(api_key) < 10:
            return False
        
        # Check for common API key patterns
        allowed_chars = ascii_letters + digits + '-_.'
        return all(c in allowed_chars for c in api_key)


class InputValidator:
    """Input validation and sanitization utilities."""
    
    MAX_TEXT_LENGTH = 10000
    MAX_TITLE_LENGTH = 200
    MAX_DESCRIPTION_LENGTH = 1000
    
    @classmethod
    def sanitize_text(cls, text: str, max_length: Optional[int] = None) -> str:
        """
        Sanitize text input by removing harmful characters and limiting length.
        
        Args:
            text: Input text to sanitize
            max_length: Maximum allowed length (defaults to MAX_TEXT_LENGTH)
            
        Returns:
            str: Sanitized text
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Remove null bytes and control characters except newlines and tabs
        text = sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
        
        # Limit length
        max_len = max_length or cls.MAX_TEXT_LENGTH
        if len(text) > max_len:
            text = text[:max_len]
        
        # Remove leading/trailing whitespace
        return text.strip()
    
    @classmethod
    def validate_notion_page_id(cls, page_id: str) -> bool:
        """
        Validate Notion page ID format.
        
        Args:
            page_id: Notion page ID to validate
            
        Returns:
            bool: True if valid format
        """
        if not isinstance(page_id, str):
            return False
        
        # Notion page IDs are UUIDs with or without hyphens
        # Remove hyphens for validation
        clean_id = page_id.replace('-', '')
        
        # Should be 32 hexadecimal characters
        return len(clean_id) == 32 and all(c in hexdigits for c in clean_id)
    
    @classmethod
    def validate_database_id(cls, database_id: str) -> bool:
        """
        Validate Notion database ID format.
        
        Args:
            database_id: Notion database ID to validate
            
        Returns:
            bool: True if valid format
        """
        # Database IDs follow the same format as page IDs
        return cls.validate_notion_page_id(database_id)


class SecureNotionConfig(BaseModel):
    """Secure Notion configuration with validation."""
    
    api_key: Optional[str] = Field(default=None, description="Notion API key")
    database_id: Optional[str] = Field(default=None, description="Default database ID")
    
    @field_validator('api_key')
    @classmethod
    def validate_api_key(cls, v):
        """Validate API key format."""
        if v is not None and not SecureKeyManager._validate_api_key(v):
            raise ValueError("Invalid API key format")
        return v
    
    @field_validator('database_id')
    @classmethod
    def validate_database_id(cls, v):
        """Validate database ID format."""
        if v is not None and not InputValidator.validate_database_id(v):
            raise ValueError("Invalid database ID format")
        return v


class SecureTextInput(BaseModel):
    """Secure text input validation model."""
    
    content: str = Field(..., min_length=1, max_length=InputValidator.MAX_TEXT_LENGTH)
    title: Optional[str] = Field(default=None, max_length=InputValidator.MAX_TITLE_LENGTH)
    description: Optional[str] = Field(default=None, max_length=InputValidator.MAX_DESCRIPTION_LENGTH)
    
    @field_validator('content', 'title', 'description', mode='before')
    @classmethod
    def sanitize_text_fields(cls, v):
        """Sanitize text fields."""
        if v is not None:
            return InputValidator.sanitize_text(v)
        return v


class SecureQueryInput(BaseModel):
    """Secure query input validation model."""
    
    query: str = Field(..., min_length=1, max_length=500)
    limit: Optional[int] = Field(default=10, ge=1, le=100)
    
    @field_validator('query', mode='before')
    @classmethod
    def sanitize_query(cls, v):
        """Sanitize query input."""
        if v is not None:
            return InputValidator.sanitize_text(v, max_length=500)
        return v 