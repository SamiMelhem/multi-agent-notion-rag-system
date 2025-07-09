"""
Tests for the security module.
"""

from pytest import raises
from unittest.mock import patch
from pydantic import ValidationError

from notion_rag.security import (
    SecureKeyManager,
    InputValidator,
    SecureNotionConfig,
    SecureTextInput,
    SecureQueryInput
)


class TestSecureKeyManager:
    """Test cases for SecureKeyManager."""
    
    def test_validate_key_name_valid(self):
        """Test valid key name validation."""
        assert SecureKeyManager._validate_key_name("notion_api_key")
        assert SecureKeyManager._validate_key_name("openai-api-key")
        assert SecureKeyManager._validate_key_name("test123")
        assert SecureKeyManager._validate_key_name("a")
    
    def test_validate_key_name_invalid(self):
        """Test invalid key name validation."""
        assert not SecureKeyManager._validate_key_name("")
        assert not SecureKeyManager._validate_key_name("key with spaces")
        assert not SecureKeyManager._validate_key_name("key@symbol")
        assert not SecureKeyManager._validate_key_name(None)
        assert not SecureKeyManager._validate_key_name(123)
    
    def test_validate_api_key_valid(self):
        """Test valid API key validation."""
        assert SecureKeyManager._validate_api_key("secret_key_123")
        assert SecureKeyManager._validate_api_key("sk-1234567890abcdef")
        assert SecureKeyManager._validate_api_key("abc.def-123_456")
    
    def test_validate_api_key_invalid(self):
        """Test invalid API key validation."""
        assert not SecureKeyManager._validate_api_key("")
        assert not SecureKeyManager._validate_api_key("short")
        assert not SecureKeyManager._validate_api_key("key with spaces")
        assert not SecureKeyManager._validate_api_key("key@symbol!")
        assert not SecureKeyManager._validate_api_key(None)
        assert not SecureKeyManager._validate_api_key(123)
    
    @patch('notion_rag.security.keyring')
    def test_store_api_key_success(self, mock_keyring):
        """Test successful API key storage."""
        mock_keyring.set_password.return_value = None
        
        result = SecureKeyManager.store_api_key("test_key", "valid_api_key_123")
        
        assert result is True
        mock_keyring.set_password.assert_called_once_with(
            "notion-rag-cli", "test_key", "valid_api_key_123"
        )
    
    @patch('notion_rag.security.keyring')
    def test_store_api_key_failure(self, mock_keyring):
        """Test API key storage failure."""
        mock_keyring.set_password.side_effect = Exception("Keyring error")
        
        result = SecureKeyManager.store_api_key("test_key", "valid_api_key_123")
        
        assert result is False
    
    def test_store_api_key_invalid_key_name(self):
        """Test storing API key with invalid key name."""
        with raises(ValueError, match="Invalid key name"):
            SecureKeyManager.store_api_key("invalid key", "valid_api_key_123")
    
    def test_store_api_key_invalid_api_key(self):
        """Test storing invalid API key."""
        with raises(ValueError, match="Invalid API key format"):
            SecureKeyManager.store_api_key("valid_key", "short")
    
    @patch('notion_rag.security.keyring')
    def test_retrieve_api_key_success(self, mock_keyring):
        """Test successful API key retrieval."""
        mock_keyring.get_password.return_value = "retrieved_key_123"
        
        result = SecureKeyManager.retrieve_api_key("test_key")
        
        assert result == "retrieved_key_123"
        mock_keyring.get_password.assert_called_once_with("notion-rag-cli", "test_key")
    
    @patch('notion_rag.security.keyring')
    def test_retrieve_api_key_not_found(self, mock_keyring):
        """Test API key retrieval when key not found."""
        mock_keyring.get_password.return_value = None
        
        result = SecureKeyManager.retrieve_api_key("test_key")
        
        assert result is None
    
    @patch('notion_rag.security.keyring')
    def test_retrieve_api_key_failure(self, mock_keyring):
        """Test API key retrieval failure."""
        mock_keyring.get_password.side_effect = Exception("Keyring error")
        
        result = SecureKeyManager.retrieve_api_key("test_key")
        
        assert result is None
    
    def test_retrieve_api_key_invalid_key_name(self):
        """Test retrieving API key with invalid key name."""
        with raises(ValueError, match="Invalid key name"):
            SecureKeyManager.retrieve_api_key("invalid key")
    
    @patch('notion_rag.security.keyring')
    def test_delete_api_key_success(self, mock_keyring):
        """Test successful API key deletion."""
        mock_keyring.delete_password.return_value = None
        
        result = SecureKeyManager.delete_api_key("test_key")
        
        assert result is True
        mock_keyring.delete_password.assert_called_once_with("notion-rag-cli", "test_key")
    
    @patch('notion_rag.security.keyring')
    def test_delete_api_key_failure(self, mock_keyring):
        """Test API key deletion failure."""
        mock_keyring.delete_password.side_effect = Exception("Keyring error")
        
        result = SecureKeyManager.delete_api_key("test_key")
        
        assert result is False


class TestInputValidator:
    """Test cases for InputValidator."""
    
    def test_sanitize_text_basic(self):
        """Test basic text sanitization."""
        result = InputValidator.sanitize_text("  Hello World  ")
        assert result == "Hello World"
    
    def test_sanitize_text_control_characters(self):
        """Test removal of control characters."""
        text_with_control = "Hello\x00\x08\x0C\x1FWorld"
        result = InputValidator.sanitize_text(text_with_control)
        assert result == "HelloWorld"
    
    def test_sanitize_text_preserve_newlines_tabs(self):
        """Test preservation of newlines and tabs."""
        text = "Hello\nWorld\tTest"
        result = InputValidator.sanitize_text(text)
        assert result == "Hello\nWorld\tTest"
    
    def test_sanitize_text_length_limit(self):
        """Test text length limitation."""
        long_text = "a" * 15000
        result = InputValidator.sanitize_text(long_text)
        assert len(result) == InputValidator.MAX_TEXT_LENGTH
    
    def test_sanitize_text_custom_length(self):
        """Test text sanitization with custom length."""
        long_text = "a" * 100
        result = InputValidator.sanitize_text(long_text, max_length=50)
        assert len(result) == 50
    
    def test_sanitize_text_non_string_input(self):
        """Test sanitization with non-string input."""
        result = InputValidator.sanitize_text(123)
        assert result == "123"
    
    def test_validate_notion_page_id_valid(self):
        """Test valid Notion page ID validation."""
        # With hyphens
        assert InputValidator.validate_notion_page_id("123e4567-e89b-12d3-a456-426614174000")
        # Without hyphens
        assert InputValidator.validate_notion_page_id("123e4567e89b12d3a456426614174000")
    
    def test_validate_notion_page_id_invalid(self):
        """Test invalid Notion page ID validation."""
        assert not InputValidator.validate_notion_page_id("short")
        assert not InputValidator.validate_notion_page_id("123e4567-e89b-12d3-a456-42661417400g")  # invalid char
        assert not InputValidator.validate_notion_page_id("123e4567-e89b-12d3-a456-4266141740000")  # too long
        assert not InputValidator.validate_notion_page_id("")
        assert not InputValidator.validate_notion_page_id(None)
        assert not InputValidator.validate_notion_page_id(123)
    
    def test_validate_database_id_valid(self):
        """Test valid database ID validation."""
        assert InputValidator.validate_database_id("123e4567-e89b-12d3-a456-426614174000")
        assert InputValidator.validate_database_id("123e4567e89b12d3a456426614174000")
    
    def test_validate_database_id_invalid(self):
        """Test invalid database ID validation."""
        assert not InputValidator.validate_database_id("invalid")
        assert not InputValidator.validate_database_id("")
        assert not InputValidator.validate_database_id(None)


class TestSecureNotionConfig:
    """Test cases for SecureNotionConfig."""
    
    def test_valid_config(self):
        """Test valid Notion configuration."""
        config = SecureNotionConfig(
            api_key="valid_api_key_123",
            database_id="123e4567e89b12d3a456426614174000"
        )
        assert config.api_key == "valid_api_key_123"
        assert config.database_id == "123e4567e89b12d3a456426614174000"
    
    def test_invalid_api_key(self):
        """Test invalid API key validation."""
        with raises(ValidationError):
            SecureNotionConfig(api_key="short")
    
    def test_invalid_database_id(self):
        """Test invalid database ID validation."""
        with raises(ValidationError):
            SecureNotionConfig(database_id="invalid")
    
    def test_optional_fields(self):
        """Test optional fields."""
        config = SecureNotionConfig()
        assert config.api_key is None
        assert config.database_id is None


class TestSecureTextInput:
    """Test cases for SecureTextInput."""
    
    def test_valid_input(self):
        """Test valid text input."""
        input_data = SecureTextInput(
            content="This is valid content",
            title="Valid Title",
            description="Valid description"
        )
        assert input_data.content == "This is valid content"
        assert input_data.title == "Valid Title"
        assert input_data.description == "Valid description"
    
    def test_content_required(self):
        """Test that content is required."""
        with raises(ValidationError):
            SecureTextInput()
    
    def test_content_too_long(self):
        """Test content length validation after sanitization."""
        # Content gets sanitized first, so it should be truncated to max length
        long_content = "a" * (InputValidator.MAX_TEXT_LENGTH + 1)
        input_data = SecureTextInput(content=long_content)
        assert len(input_data.content) == InputValidator.MAX_TEXT_LENGTH
    
    def test_title_too_long(self):
        """Test title length validation."""
        long_title = "a" * (InputValidator.MAX_TITLE_LENGTH + 1)
        with raises(ValidationError):
            SecureTextInput(content="Valid content", title=long_title)
    
    def test_description_too_long(self):
        """Test description length validation."""
        long_desc = "a" * (InputValidator.MAX_DESCRIPTION_LENGTH + 1)
        with raises(ValidationError):
            SecureTextInput(content="Valid content", description=long_desc)
    
    def test_text_sanitization(self):
        """Test automatic text sanitization."""
        input_data = SecureTextInput(
            content="  Content with\x00control chars  ",
            title="  Title with\x08control  ",
            description="  Description\x1F  "
        )
        assert input_data.content == "Content withcontrol chars"
        assert input_data.title == "Title withcontrol"
        assert input_data.description == "Description"


class TestSecureQueryInput:
    """Test cases for SecureQueryInput."""
    
    def test_valid_query(self):
        """Test valid query input."""
        query = SecureQueryInput(query="search term", limit=5)
        assert query.query == "search term"
        assert query.limit == 5
    
    def test_query_required(self):
        """Test that query is required."""
        with raises(ValidationError):
            SecureQueryInput()
    
    def test_query_too_long(self):
        """Test query length validation after sanitization."""
        # Query gets sanitized first, so it should be truncated to max length
        long_query = "a" * 501
        query = SecureQueryInput(query=long_query)
        assert len(query.query) == 500
    
    def test_limit_validation(self):
        """Test limit validation."""
        with raises(ValidationError):
            SecureQueryInput(query="test", limit=0)
        
        with raises(ValidationError):
            SecureQueryInput(query="test", limit=101)
    
    def test_default_limit(self):
        """Test default limit value."""
        query = SecureQueryInput(query="test")
        assert query.limit == 10
    
    def test_query_sanitization(self):
        """Test automatic query sanitization."""
        query = SecureQueryInput(query="  search\x00term  ")
        assert query.query == "searchterm" 