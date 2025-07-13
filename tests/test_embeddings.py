"""
Tests for embedding generation and chunking functionality.
"""

import pytest
from unittest.mock import Mock, patch
import tempfile
import shutil
from pathlib import Path

from notion_rag.config import Config
from notion_rag.embeddings import (
    EmbeddingGenerator,
    HuggingFaceEmbeddingGenerator,
    create_embedding_generator,
    generate_embeddings
)
from notion_rag.chunking import chunk_text, count_tokens


class TestEmbeddingGenerator:
    """Test the base EmbeddingGenerator class."""
    
    def test_base_class_initialization(self):
        """Test base class initialization."""
        config = Config()
        generator = EmbeddingGenerator(config)
        
        assert generator.model_name == "BAAI/bge-small-en-v1.5"
        assert generator.embedding_dimension == 384
        assert generator.config == config
    
    def test_base_class_methods(self):
        """Test base class methods."""
        config = Config()
        generator = EmbeddingGenerator(config)
        
        # Test get_embedding_dimension
        assert generator.get_embedding_dimension() == 384
        
        # Test generate_embeddings raises NotImplementedError
        with pytest.raises(NotImplementedError):
            generator.generate_embeddings(["test text"])


class TestHuggingFaceEmbeddingGenerator:
    """Test HuggingFace embedding generator."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock config."""
        return Mock(spec=Config)
    
    def test_initialization_without_sentence_transformers(self):
        """Test initialization when sentence-transformers is not available."""
        with patch('notion_rag.embeddings.SENTENCE_TRANSFORMERS_AVAILABLE', False):
            config = Mock(spec=Config)
            with pytest.raises(ImportError, match="sentence-transformers is not installed"):
                HuggingFaceEmbeddingGenerator(config)
    
    def test_initialization_with_custom_model(self, mock_config):
        """Test initialization with custom model name."""
        with patch('notion_rag.embeddings.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('notion_rag.embeddings.SentenceTransformer') as mock_transformer:
            
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 768
            mock_transformer.return_value = mock_model
            
            generator = HuggingFaceEmbeddingGenerator(mock_config, "custom-model")
            
            assert generator.model_name == "custom-model"
            assert generator.embedding_dimension == 768
            mock_transformer.assert_called_once_with("custom-model")
    
    def test_generate_embeddings(self, mock_config):
        """Test embedding generation."""
        with patch('notion_rag.embeddings.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('notion_rag.embeddings.SentenceTransformer') as mock_transformer:
            
            # Mock the model
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_transformer.return_value = mock_model
            
            # Mock the encode method
            import numpy as np
            mock_embeddings = np.array([[0.1, 0.2, 0.3] * 128])  # 384 dimensions
            mock_model.encode.return_value = mock_embeddings
            
            generator = HuggingFaceEmbeddingGenerator(mock_config)
            
            # Test single text
            texts = ["This is a test sentence."]
            embeddings = generator.generate_embeddings(texts)
            
            assert len(embeddings) == 1
            assert len(embeddings[0]) == 384
            mock_model.encode.assert_called_once()
    
    def test_generate_single_embedding(self, mock_config):
        """Test single embedding generation."""
        with patch('notion_rag.embeddings.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('notion_rag.embeddings.SentenceTransformer') as mock_transformer:
            
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_transformer.return_value = mock_model
            
            import numpy as np
            mock_embeddings = np.array([[0.1, 0.2, 0.3] * 128])
            mock_model.encode.return_value = mock_embeddings
            
            generator = HuggingFaceEmbeddingGenerator(mock_config)
            
            embedding = generator.generate_single_embedding("Test text")
            
            assert len(embedding) == 384
            mock_model.encode.assert_called_once()
    
    def test_get_model_info(self, mock_config):
        """Test getting model information."""
        with patch('notion_rag.embeddings.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('notion_rag.embeddings.SentenceTransformer') as mock_transformer:
            
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_model.max_seq_length = 512
            mock_transformer.return_value = mock_model
            
            generator = HuggingFaceEmbeddingGenerator(mock_config)
            
            info = generator.get_model_info()
            
            assert info["model_name"] == "BAAI/bge-small-en-v1.5"
            assert info["embedding_dimension"] == 384
            assert info["max_sequence_length"] == 512
            assert info["model_type"] == "sentence-transformers"


class TestChunking:
    """Test text chunking functionality."""
    
    def test_chunk_text_basic(self):
        """Test basic text chunking."""
        text = "This is a test sentence. " * 100  # Create a longer text
        
        chunks = chunk_text(text, chunk_size=50, overlap=10)
        
        assert len(chunks) > 1
        assert all(len(chunk.strip()) > 0 for chunk in chunks)
    
    def test_chunk_text_single_chunk(self):
        """Test chunking when text fits in single chunk."""
        text = "This is a short text."
        
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        
        assert len(chunks) == 1
        assert chunks[0].strip() == text.strip()
    
    def test_chunk_text_overlap(self):
        """Test that chunks have proper overlap."""
        text = "This is a test sentence with multiple words to ensure proper chunking behavior."
        
        chunks = chunk_text(text, chunk_size=10, overlap=3)
        
        assert len(chunks) > 1
        
        # Check that consecutive chunks share some content (not necessarily complete words)
        for i in range(len(chunks) - 1):
            chunk1 = chunks[i]
            chunk2 = chunks[i + 1]
            
            # Check for overlap by looking for common substrings
            # This is more realistic since chunking works at token level
            overlap_found = False
            for j in range(len(chunk1) - 2):  # Look for 3+ character overlaps
                substring = chunk1[j:j+3]
                if substring in chunk2 and len(substring.strip()) > 0:
                    overlap_found = True
                    break
            
            assert overlap_found, f"No overlap found between chunks {i} and {i+1}"
    
    def test_count_tokens(self):
        """Test token counting."""
        text = "This is a test sentence."
        
        count = count_tokens(text)
        
        assert count > 0
        assert isinstance(count, int)


class TestIntegration:
    """Integration tests for embedding and chunking."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        try:
            shutil.rmtree(temp_dir)
        except PermissionError:
            pass
    
    @pytest.fixture
    def config(self, temp_dir):
        """Create a config with temp directory."""
        config = Config()
        config.chroma.db_path = str(temp_dir)
        return config
    
    def test_create_embedding_generator(self, config):
        """Test creating embedding generator."""
        try:
            generator = create_embedding_generator(config)
            assert isinstance(generator, HuggingFaceEmbeddingGenerator)
            assert generator.model_name == "BAAI/bge-small-en-v1.5"
        except ImportError:
            pytest.skip("sentence-transformers not available")
    
    def test_generate_embeddings_convenience(self, config):
        """Test the convenience function."""
        try:
            texts = ["First sentence.", "Second sentence."]
            embeddings = generate_embeddings(texts, config)
            
            assert len(embeddings) == 2
            assert all(len(emb) > 0 for emb in embeddings)
        except ImportError:
            pytest.skip("sentence-transformers not available")


if __name__ == "__main__":
    pytest.main([__file__]) 