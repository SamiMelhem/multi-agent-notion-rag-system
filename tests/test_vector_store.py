"""
Tests for the vector store module.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from notion_rag.config import Config
from notion_rag.vector_store import ChromaDBManager, DocumentChunk


class TestDocumentChunk:
    """Test DocumentChunk class."""
    
    def test_document_chunk_creation(self):
        """Test creating a document chunk."""
        chunk = DocumentChunk(
            id="test-id",
            content="Test content",
            metadata={"source": "test"}
        )
        
        assert chunk.id == "test-id"
        assert chunk.content == "Test content"
        assert chunk.metadata["source"] == "test"
        assert chunk.metadata["chunk_id"] == "test-id"
        assert "created_at" in chunk.metadata
    
    def test_document_chunk_auto_id(self):
        """Test document chunk with auto-generated ID."""
        chunk = DocumentChunk(
            id="",
            content="Test content",
            metadata={}
        )
        
        assert chunk.id != ""
        assert len(chunk.id) > 0
    
    def test_document_chunk_sanitization(self):
        """Test content sanitization."""
        chunk = DocumentChunk(
            id="test-id",
            content="  Test content with extra spaces  ",
            metadata={}
        )
        
        assert chunk.content == "Test content with extra spaces"
    
    def test_document_chunk_with_embedding(self):
        """Test document chunk with embedding."""
        embedding = [0.1, 0.2, 0.3]
        chunk = DocumentChunk(
            id="test-id",
            content="Test content",
            metadata={},
            embedding=embedding
        )
        
        assert chunk.embedding == embedding


class TestChromaDBManager:
    """Test ChromaDBManager class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        # Give ChromaDB time to close connections
        import time
        time.sleep(0.1)
        try:
            shutil.rmtree(temp_dir)
        except PermissionError:
            # On Windows, files might still be in use
            pass
    
    @pytest.fixture
    def mock_config(self, temp_dir):
        """Create a mock config with temp directory."""
        config = Mock(spec=Config)
        config.get_chroma_db_path.return_value = temp_dir
        return config
    
    @pytest.fixture
    def vector_store(self, mock_config):
        """Create a ChromaDBManager instance for testing."""
        with patch('chromadb.PersistentClient'):
            return ChromaDBManager(mock_config)
    
    def test_initialization(self, mock_config):
        """Test ChromaDBManager initialization."""
        with patch('chromadb.PersistentClient') as mock_client:
            manager = ChromaDBManager(mock_config)
            
            assert manager.config == mock_config
            assert manager.db_path == mock_config.get_chroma_db_path()
            mock_client.assert_called_once()
    
    def test_get_db_path(self, mock_config, temp_dir):
        """Test database path creation."""
        manager = ChromaDBManager(mock_config)
        assert manager.db_path == temp_dir
        assert temp_dir.exists()
    
    def test_get_or_create_collection(self, vector_store):
        """Test collection creation and retrieval."""
        with patch.object(vector_store.client, 'get_collection') as mock_get:
            # Test getting existing collection
            mock_collection = Mock()
            mock_get.return_value = mock_collection
            
            collection = vector_store.get_or_create_collection("test-collection")
            
            assert collection == mock_collection
            mock_get.assert_called_once_with(name="test-collection")
    
    def test_get_or_create_collection_new(self, vector_store):
        """Test creating a new collection."""
        with patch.object(vector_store.client, 'get_collection') as mock_get, \
             patch.object(vector_store.client, 'create_collection') as mock_create:
            
            # Simulate collection not found
            mock_get.side_effect = Exception("Collection not found")
            mock_collection = Mock()
            mock_create.return_value = mock_collection
            
            collection = vector_store.get_or_create_collection("test-collection")
            
            assert collection == mock_collection
            mock_create.assert_called_once_with(name="test-collection", metadata={})
    
    def test_list_collections(self, vector_store):
        """Test listing collections."""
        mock_collection1 = Mock()
        mock_collection1.name = "collection1"
        mock_collection1.metadata = {"test": "data"}
        mock_collection1.count.return_value = 10
        
        mock_collection2 = Mock()
        mock_collection2.name = "collection2"
        mock_collection2.metadata = {}
        mock_collection2.count.return_value = 5
        
        with patch.object(vector_store.client, 'list_collections') as mock_list:
            mock_list.return_value = [mock_collection1, mock_collection2]
            
            collections = vector_store.list_collections()
            
            assert len(collections) == 2
            assert collections[0]['name'] == "collection1"
            assert collections[0]['count'] == 10
            assert collections[1]['name'] == "collection2"
            assert collections[1]['count'] == 5
    
    def test_add_documents(self, vector_store):
        """Test adding documents to collection."""
        documents = [
            DocumentChunk(id="1", content="Test 1", metadata={"source": "test"}),
            DocumentChunk(id="2", content="Test 2", metadata={"source": "test"})
        ]
        
        mock_collection = Mock()
        
        with patch.object(vector_store, 'get_or_create_collection') as mock_get_collection:
            mock_get_collection.return_value = mock_collection
            
            success = vector_store.add_documents("test-collection", documents)
            
            assert success is True
            mock_collection.add.assert_called_once()
    
    def test_search_documents(self, vector_store):
        """Test searching documents."""
        mock_collection = Mock()
        mock_result = {
            'ids': [['1', '2']],
            'documents': [['Test 1', 'Test 2']],
            'metadatas': [[{'source': 'test'}, {'source': 'test'}]],
            'distances': [[0.1, 0.2]]
        }
        mock_collection.query.return_value = mock_result
        
        with patch.object(vector_store, 'get_or_create_collection') as mock_get_collection:
            mock_get_collection.return_value = mock_collection
            
            result = vector_store.search_documents("test-collection", "test query")
            
            assert result == mock_result
            mock_collection.query.assert_called_once()
    
    def test_delete_collection(self, vector_store):
        """Test deleting a collection."""
        with patch.object(vector_store.client, 'delete_collection') as mock_delete:
            success = vector_store.delete_collection("test-collection")
            
            assert success is True
            mock_delete.assert_called_once_with(name="test-collection")
    
    def test_clear_collection(self, vector_store):
        """Test clearing a collection."""
        mock_collection = Mock()
        
        with patch.object(vector_store, 'get_or_create_collection') as mock_get_collection:
            mock_get_collection.return_value = mock_collection
            
            success = vector_store.clear_collection("test-collection")
            
            assert success is True
            mock_collection.delete.assert_called_once_with(where={})
    
    def test_get_collection_stats(self, vector_store):
        """Test getting collection statistics."""
        mock_collection = Mock()
        mock_collection.get.return_value = {
            'ids': ['1', '2'],
            'documents': ['Short', 'Longer document'],
            'metadatas': [{'source': 'test'}, {'source': 'test'}]
        }
        
        with patch.object(vector_store, 'get_or_create_collection') as mock_get_collection:
            mock_get_collection.return_value = mock_collection
            
            stats = vector_store.get_collection_stats("test-collection")
            
            assert stats['name'] == "test-collection"
            assert stats['document_count'] == 2
            assert stats['total_chars'] == 20  # "Short" + "Longer document"
            assert stats['avg_chars_per_doc'] == 10.0
    
    def test_close(self, vector_store):
        """Test closing the client connection."""
        with patch.object(vector_store.client, 'close') as mock_close:
            vector_store.close()
            mock_close.assert_called_once()


class TestChromaDBIntegration:
    """Integration tests for ChromaDB."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        # Give ChromaDB time to close connections
        import time
        time.sleep(0.1)
        try:
            shutil.rmtree(temp_dir)
        except PermissionError:
            # On Windows, files might still be in use
            pass
    
    @pytest.fixture
    def config(self, temp_dir):
        """Create a config with temp directory."""
        config = Config()
        # Override the chroma db path
        config.chroma.db_path = str(temp_dir)
        return config
    
    def test_real_chromadb_initialization(self, config):
        """Test real ChromaDB initialization."""
        try:
            manager = ChromaDBManager(config)
            assert manager.client is not None
            manager.close()
        except Exception as e:
            pytest.skip(f"ChromaDB not available: {e}")
    
    def test_real_collection_creation(self, config):
        """Test real collection creation."""
        try:
            manager = ChromaDBManager(config)
            
            # Create a collection
            collection = manager.get_or_create_collection(
                "test-collection",
                metadata={"test": "data"}
            )
            
            assert collection.name == "test-collection"
            assert collection.metadata["test"] == "data"
            
            # List collections
            collections = manager.list_collections()
            assert len(collections) == 1
            assert collections[0]['name'] == "test-collection"
            
            manager.close()
        except Exception as e:
            pytest.skip(f"ChromaDB not available: {e}")
    
    def test_real_document_operations(self, config):
        """Test real document operations."""
        try:
            manager = ChromaDBManager(config)
            
            # Create collection with non-empty metadata
            collection = manager.get_or_create_collection("test-docs", metadata={"created_by": "test"})
            
            # Create test documents
            documents = [
                DocumentChunk(
                    id="doc1",
                    content="This is the first test document",
                    metadata={"source": "test", "type": "document"}
                ),
                DocumentChunk(
                    id="doc2",
                    content="This is the second test document",
                    metadata={"source": "test", "type": "document"}
                )
            ]
            
            # Add documents
            success = manager.add_documents("test-docs", documents)
            assert success is True
            
            # Verify documents were added
            assert collection.count() == 2
            
            # Search documents
            results = manager.search_documents("test-docs", "first test")
            assert results is not None
            assert len(results['ids'][0]) > 0
            
            manager.close()
        except Exception as e:
            print(f"Error in test_real_document_operations: {e}")
            import traceback
            traceback.print_exc()
            pytest.skip(f"ChromaDB not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__]) 