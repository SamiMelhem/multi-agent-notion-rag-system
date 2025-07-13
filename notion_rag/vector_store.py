"""
ChromaDB vector store integration for the Notion RAG system.
Provides local persistent storage and collection management utilities.
"""

import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from logging import getLogger

import chromadb
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection
from chromadb.api.types import (
    Documents,
    Embeddings,
    Metadatas,
    QueryResult
)

from .config import Config
from .security import InputValidator
from .embeddings import create_embedding_generator
from .chunking import chunk_text

logger = getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a chunk of a document for vector storage."""
    
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    
    def __post_init__(self):
        """Validate and sanitize the document chunk."""
        if not self.id:
            self.id = str(uuid.uuid4())
        
        # Sanitize content
        self.content = InputValidator.sanitize_text(self.content)
        
        # Ensure metadata is a dict
        if not isinstance(self.metadata, dict):
            self.metadata = {}
        
        # Add default metadata if not present
        if 'chunk_id' not in self.metadata:
            self.metadata['chunk_id'] = self.id
        if 'created_at' not in self.metadata:
            self.metadata['created_at'] = str(uuid.uuid1().time)


class ChromaDBManager:
    """Manages ChromaDB operations with local persistent storage."""
    
    def __init__(self, config: Config):
        """
        Initialize ChromaDB manager.
        
        Args:
            config: Configuration object containing ChromaDB settings
        """
        self.config = config
        self.db_path = self._get_db_path()
        self.client = self._initialize_client()
        self._collections: Dict[str, Collection] = {}
        
        # Initialize embedding generator
        try:
            self.embedding_generator = create_embedding_generator(config)
            logger.info(f"Embedding generator initialized: {self.embedding_generator.model_name}")
        except ImportError as e:
            logger.warning(f"Embedding generator not available: {e}")
            self.embedding_generator = None
        
        logger.info(f"ChromaDB initialized at: {self.db_path}")
    
    def _get_db_path(self) -> Path:
        """Get the ChromaDB storage path."""
        db_path = self.config.get_chroma_db_path()
        db_path.mkdir(parents=True, exist_ok=True)
        return db_path
    
    def _initialize_client(self) -> chromadb.PersistentClient:
        """Initialize ChromaDB persistent client."""
        try:
            client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info("ChromaDB client initialized successfully")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {str(e)}")
            raise
    
    def get_or_create_collection(
        self, 
        name: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Collection:
        """
        Get an existing collection or create a new one.
        
        Args:
            name: Collection name
            metadata: Optional metadata for the collection
            
        Returns:
            Collection: ChromaDB collection object
        """
        if name in self._collections:
            return self._collections[name]
        
        try:
            # Try to get existing collection
            collection = self.client.get_collection(name=name)
            logger.info(f"Retrieved existing collection: {name}")
        except Exception:
            # Create new collection if it doesn't exist
            collection = self.client.create_collection(
                name=name,
                metadata=metadata or {}
            )
            logger.info(f"Created new collection: {name}")
        
        self._collections[name] = collection
        return collection
    
    def delete_collection(self, name: str) -> bool:
        """
        Delete a collection.
        
        Args:
            name: Collection name to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.client.delete_collection(name=name)
            if name in self._collections:
                del self._collections[name]
            logger.info(f"Deleted collection: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection {name}: {str(e)}")
            return False
    
    def list_collections(self) -> List[Dict[str, Any]]:
        """
        List all collections with their metadata.
        
        Returns:
            List[Dict[str, Any]]: List of collection information
        """
        try:
            collections = self.client.list_collections()
            collection_info = []
            
            for collection in collections:
                info = {
                    'name': collection.name,
                    'metadata': collection.metadata,
                    'count': collection.count()
                }
                collection_info.append(info)
            
            return collection_info
        except Exception as e:
            logger.error(f"Failed to list collections: {str(e)}")
            return []
    
    def get_collection_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a collection.
        
        Args:
            name: Collection name
            
        Returns:
            Optional[Dict[str, Any]]: Collection information or None
        """
        try:
            collection = self.get_or_create_collection(name)
            return {
                'name': collection.name,
                'metadata': collection.metadata,
                'count': collection.count()
            }
        except Exception as e:
            logger.error(f"Failed to get collection info for {name}: {str(e)}")
            return None
    
    def add_documents(
        self, 
        collection_name: str, 
        documents: List[DocumentChunk],
        batch_size: int = 100,
        generate_embeddings: bool = True
    ) -> bool:
        """
        Add documents to a collection in batches.
        
        Args:
            collection_name: Name of the collection
            documents: List of document chunks to add
            batch_size: Number of documents to process in each batch
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            collection = self.get_or_create_collection(collection_name)
            
            # Generate embeddings if requested and generator is available
            if generate_embeddings and self.embedding_generator:
                logger.info(f"Generating embeddings for {len(documents)} documents")
                texts = [doc.content for doc in documents]
                embeddings = self.embedding_generator.generate_embeddings(texts, batch_size=32)
                
                # Assign embeddings to documents
                for doc, embedding in zip(documents, embeddings):
                    doc.embedding = embedding
            
            # Process documents in batches
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                
                # Prepare batch data
                ids = [doc.id for doc in batch]
                contents = [doc.content for doc in batch]
                metadatas = [doc.metadata for doc in batch]
                batch_embeddings = [doc.embedding for doc in batch if doc.embedding]
                
                # Add to collection
                if batch_embeddings and len(batch_embeddings) == len(batch):
                    collection.add(
                        ids=ids,
                        documents=contents,
                        metadatas=metadatas,
                        embeddings=batch_embeddings
                    )
                else:
                    collection.add(
                        ids=ids,
                        documents=contents,
                        metadatas=metadatas
                    )
                
                logger.info(f"Added batch {i//batch_size + 1} to collection {collection_name}")
            
            logger.info(f"Successfully added {len(documents)} documents to collection {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents to collection {collection_name}: {str(e)}")
            return False
    
    def search_documents(
        self,
        collection_name: str,
        query: str,
        query_embedding: Optional[List[float]] = None,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        use_embeddings: bool = True
    ) -> Optional[QueryResult]:
        """
        Search for documents in a collection.
        
        Args:
            collection_name: Name of the collection to search
            query: Text query for search
            query_embedding: Optional pre-computed embedding for the query
            n_results: Number of results to return
            where: Filter conditions for metadata
            where_document: Filter conditions for document content
            
        Returns:
            Optional[QueryResult]: Search results or None if failed
        """
        try:
            collection = self.get_or_create_collection(collection_name)
            
            # Prepare query parameters
            query_params = {
                'n_results': n_results,
                'include': ['metadatas', 'documents', 'distances']
            }
            
            if where:
                query_params['where'] = where
            if where_document:
                query_params['where_document'] = where_document
            
            # Generate query embedding if needed
            if use_embeddings and self.embedding_generator and not query_embedding:
                logger.info(f"Generating embedding for query: {query[:50]}...")
                query_embedding = self.embedding_generator.generate_single_embedding(query)
            
            # Perform search
            if query_embedding:
                logger.info("Performing vector similarity search")
                results = collection.query(
                    query_embeddings=[query_embedding],
                    **query_params
                )
            else:
                logger.info("Performing text-based search")
                results = collection.query(
                    query_texts=[query],
                    **query_params
                )
            
            logger.info(f"Search completed for collection {collection_name}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search collection {collection_name}: {str(e)}")
            return None
    
    def update_document(
        self,
        collection_name: str,
        document_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None
    ) -> bool:
        """
        Update a specific document in a collection.
        
        Args:
            collection_name: Name of the collection
            document_id: ID of the document to update
            content: New content (optional)
            metadata: New metadata (optional)
            embedding: New embedding (optional)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            collection = self.get_or_create_collection(collection_name)
            
            update_params = {'ids': [document_id]}
            
            if content is not None:
                update_params['documents'] = [content]
            if metadata is not None:
                update_params['metadatas'] = [metadata]
            if embedding is not None:
                update_params['embeddings'] = [embedding]
            
            collection.update(**update_params)
            logger.info(f"Updated document {document_id} in collection {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update document {document_id}: {str(e)}")
            return False
    
    def delete_documents(
        self,
        collection_name: str,
        document_ids: List[str]
    ) -> bool:
        """
        Delete documents from a collection.
        
        Args:
            collection_name: Name of the collection
            document_ids: List of document IDs to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            collection = self.get_or_create_collection(collection_name)
            collection.delete(ids=document_ids)
            logger.info(f"Deleted {len(document_ids)} documents from collection {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete documents from collection {collection_name}: {str(e)}")
            return False
    
    def get_document(
        self,
        collection_name: str,
        document_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get a specific document from a collection.
        
        Args:
            collection_name: Name of the collection
            document_id: ID of the document to retrieve
            
        Returns:
            Optional[Dict[str, Any]]: Document data or None
        """
        try:
            collection = self.get_or_create_collection(collection_name)
            result = collection.get(
                ids=[document_id],
                include=['metadatas', 'documents', 'embeddings']
            )
            
            if result['ids']:
                return {
                    'id': result['ids'][0],
                    'content': result['documents'][0],
                    'metadata': result['metadatas'][0],
                    'embedding': result['embeddings'][0] if result['embeddings'] else None
                }
            return None
            
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {str(e)}")
            return None
    
    def clear_collection(self, collection_name: str) -> bool:
        """
        Clear all documents from a collection.
        
        Args:
            collection_name: Name of the collection to clear
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            collection = self.get_or_create_collection(collection_name)
            collection.delete(where={})  # Delete all documents
            logger.info(f"Cleared collection {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear collection {collection_name}: {str(e)}")
            return False
    
    def get_collection_stats(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """
        Get statistics about a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Optional[Dict[str, Any]]: Collection statistics or None
        """
        try:
            collection = self.get_or_create_collection(collection_name)
            
            # Get all documents to calculate stats
            all_docs = collection.get(
                include=['metadatas', 'documents']
            )
            
            if not all_docs['ids']:
                return {
                    'name': collection_name,
                    'document_count': 0,
                    'total_chars': 0,
                    'avg_chars_per_doc': 0
                }
            
            # Calculate statistics
            total_chars = sum(len(doc) for doc in all_docs['documents'])
            avg_chars = total_chars / len(all_docs['documents'])
            
            return {
                'name': collection_name,
                'document_count': len(all_docs['ids']),
                'total_chars': total_chars,
                'avg_chars_per_doc': round(avg_chars, 2)
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats for collection {collection_name}: {str(e)}")
            return None
    
    def add_text_as_chunks(
        self,
        collection_name: str,
        text: str,
        metadata: Dict[str, Any],
        chunk_size: int = 512,
        overlap: int = 50,
        generate_embeddings: bool = True
    ) -> bool:
        """
        Add text to collection by chunking it and generating embeddings.
        
        Args:
            collection_name: Name of the collection
            text: Text to chunk and add
            metadata: Metadata for the text
            chunk_size: Size of each chunk in tokens
            overlap: Overlap between chunks in tokens
            generate_embeddings: Whether to generate embeddings
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Chunk the text
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            logger.info(f"Created {len(chunks)} chunks from text")
            
            # Create DocumentChunk objects
            documents = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'chunk_size': chunk_size,
                    'overlap': overlap
                })
                
                doc = DocumentChunk(
                    id=f"{metadata.get('source_id', 'unknown')}_chunk_{i}",
                    content=chunk,
                    metadata=chunk_metadata
                )
                documents.append(doc)
            
            # Add documents to collection
            return self.add_documents(
                collection_name=collection_name,
                documents=documents,
                generate_embeddings=generate_embeddings
            )
            
        except Exception as e:
            logger.error(f"Failed to add text as chunks: {str(e)}")
            return False
    
    def close(self) -> None:
        """Close the ChromaDB client connection."""
        try:
            if hasattr(self.client, 'close'):
                self.client.close()
            logger.info("ChromaDB client connection closed")
        except Exception as e:
            logger.error(f"Error closing ChromaDB client: {str(e)}") 