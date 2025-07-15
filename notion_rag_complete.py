#!/usr/bin/env python3
"""
Complete Notion RAG System
Combines fetching, loading, and interactive searching in one seamless workflow.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from notion_rag.config import Config
from notion_rag.vector_store import ChromaDBManager, DocumentChunk
from notion_rag.chunking import chunk_text
from notion_rag.gemini_client import GeminiClient, GeminiMessage
from notion_rag.embeddings import create_embedding_generator
from notion_rag.cost_tracker import get_cost_tracker
from notion_rag.prompt_utils import get_prompt_library, get_summarizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NotionRAGComplete:
    """Complete Notion RAG system with fetch, load, and search capabilities."""
    
    def __init__(self, config: Config):
        """Initialize the complete RAG system."""
        self.config = config
        self.vector_store = None
        self.gemini_client = None
        self.embeddings = None
        self.cost_tracker = None
        self.prompt_library = None
        self.summarizer = None
        
    def run_notion_fetch(self, page_id: str) -> bool:
        """
        Fetch a page and all its children from Notion.
        
        Args:
            page_id: Notion page ID to fetch
            
        Returns:
            bool: True if successful
        """
        print(f"🚀 Fetching page {page_id} and all children from Notion...")
        start_time = time.time()
        
        try:
            import subprocess
            result = subprocess.run([
                sys.executable, "turbo_notion_fetch.py", page_id
            ], capture_output=True, text=True, encoding='utf-8', errors='replace', cwd=project_root)
            
            fetch_time = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ Successfully fetched in {fetch_time:.2f} seconds")
                return True
            else:
                print(f"❌ Failed to fetch: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return False

    def get_cached_page_content(self, page_id: str) -> Dict[str, Any]:
        """Get content from a cached Notion page."""
        try:
            from turbo_notion_fetch import get_page_tree_for_context
            
            page_data = get_page_tree_for_context(page_id, max_depth=3)
            
            if "error" in page_data:
                return None
            
            def extract_text_from_tree(tree_data):
                text_parts = []
                if isinstance(tree_data, list):
                    for item in tree_data:
                        text_parts.extend(extract_text_from_tree(item))
                elif isinstance(tree_data, dict):
                    if 'content' in tree_data:
                        text_parts.append(tree_data['content'])
                    if 'children' in tree_data:
                        text_parts.extend(extract_text_from_tree(tree_data['children']))
                return text_parts
            
            content_parts = extract_text_from_tree(page_data.get('content', []))
            full_content = '\n\n'.join(content_parts)
            
            return {
                'page_id': page_id,
                'title': f"Notion Page {page_id}",
                'content': full_content,
                'url': f"https://notion.so/{page_id}",
                'cached_at': page_data.get('cached_at', 'unknown')
            }
            
        except Exception as e:
            print(f"❌ Error getting page {page_id}: {str(e)}")
            return None

    def load_to_vector_db(self, collection_name: str = "notion_documents") -> Dict[str, Any]:
        """
        Load all cached pages into the vector database.
        
        Returns:
            Dictionary with statistics
        """
        print(f"📚 Loading all cached pages into vector database...")
        start_time = time.time()
        
        self.vector_store = ChromaDBManager(self.config)
        
        collection = self.vector_store.get_or_create_collection(
            name=collection_name,
            metadata={
                "source": "notion_rag_complete",
                "created_by": "notion_rag_complete",
                "version": "0.1.0"
            }
        )
        
        # Get all cached pages
        try:
            from turbo_notion_fetch import cache
            cached_pages = cache.list_cached_pages()
            all_page_ids = [page['page_id'] for page in cached_pages]
            print(f"  📋 Found {len(all_page_ids)} cached pages")
        except ImportError:
            print("  ❌ Could not access cache")
            return {"success": False}
        
        successful_pages = 0
        total_chunks = 0
        total_chars = 0
        
        for i, page_id in enumerate(all_page_ids, 1):
            print(f"  📄 Processing {i}/{len(all_page_ids)}: {page_id}")
            
            page_data = self.get_cached_page_content(page_id)
            if not page_data or not page_data['content'].strip():
                continue
            
            content_length = len(page_data['content'])
            total_chars += content_length
            
            chunks = chunk_text(page_data['content'], chunk_size=512, overlap=50)
            
            doc_chunks = []
            for j, chunk in enumerate(chunks):
                doc_chunk = DocumentChunk(
                    id=f"{page_id}_chunk_{j}",
                    content=chunk,
                    metadata={
                        "source_id": page_id,
                        "source_type": "notion_page",
                        "title": page_data['title'],
                        "url": page_data['url'],
                        "chunk_index": j,
                        "total_chunks": len(chunks),
                        "cached_at": page_data['cached_at']
                    }
                )
                doc_chunks.append(doc_chunk)
            
            success = self.vector_store.add_documents(collection_name, doc_chunks)
            if success:
                successful_pages += 1
                total_chunks += len(doc_chunks)
        
        load_time = time.time() - start_time
        final_count = collection.count()
        
        print(f"✅ Loaded {successful_pages}/{len(all_page_ids)} pages")
        print(f"📄 Total chunks: {total_chunks}")
        print(f"📚 Total documents: {final_count}")
        print(f"📈 Characters: {total_chars:,}")
        print(f"⏱️  Time: {load_time:.2f}s")
        
        return {
            "success": successful_pages > 0,
            "load_time": load_time,
            "successful_pages": successful_pages,
            "total_pages": len(all_page_ids),
            "total_chunks": total_chunks,
            "total_chars": total_chars,
            "final_document_count": final_count
        }

    def initialize_search_components(self):
        """Initialize components needed for searching."""
        try:
            logger.info("Initializing search components...")
            
            # Initialize embeddings
            logger.info("Loading sentence transformer embeddings...")
            self.embeddings = create_embedding_generator(self.config)
            
            # Initialize Gemini client
            logger.info("Initializing Gemini client...")
            self.prompt_library = get_prompt_library()
            self.gemini_client = GeminiClient(self.config, prompt_library=self.prompt_library)
            
            # Initialize cost tracker
            logger.info("Initializing cost tracker...")
            self.cost_tracker = get_cost_tracker()
            
            # Initialize summarizer
            logger.info("Loading summarizer...")
            self.summarizer = get_summarizer()
            
            # Test connections
            logger.info("Testing connections...")
            if not self.gemini_client.test_connection():
                raise Exception("Failed to connect to Gemini API")
            
            logger.info("✅ Search components initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize search components: {e}")
            return False

    def search_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents in the vector store."""
        try:
            logger.info(f"Searching for documents with query: '{query}'")
            results = self.vector_store.search_documents(
                collection_name=self.config.chroma.collection_name,
                query=query,
                n_results=top_k
            )
            
            if not results or not results.get('documents'):
                logger.warning("No search results found")
                return []
            
            # Convert to the format expected by Gemini
            documents = []
            for i in range(len(results['documents'][0])):
                content = results['documents'][0][i]
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                distance = results['distances'][0][i] if results['distances'] else 0.0
                
                doc = {
                    "content": content,
                    "metadata": {
                        "title": metadata.get("title", f"Document {i+1}"),
                        "source_id": metadata.get("source_id", "Unknown"),
                        "page_id": metadata.get("page_id", ""),
                        "score": 1.0 - distance  # Convert distance to similarity score
                    }
                }
                documents.append(doc)
            
            logger.info(f"Found {len(documents)} relevant documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []

    def query_rag(self, query: str, prompt_template: str = "rag_qa", top_k: int = 5) -> Optional[str]:
        """Perform a RAG query using Gemini with retrieved context."""
        try:
            # Search for relevant documents
            documents = self.search_documents(query, top_k)
            
            if not documents:
                logger.warning("No relevant documents found")
                return "I couldn't find any relevant information in the database to answer your question."
            
            # Generate response using Gemini
            logger.info(f"Generating response with Gemini using template: {prompt_template}")
            response = self.gemini_client.rag_completion(
                query=query,
                context_documents=documents,
                prompt_template=prompt_template,
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.content
            
        except Exception as e:
            logger.error(f"Error in RAG query: {e}")
            return f"Sorry, I encountered an error while processing your query: {str(e)}"

    def interactive_search(self, collection_name: str = "notion_documents"):
        """Start interactive search mode with Gemini."""
        print("\n" + "="*60)
        print("🔍 Notion RAG Interactive Search")
        print("="*60)
        print("Ask questions about your Notion database!")
        print("Type 'quit' or 'exit' to end the session.")
        print("Type 'help' for available commands.")
        print("-"*60)
        
        while True:
            try:
                # Get user input
                query = input("\n🤔 Your question: ").strip()
                
                if not query:
                    continue
                
                # Handle special commands
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if query.lower() == 'help':
                    self.show_help()
                    continue
                
                if query.lower() == 'stats':
                    self.show_stats()
                    continue
                
                if query.lower() == 'costs':
                    self.show_cost_summary()
                    continue
                
                if query.lower() == 'templates':
                    self.show_available_templates()
                    continue
                
                # Check for special prompt commands
                prompt_template = "rag_qa"  # Default template
                if query.lower().startswith('summarize:'):
                    prompt_template = "rag_summary"
                    query = query[10:].strip()  # Remove 'summarize:' prefix
                    print(f"📝 Using RAG summarization template...")
                elif query.lower().startswith('analyze:'):
                    prompt_template = "rag_analysis"
                    query = query[8:].strip()  # Remove 'analyze:' prefix
                    print(f"🔍 Using RAG analysis template...")
                elif query.lower().startswith('extract:'):
                    prompt_template = "rag_extraction"
                    query = query[8:].strip()  # Remove 'extract:' prefix
                    print(f"📋 Using RAG extraction template...")
                elif query.lower().startswith('bullet:'):
                    prompt_template = "rag_summary"  # Use RAG summary for bullet points too
                    query = query[7:].strip()  # Remove 'bullet:' prefix
                    print(f"• Using RAG summary template...")
                
                # Process the query
                print("\n🔍 Searching...")
                response = self.query_rag(query, prompt_template=prompt_template)
                
                if response:
                    print(f"\n🤖 Answer:\n{response}")
                else:
                    print("\n❌ No response generated.")
                
                print("-" * 60)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in interactive session: {e}")
                print(f"\n❌ Error: {e}")

    def show_cost_summary(self):
        """Show cost summary."""
        try:
            if not self.cost_tracker:
                print("❌ Cost tracker not initialized")
                return
            
            self.cost_tracker.print_cost_summary()
                
        except Exception as e:
            logger.error(f"Error getting cost summary: {e}")
            print(f"❌ Error getting cost summary: {e}")

    def show_help(self):
        """Show help information."""
        help_text = """
📚 Available Commands:
- help: Show this help message
- stats: Show database statistics
- costs: Show cost summary and usage
- templates: Show available prompt templates
- quit/exit/q: Exit the program

🎯 Special Query Commands:
- summarize: [question] - Use summarization template
- analyze: [question] - Use content analysis template
- extract: [question] - Use key points extraction template
- bullet: [question] - Use bullet-point summary template

💡 Tips:
- Ask specific questions for better results
- Use natural language
- Try different phrasings if you don't get the expected answer
- The system searches through your entire Notion database
- Cost tracking is automatically enabled for all queries
- Use special commands for different types of responses
        """
        print(help_text)

    def show_available_templates(self):
        """Show available prompt templates."""
        if not self.prompt_library:
            print("❌ Prompt library not initialized")
            return
        
        templates = self.prompt_library.list_templates()
        print("\n📚 Available Prompt Templates:")
        print("=" * 50)
        
        for template in templates:
            print(f"📝 {template.name}")
            print(f"   Type: {template.task_type.value}")
            print(f"   Description: {template.description}")
            print(f"   Variables: {', '.join(template.variables)}")
            print()
        
        print("💡 Usage Examples:")
        print("  • Regular query: 'What is cybersecurity?'")
        print("  • Summarize: 'summarize: What is cybersecurity?'")
        print("  • Analyze: 'analyze: What is cybersecurity?'")
        print("  • Extract: 'extract: What is cybersecurity?'")
        print("  • Bullet: 'bullet: What is cybersecurity?'")
        print("\n🎯 RAG-Specific Templates:")
        print("  • rag_qa: Standard question answering")
        print("  • rag_summary: Detailed summarization")
        print("  • rag_analysis: Comprehensive analysis")
        print("  • rag_extraction: Key points extraction")

    def show_stats(self):
        """Show database statistics."""
        try:
            if not self.vector_store:
                print("❌ Vector store not initialized")
                return
            
            # Get collection info
            collection_info = self.vector_store.get_collection_info(self.config.chroma.collection_name)
            if collection_info:
                print(f"\n📊 Database Statistics:")
                print(f"   Total documents: {collection_info['count']}")
                print(f"   Collection name: {self.config.chroma.collection_name}")
                print(f"   Database path: {self.config.get_chroma_db_path()}")
            else:
                print("❌ Could not retrieve collection information")
                
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            print(f"❌ Error getting statistics: {e}")

    def cleanup(self):
        """Clean up resources."""
        if self.vector_store:
            self.vector_store.close()


def main():
    """Main workflow function."""
    print("🎯 Complete Notion RAG System")
    print("=" * 50)
    
    # Check environment variables
    notion_api_key = os.getenv("NOTION_API_KEY")
    notion_home_page_id = os.getenv("NOTION_HOME_PAGE_ID")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    if not notion_api_key or not notion_home_page_id:
        print("❌ Please set NOTION_API_KEY and NOTION_HOME_PAGE_ID in your .env file")
        return 1
    
    if not gemini_api_key:
        print("❌ Please set GEMINI_API_KEY in your .env file")
        return 1
    
    # Load configuration
    config = Config()
    
    # Get page ID to process
    if len(sys.argv) > 1:
        page_id = sys.argv[1]
    else:
        page_id = notion_home_page_id
    
    print(f"📄 Processing page: {page_id}")
    
    # Initialize the complete RAG system
    rag_system = NotionRAGComplete(config)
    
    try:
        # Step 1: Fetch from Notion
        if not rag_system.run_notion_fetch(page_id):
            print("❌ Failed to fetch from Notion")
            return 1
        
        # Step 2: Load into vector database
        load_result = rag_system.load_to_vector_db()
        if not load_result["success"]:
            print("❌ Failed to load into vector database")
            return 1
        
        # Step 3: Initialize search components
        if not rag_system.initialize_search_components():
            print("❌ Failed to initialize search components")
            return 1
        
        # Step 4: Start interactive search or single query
        if len(sys.argv) > 2:
            # Single search query
            query = " ".join(sys.argv[2:])
            print(f"\n🔍 Searching for: '{query}'")
            response = rag_system.query_rag(query)
            if response:
                print(f"\n🤖 Answer:\n{response}")
            else:
                print("\n❌ No response generated.")
        else:
            # Interactive mode
            rag_system.interactive_search()
        
        print("\n🎉 Complete workflow finished!")
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"❌ Fatal error: {e}")
        return 1
    finally:
        rag_system.cleanup()


if __name__ == "__main__":
    sys.exit(main()) 