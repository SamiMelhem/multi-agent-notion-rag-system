#!/usr/bin/env python3
"""
Interactive RAG Query System for Notion Database
Uses Gemini 2.5 Flash-Lite Preview with ChromaDB vector store.
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from notion_rag.config import Config
from notion_rag.gemini_client import GeminiClient, GeminiMessage
from notion_rag.vector_store import ChromaDBManager
from notion_rag.embeddings import create_embedding_generator
from notion_rag.cost_tracker import get_cost_tracker
from notion_rag.prompt_utils import get_prompt_library, get_summarizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NotionRAGQuery:
    """Interactive RAG query system for Notion database."""
    
    def __init__(self, config: Config):
        """Initialize the RAG query system."""
        self.config = config
        self.vector_store = None
        self.gemini_client = None
        self.embeddings = None
        self.cost_tracker = None
        self.prompt_library = None
        self.summarizer = None
        
    def initialize(self):
        """Initialize all components."""
        try:
            logger.info("Initializing RAG query system...")
            
            # Initialize embeddings
            logger.info("Loading sentence transformer embeddings...")
            self.embeddings = create_embedding_generator(self.config)
            
            # Initialize vector store
            logger.info("Connecting to ChromaDB...")
            self.vector_store = ChromaDBManager(self.config)
            
            # Initialize Gemini client
            logger.info("Initializing Gemini client...")
            self.gemini_client = GeminiClient(self.config, prompt_library=self.prompt_library)
            
            # Initialize cost tracker
            logger.info("Initializing cost tracker...")
            self.cost_tracker = get_cost_tracker()
            
            # Initialize prompt library and summarizer
            logger.info("Loading prompt library and summarizer...")
            self.prompt_library = get_prompt_library()
            self.summarizer = get_summarizer()
            
            # Test connections
            logger.info("Testing connections...")
            if not self.gemini_client.test_connection():
                raise Exception("Failed to connect to Gemini API")
            
            logger.info("✅ RAG query system initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize RAG query system: {e}")
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
    
    def interactive_query(self):
        """Start interactive query session."""
        print("\n" + "="*60)
        print("🔍 Notion RAG Query System")
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


def main():
    """Main entry point."""
    try:
        # Load configuration
        config = Config()
        
        # Check for required environment variables
        if not config.get_gemini_api_key():
            print("❌ GEMINI_API_KEY environment variable is required")
            print("Please set it with: export GEMINI_API_KEY='your-api-key'")
            return 1
        
        # Initialize RAG query system
        rag_system = NotionRAGQuery(config)
        
        if not rag_system.initialize():
            print("❌ Failed to initialize RAG query system")
            return 1
        
        # Start interactive session
        rag_system.interactive_query()
        
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 