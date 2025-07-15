#!/usr/bin/env python3
"""
Quick Chat with Notion Database
Lightweight script to start chatting with Gemini about your Notion database.
Assumes data is already fetched and loaded in ChromaDB.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add the project root to Python path
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


class NotionChat:
    """Quick chat interface for Notion database with Gemini."""
    
    def __init__(self, config: Config):
        """Initialize the chat system."""
        self.config = config
        self.vector_store = None
        self.gemini_client = None
        self.embeddings = None
        self.cost_tracker = None
        self.prompt_library = None
        self.summarizer = None
        
    def quick_initialize(self) -> bool:
        """Quickly initialize components for chatting."""
        try:
            print("🚀 Quick loading...")
            
            # Initialize vector store (fast - just connects to existing DB)
            print("📚 Connecting to ChromaDB...")
            self.vector_store = ChromaDBManager(self.config)
            
            # Check if collection exists
            collection_info = self.vector_store.get_collection_info(self.config.chroma.collection_name)
            if not collection_info:
                print(f"❌ No collection '{self.config.chroma.collection_name}' found")
                print("💡 Please run 'python notion_rag_complete.py' first to set up your database")
                return False
            
            print(f"✅ Found {collection_info['count']} documents in database")
            
            # Initialize embeddings (fast - loads existing model)
            print("🧠 Loading embeddings...")
            self.embeddings = create_embedding_generator(self.config)
            
            # Initialize Gemini client (fast - just API connection)
            print("🤖 Connecting to Gemini...")
            self.prompt_library = get_prompt_library()
            self.gemini_client = GeminiClient(self.config, prompt_library=self.prompt_library)
            
            # Initialize cost tracker
            print("💰 Loading cost tracker...")
            self.cost_tracker = get_cost_tracker()
            
            # Test Gemini connection
            if not self.gemini_client.test_connection():
                raise Exception("Failed to connect to Gemini API")
            
            print("✅ Ready to chat!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize: {e}")
            print(f"❌ Error: {e}")
            return False

    def search_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents in the vector store."""
        try:
            results = self.vector_store.search_documents(
                collection_name=self.config.chroma.collection_name,
                query=query,
                n_results=top_k
            )
            
            if not results or not results.get('documents'):
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
                        "score": 1.0 - distance
                    }
                }
                documents.append(doc)
            
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
                return "I couldn't find any relevant information in your Notion database to answer your question."
            
            # Generate response using Gemini
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

    def chat(self):
        """Start the chat interface."""
        print("\n" + "="*60)
        print("💬 Chat with Your Notion Database")
        print("="*60)
        print("Ask questions about your Notion content!")
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
                    query = query[10:].strip()
                    print(f"📝 Using summarization template...")
                elif query.lower().startswith('analyze:'):
                    prompt_template = "rag_analysis"
                    query = query[8:].strip()
                    print(f"🔍 Using analysis template...")
                elif query.lower().startswith('extract:'):
                    prompt_template = "rag_extraction"
                    query = query[8:].strip()
                    print(f"📋 Using extraction template...")
                elif query.lower().startswith('bullet:'):
                    prompt_template = "rag_summary"
                    query = query[7:].strip()
                    print(f"• Using bullet-point template...")
                
                # Process the query
                print("\n🔍 Searching your database...")
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
                logger.error(f"Error in chat session: {e}")
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
- The system searches through your entire Notion database
- Cost tracking is automatically enabled
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
    """Main entry point."""
    print("💬 Quick Chat with Notion Database")
    print("=" * 50)
    
    # Check environment variables
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    if not gemini_api_key:
        print("❌ Please set GEMINI_API_KEY in your .env file")
        return 1
    
    # Load configuration
    config = Config()
    
    # Initialize chat system
    chat_system = NotionChat(config)
    
    try:
        # Quick initialization
        if not chat_system.quick_initialize():
            print("\n💡 To set up your database, run:")
            print("   python notion_rag_complete.py")
            return 1
        
        # Start chat
        chat_system.chat()
        
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"❌ Fatal error: {e}")
        return 1
    finally:
        chat_system.cleanup()


if __name__ == "__main__":
    sys.exit(main()) 