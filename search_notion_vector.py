#!/usr/bin/env python3
"""
Interactive search script for the Notion vector database.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from notion_rag.config import Config
from notion_rag.vector_store import ChromaDBManager

def search_notion_content(query: str, collection_name: str = "notion_documents", n_results: int = 5):
    """
    Search for content in the Notion vector database.
    
    Args:
        query: Search query
        collection_name: Name of the collection to search
        n_results: Number of results to return
    """
    config = Config()
    vector_store = ChromaDBManager(config)
    
    print(f"🔍 Searching for: '{query}'")
    print(f"📚 Collection: {collection_name}")
    print(f"📊 Results: {n_results}")
    print("-" * 60)
    
    try:
        results = vector_store.search_documents(
            collection_name=collection_name,
            query=query,
            n_results=n_results
        )
        
        if results and results['ids']:
            print(f"✅ Found {len(results['ids'][0])} relevant documents:\n")
            
            for i in range(len(results['ids'][0])):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                content = results['documents'][0][i] if results['documents'] else ""
                distance = results['distances'][0][i] if results['distances'] else 0.0
                
                title = metadata.get('title', 'Unknown')
                page_id = metadata.get('source_id', 'Unknown')
                url = metadata.get('url', 'No URL')
                chunk_info = f"{metadata.get('chunk_index', 'N/A')}/{metadata.get('total_chunks', 'N/A')}"
                score = 1.0 - distance
                
                print(f"📄 Result {i+1}: {title}")
                print(f"   🔗 Page ID: {page_id}")
                print(f"   🌐 URL: {url}")
                print(f"   📊 Relevance Score: {score:.3f}")
                print(f"   🔪 Chunk: {chunk_info}")
                print(f"   📝 Content Preview:")
                print(f"      {content[:200]}...")
                print()
        else:
            print("❌ No results found")
            
    except Exception as e:
        print(f"❌ Error during search: {str(e)}")
    finally:
        vector_store.close()

def interactive_search(collection_name: str = "notion_documents"):
    """
    Start interactive search mode.
    
    Args:
        collection_name: Name of the collection to search
    """
    print("🔍 Interactive Notion Vector Search")
    print("=" * 50)
    print("Type your search queries. Type 'quit' or 'exit' to stop.")
    print("Type 'help' for search tips.")
    print()
    
    while True:
        try:
            query = input("🔍 Search query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif query.lower() == 'help':
                print("\n💡 Search Tips:")
                print("  • Use natural language: 'How to plan a project?'")
                print("  • Ask specific questions: 'What tools are mentioned?'")
                print("  • Search for concepts: 'team collaboration strategies'")
                print("  • Look for processes: 'data analysis workflow'")
                print("  • Find best practices: 'communication guidelines'")
                print()
                continue
            elif not query:
                continue
            
            search_notion_content(query, collection_name)
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def main():
    """Main function."""
    if len(sys.argv) > 1:
        # Single search query from command line
        query = " ".join(sys.argv[1:])
        search_notion_content(query)
    else:
        # Interactive mode
        interactive_search()

if __name__ == "__main__":
    main() 