#!/usr/bin/env python3
"""
Streamlined Notion RAG Workflow
Combines fetching, loading, and searching in one efficient script.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from notion_rag.config import Config
from notion_rag.vector_store import ChromaDBManager, DocumentChunk
from notion_rag.chunking import chunk_text

def run_notion_fetch(page_id: str) -> bool:
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

def get_cached_page_content(page_id: str) -> Dict[str, Any]:
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

def load_to_vector_db(collection_name: str = "notion_documents") -> Dict[str, Any]:
    """
    Load all cached pages into the vector database.
    
    Returns:
        Dictionary with statistics
    """
    print(f"📚 Loading all cached pages into vector database...")
    start_time = time.time()
    
    config = Config()
    vector_store = ChromaDBManager(config)
    
    collection = vector_store.get_or_create_collection(
        name=collection_name,
        metadata={
            "source": "notion_workflow",
            "created_by": "notion_rag_workflow",
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
        
        page_data = get_cached_page_content(page_id)
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
        
        success = vector_store.add_documents(collection_name, doc_chunks)
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
    
    vector_store.close()
    
    return {
        "success": successful_pages > 0,
        "load_time": load_time,
        "successful_pages": successful_pages,
        "total_pages": len(all_page_ids),
        "total_chunks": total_chunks,
        "total_chars": total_chars,
        "final_document_count": final_count
    }

def search_content(query: str, collection_name: str = "notion_documents", n_results: int = 5):
    """Search for content in the vector database."""
    print(f"🔍 Searching for: '{query}'")
    
    config = Config()
    vector_store = ChromaDBManager(config)
    
    try:
        results = vector_store.search_documents(
            collection_name=collection_name,
            query=query,
            n_results=n_results
        )
        
        if results and results['ids']:
            print(f"✅ Found {len(results['ids'][0])} results:\n")
            
            for i in range(len(results['ids'][0])):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                content = results['documents'][0][i] if results['documents'] else ""
                distance = results['distances'][0][i] if results['distances'] else 0.0
                
                title = metadata.get('title', 'Unknown')
                page_id = metadata.get('source_id', 'Unknown')
                score = 1.0 - distance
                
                print(f"📄 {i+1}. {title}")
                print(f"   🔗 Page ID: {page_id}")
                print(f"   📊 Score: {score:.3f}")
                print(f"   📝 Preview: {content[:150]}...")
                print()
        else:
            print("❌ No results found")
            
    except Exception as e:
        print(f"❌ Search error: {str(e)}")
    finally:
        vector_store.close()

def interactive_search(collection_name: str = "notion_documents"):
    """Start interactive search mode."""
    print("🔍 Interactive Search Mode")
    print("Type your queries. Type 'quit' to exit.")
    print()
    
    while True:
        try:
            query = input("🔍 Query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif not query:
                continue
            
            search_content(query, collection_name)
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def main():
    """Main workflow function."""
    print("🎯 Notion RAG Workflow")
    print("=" * 40)
    
    # Check environment variables
    notion_api_key = os.getenv("NOTION_API_KEY")
    notion_home_page_id = os.getenv("NOTION_HOME_PAGE_ID")
    
    if not notion_api_key or not notion_home_page_id:
        print("❌ Please set NOTION_API_KEY and NOTION_HOME_PAGE_ID in your .env file")
        return
    
    # Get page ID to process
    if len(sys.argv) > 1:
        page_id = sys.argv[1]
    else:
        page_id = notion_home_page_id
    
    print(f"📄 Processing page: {page_id}")
    
    # Step 1: Fetch from Notion
    if not run_notion_fetch(page_id):
        print("❌ Failed to fetch from Notion")
        return
    
    # Step 2: Load into vector database
    load_result = load_to_vector_db()
    if not load_result["success"]:
        print("❌ Failed to load into vector database")
        return
    
    # Step 3: Search or interactive mode
    if len(sys.argv) > 2:
        # Single search query
        query = " ".join(sys.argv[2:])
        search_content(query)
    else:
        # Interactive mode
        interactive_search()
    
    print("\n🎉 Workflow completed!")

if __name__ == "__main__":
    main() 