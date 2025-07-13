#!/usr/bin/env python3
"""
Performance test script to measure Notion fetch + vector database load time.
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from notion_rag.config import Config
from notion_rag.vector_store import ChromaDBManager, DocumentChunk
from notion_rag.chunking import chunk_text

def measure_turbo_fetch(page_id: str) -> Dict[str, Any]:
    """
    Measure the time to fetch a page using turbo_notion_fetch.py.
    
    Args:
        page_id: Notion page ID to fetch
        
    Returns:
        Dictionary with timing and success information
    """
    print(f"🚀 Fetching page {page_id} and all children from Notion API...")
    start_time = time.time()
    
    try:
        # Run the turbo fetch script with UTF-8 encoding (this automatically fetches children)
        result = subprocess.run([
            sys.executable, "turbo_notion_fetch.py", page_id
        ], capture_output=True, text=True, encoding='utf-8', errors='replace', cwd=project_root)
        
        fetch_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Successfully fetched page {page_id} and children in {fetch_time:.2f} seconds")
            return {
                "success": True,
                "fetch_time": fetch_time,
                "page_id": page_id
            }
        else:
            print(f"❌ Failed to fetch page {page_id}")
            print(f"Error: {result.stderr}")
            return {
                "success": False,
                "fetch_time": fetch_time,
                "error": result.stderr,
                "page_id": page_id
            }
            
    except Exception as e:
        fetch_time = time.time() - start_time
        print(f"❌ Error running turbo fetch: {str(e)}")
        return {
            "success": False,
            "fetch_time": fetch_time,
            "error": str(e),
            "page_id": page_id
        }

def get_cached_page_content(page_id: str) -> Dict[str, Any]:
    """
    Get content from a cached Notion page.
    
    Args:
        page_id: Notion page ID
        
    Returns:
        Dictionary with page content and metadata
    """
    try:
        # Import the turbo fetch functionality
        from turbo_notion_fetch import get_page_tree_for_context
        
        # Get the page content
        page_data = get_page_tree_for_context(page_id, max_depth=3)
        
        if "error" in page_data:
            print(f"❌ Error getting page {page_id}: {page_data['error']}")
            return None
        
        # Extract text content from the tree
        def extract_text_from_tree(tree_data):
            """Recursively extract text from the tree structure."""
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
        
    except ImportError:
        print("❌ turbo_notion_fetch module not available")
        return None
    except Exception as e:
        print(f"❌ Error getting page {page_id}: {str(e)}")
        return None

def measure_vector_db_load(page_ids: List[str], collection_name: str = "notion_documents") -> Dict[str, Any]:
    """
    Measure the time to load ALL cached pages into the vector database.
    
    Args:
        page_ids: List of Notion page IDs (used for reference, but loads all cached pages)
        collection_name: Name of the collection to use
        
    Returns:
        Dictionary with timing and statistics
    """
    print(f"\n📚 Loading ALL cached pages into vector database...")
    start_time = time.time()
    
    # Initialize configuration and vector store
    config = Config()
    vector_store = ChromaDBManager(config)
    
    # Get or create collection
    collection = vector_store.get_or_create_collection(
        name=collection_name,
        metadata={
            "source": "notion_turbo_fetch",
            "created_by": "performance_test",
            "version": "0.1.0"
        }
    )
    
    initial_count = collection.count()
    
    # Get all cached pages
    try:
        from turbo_notion_fetch import cache
        cached_pages = cache.list_cached_pages()
        all_page_ids = [page['page_id'] for page in cached_pages]
        print(f"  📋 Found {len(all_page_ids)} cached pages to load")
    except ImportError:
        print("  ❌ Could not access cache, using provided page IDs")
        all_page_ids = page_ids
    
    # Load each page
    successful_pages = 0
    total_chunks = 0
    total_chars = 0
    
    for i, page_id in enumerate(all_page_ids, 1):
        print(f"  📄 Processing page {i}/{len(all_page_ids)}: {page_id}")
        
        # Get cached page content
        page_data = get_cached_page_content(page_id)
        if not page_data:
            print(f"    ❌ Failed to get page {page_id} from cache")
            continue
        
        # Check if content is meaningful
        if not page_data['content'].strip():
            print(f"    ⚠️  Page {page_id} has no content")
            continue
        
        content_length = len(page_data['content'])
        total_chars += content_length
        print(f"    📏 Content length: {content_length} characters")
        
        # Chunk the content
        chunks = chunk_text(page_data['content'], chunk_size=512, overlap=50)
        print(f"    🔪 Created {len(chunks)} chunks")
        
        # Create document chunks
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
        
        # Add to vector store
        success = vector_store.add_documents(collection_name, doc_chunks)
        if success:
            print(f"    ✅ Added {len(doc_chunks)} chunks successfully")
            successful_pages += 1
            total_chunks += len(doc_chunks)
        else:
            print(f"    ❌ Failed to add chunks for page {page_id}")
    
    load_time = time.time() - start_time
    final_count = collection.count()
    new_documents = final_count - initial_count
    
    # Show final statistics
    print(f"\n📊 Vector Database Loading Complete!")
    print(f"✅ Successfully loaded {successful_pages}/{len(all_page_ids)} pages")
    print(f"📄 Total chunks added: {total_chunks}")
    print(f"📚 Total documents in collection: {final_count}")
    print(f"📈 Total characters processed: {total_chars:,}")
    
    vector_store.close()
    
    return {
        "success": successful_pages > 0,
        "load_time": load_time,
        "successful_pages": successful_pages,
        "total_pages": len(page_ids),
        "total_chunks": total_chunks,
        "total_chars": total_chars,
        "new_documents": new_documents,
        "final_document_count": final_count
    }

def measure_search_performance(query: str, collection_name: str = "notion_documents") -> Dict[str, Any]:
    """
    Measure the time to perform a search query.
    
    Args:
        query: Search query to test
        collection_name: Name of the collection to search
        
    Returns:
        Dictionary with timing and search results
    """
    print(f"\n🔍 Testing search performance...")
    print(f"Query: '{query}'")
    
    start_time = time.time()
    
    config = Config()
    vector_store = ChromaDBManager(config)
    
    try:
        results = vector_store.search_documents(
            collection_name=collection_name,
            query=query,
            n_results=5
        )
        
        search_time = time.time() - start_time
        
        if results and results['ids']:
            result_count = len(results['ids'][0])
            print(f"✅ Found {result_count} results in {search_time:.3f} seconds")
            
            # Show top result
            if result_count > 0:
                metadata = results['metadatas'][0][0] if results['metadatas'] else {}
                content = results['documents'][0][0] if results['documents'] else ""
                distance = results['distances'][0][0] if results['distances'] else 0.0
                
                title = metadata.get('title', 'Unknown')
                page_id = metadata.get('source_id', 'Unknown')
                score = 1.0 - distance
                
                print(f"📄 Top result: {title}")
                print(f"   🔗 Page ID: {page_id}")
                print(f"   📊 Relevance Score: {score:.3f}")
                print(f"   📝 Preview: {content[:100]}...")
            
            return {
                "success": True,
                "search_time": search_time,
                "result_count": result_count,
                "query": query
            }
        else:
            print(f"❌ No results found in {search_time:.3f} seconds")
            return {
                "success": False,
                "search_time": search_time,
                "result_count": 0,
                "query": query
            }
            
    except Exception as e:
        search_time = time.time() - start_time
        print(f"❌ Search error: {str(e)}")
        return {
            "success": False,
            "search_time": search_time,
            "error": str(e),
            "query": query
        }
    finally:
        vector_store.close()

def main():
    """Main performance test function."""
    print("🎯 Notion RAG Performance Test - Fresh Start Scenario")
    print("=" * 60)
    
    # Check if environment variables are set
    notion_api_key = os.getenv("NOTION_API_KEY")
    notion_home_page_id = os.getenv("NOTION_HOME_PAGE_ID")
    
    if not notion_api_key:
        print("❌ NOTION_API_KEY environment variable not set")
        return
    
    if not notion_home_page_id:
        print("❌ NOTION_HOME_PAGE_ID environment variable not set")
        return
    
    print("✅ Environment variables configured")
    
    # Get page IDs to test
    if len(sys.argv) > 1:
        page_ids = sys.argv[1:]
    else:
        page_ids = [notion_home_page_id]
    
    print(f"📄 Testing with {len(page_ids)} pages: {page_ids}")
    
    # Step 1: Measure Notion API fetch time
    print("\n" + "="*60)
    print("STEP 1: FETCHING FROM NOTION API")
    print("="*60)
    
    fetch_results = []
    total_fetch_time = 0
    
    for page_id in page_ids:
        result = measure_turbo_fetch(page_id)
        fetch_results.append(result)
        if result["success"]:
            total_fetch_time += result["fetch_time"]
    
    successful_fetches = sum(1 for r in fetch_results if r["success"])
    
    if successful_fetches == 0:
        print("❌ No pages were successfully fetched")
        return
    
    print(f"\n📊 Fetch Summary:")
    print(f"✅ Successfully fetched {successful_fetches}/{len(page_ids)} pages")
    print(f"⏱️  Total fetch time: {total_fetch_time:.2f} seconds")
    print(f"📈 Average fetch time per page: {total_fetch_time/successful_fetches:.2f} seconds")
    
    # Step 2: Measure vector database load time
    print("\n" + "="*60)
    print("STEP 2: LOADING INTO VECTOR DATABASE")
    print("="*60)
    
    load_result = measure_vector_db_load(page_ids)
    
    if not load_result["success"]:
        print("❌ Failed to load pages into vector database")
        return
    
    print(f"\n📊 Load Summary:")
    print(f"⏱️  Total load time: {load_result['load_time']:.2f} seconds")
    print(f"📄 Chunks processed: {load_result['total_chunks']}")
    print(f"📈 Characters processed: {load_result['total_chars']:,}")
    print(f"🚀 Processing rate: {load_result['total_chars']/load_result['load_time']:,.0f} chars/sec")
    print(f"📚 Pages loaded: {load_result['successful_pages']} (including all child pages)")
    
    # Step 3: Measure search performance
    print("\n" + "="*60)
    print("STEP 3: TESTING SEARCH PERFORMANCE")
    print("="*60)
    
    # Test multiple search queries
    test_queries = [
        "What is this page about?",
        "cybersecurity security risks",
        "main topics discussed",
        "key concepts and ideas"
    ]
    
    search_results = []
    total_search_time = 0
    
    for query in test_queries:
        result = measure_search_performance(query)
        search_results.append(result)
        if result["success"]:
            total_search_time += result["search_time"]
    
    successful_searches = sum(1 for r in search_results if r["success"])
    
    print(f"\n📊 Search Summary:")
    print(f"✅ Successful searches: {successful_searches}/{len(test_queries)}")
    print(f"⏱️  Total search time: {total_search_time:.3f} seconds")
    if successful_searches > 0:
        print(f"📈 Average search time: {total_search_time/successful_searches:.3f} seconds")
    
    # Final Performance Summary
    print("\n" + "="*60)
    print("🎯 FINAL PERFORMANCE SUMMARY")
    print("="*60)
    
    total_time = total_fetch_time + load_result['load_time']
    
    print(f"📊 Overall Performance:")
    print(f"⏱️  Total time (fetch + load): {total_time:.2f} seconds")
    print(f"🚀 Notion API fetch: {total_fetch_time:.2f}s ({total_fetch_time/total_time*100:.1f}%)")
    print(f"📚 Vector DB load: {load_result['load_time']:.2f}s ({load_result['load_time']/total_time*100:.1f}%)")
    print(f"🔍 Search time: {total_search_time:.3f}s (average per query)")
    
    print(f"\n📈 Throughput Metrics:")
    print(f"📄 Pages processed: {load_result['successful_pages']}")
    print(f"🔪 Chunks created: {load_result['total_chunks']}")
    print(f"📝 Characters processed: {load_result['total_chars']:,}")
    print(f"🚀 Processing rate: {load_result['total_chars']/total_time:,.0f} chars/sec")
    
    print(f"\n💡 Performance Insights:")
    if total_fetch_time > load_result['load_time']:
        print(f"  • Notion API fetch is the bottleneck ({total_fetch_time/total_time*100:.1f}% of total time)")
    else:
        print(f"  • Vector database processing is the bottleneck ({load_result['load_time']/total_time*100:.1f}% of total time)")
    
    print(f"  • Search queries are very fast ({total_search_time:.3f}s total for {len(test_queries)} queries)")
    print(f"  • Ready for real-time RAG applications!")

if __name__ == "__main__":
    main() 