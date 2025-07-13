from os import getenv
from time import time
from json import dump, load
from hashlib import md5
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
from dotenv import load_dotenv
from notion_client import AsyncClient
from dataclasses import dataclass
from datetime import datetime, timedelta
from asyncio import Lock as AsyncLock, Semaphore, gather, run
from threading import Lock as ThreadLock
import json
import asyncio
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

load_dotenv()

NOTION_API_KEY = getenv("NOTION_API_KEY")
HOME_PAGE_ID = getenv("NOTION_HOME_PAGE_ID")

# Turbo configuration - optimized for speed
MAX_CONCURRENT_REQUESTS = 30  # Conservative but effective
BATCH_SIZE = 15  # Optimal batch size
CACHE_DURATION_HOURS = 168  # 1 week cache
CACHE_DIR = Path("./cache")
ENABLE_CACHE = True

# Only fetch the most essential block types
TURBO_BLOCK_TYPES = {
    "paragraph", "heading_1", "heading_2", "heading_3", 
    "bulleted_list_item", "numbered_list_item", "child_page", "toggle"
}

@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    data: Dict[str, Any]
    timestamp: datetime
    page_id: str
    etag: Optional[str] = None

class TurboCache:
    """Ultra-fast cache with memory-first approach for multiple pages."""
    
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self._memory_cache = {}
        self._lock = ThreadLock()  # Use ThreadLock alias
        self._cache_index_file = cache_dir / "cache_index.json"
        self._load_cache_index()
    
    def _load_cache_index(self):
        """Load cache index to track all cached pages."""
        if self._cache_index_file.exists():
            try:
                with open(self._cache_index_file, 'r') as f:
                    self._cache_index = load(f)
            except Exception:
                self._cache_index = {}
        else:
            self._cache_index = {}
    
    def _save_cache_index(self):
        """Save cache index to track all cached pages."""
        try:
            with open(self._cache_index_file, 'w') as f:
                dump(self._cache_index, f, indent=2)
        except Exception as e:
            print(f"Cache index save error: {e}")
    
    def _get_cache_key(self, page_id: str) -> str:
        return md5(f"notion_page_{page_id}".encode()).hexdigest()
    
    def _get_cache_path(self, page_id: str) -> Path:
        cache_key = self._get_cache_key(page_id)
        return self.cache_dir / f"{cache_key}.json"
    
    def get(self, page_id: str) -> Optional[CacheEntry]:
        """Get cached data with memory-first approach."""
        if not ENABLE_CACHE:
            return None
        
        # Check memory cache first
        with self._lock:
            if page_id in self._memory_cache:
                entry = self._memory_cache[page_id]
                if datetime.now() - entry.timestamp < timedelta(hours=CACHE_DURATION_HOURS):
                    return entry
                else:
                    del self._memory_cache[page_id]
                    # Remove from index if expired
                    if page_id in self._cache_index:
                        del self._cache_index[page_id]
                        self._save_cache_index()
        
        # Check file cache
        cache_path = self._get_cache_path(page_id)
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r') as f:
                data = load(f)
            
            timestamp = datetime.fromisoformat(data['timestamp'])
            if datetime.now() - timestamp > timedelta(hours=CACHE_DURATION_HOURS):
                cache_path.unlink()
                # Remove from index if expired
                if page_id in self._cache_index:
                    del self._cache_index[page_id]
                    self._save_cache_index()
                return None
            
            entry = CacheEntry(
                data=data['data'],
                timestamp=timestamp,
                page_id=data['page_id'],
                etag=data.get('etag')
            )
            
            # Store in memory cache
            with self._lock:
                self._memory_cache[page_id] = entry
            
            return entry
        except Exception as e:
            print(f"Cache read error: {e}")
            return None
    
    def set(self, page_id: str, data: Dict[str, Any], etag: Optional[str] = None):
        """Cache data in both memory and file."""
        if not ENABLE_CACHE:
            return
        
        entry = CacheEntry(
            data=data,
            timestamp=datetime.now(),
            page_id=page_id,
            etag=etag
        )
        
        # Store in memory cache
        with self._lock:
            self._memory_cache[page_id] = entry
        
        # Store in file cache
        cache_path = self._get_cache_path(page_id)
        try:
            cache_entry = {
                'data': data,
                'timestamp': entry.timestamp.isoformat(),
                'page_id': page_id,
                'etag': etag
            }
            with open(cache_path, 'w') as f:
                dump(cache_entry, f, indent=2)
            
            # Update cache index
            self._cache_index[page_id] = {
                'timestamp': entry.timestamp.isoformat(),
                'cache_key': self._get_cache_key(page_id),
                'file_path': str(cache_path)
            }
            self._save_cache_index()
            
        except Exception as e:
            print(f"Cache write error: {e}")
    
    def list_cached_pages(self) -> List[Dict[str, Any]]:
        """List all cached pages with their metadata."""
        cached_pages = []
        
        with self._lock:
            for page_id, index_info in self._cache_index.items():
                try:
                    timestamp = datetime.fromisoformat(index_info['timestamp'])
                    if datetime.now() - timestamp <= timedelta(hours=CACHE_DURATION_HOURS):
                        cached_pages.append({
                            'page_id': page_id,
                            'timestamp': timestamp,
                            'age_hours': (datetime.now() - timestamp).total_seconds() / 3600,
                            'in_memory': page_id in self._memory_cache
                        })
                except Exception:
                    continue
        
        # Sort by timestamp (newest first)
        cached_pages.sort(key=lambda x: x['timestamp'], reverse=True)
        return cached_pages
    
    def clear_cache(self, page_id: Optional[str] = None):
        """Clear cache for specific page or all pages."""
        if page_id:
            # Clear specific page
            with self._lock:
                if page_id in self._memory_cache:
                    del self._memory_cache[page_id]
                if page_id in self._cache_index:
                    del self._cache_index[page_id]
            
            cache_path = self._get_cache_path(page_id)
            if cache_path.exists():
                cache_path.unlink()
            
            self._save_cache_index()
            print(f"🗑️ Cleared cache for page: {page_id}")
        else:
            # Clear all cache
            with self._lock:
                self._memory_cache.clear()
                self._cache_index.clear()
            
            # Remove all cache files
            for cache_file in self.cache_dir.glob("*.json"):
                if cache_file.name != "cache_index.json":
                    cache_file.unlink()
            
            self._save_cache_index()
            print("🗑️ Cleared all cache")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            memory_count = len(self._memory_cache)
            file_count = len(self._cache_index)
            
            total_size = 0
            for cache_file in self.cache_dir.glob("*.json"):
                if cache_file.name != "cache_index.json":
                    total_size += cache_file.stat().st_size
            
            return {
                'memory_entries': memory_count,
                'file_entries': file_count,
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'cache_dir': str(self.cache_dir)
            }

class TurboRateLimiter:
    """Intelligent rate limiter that adapts to API performance."""
    
    def __init__(self, max_requests: int = 30, time_window: float = 1.0):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self._lock = AsyncLock()  # Use AsyncLock alias
        self._semaphore = Semaphore(max_requests)
        self.error_count = 0
        self.success_count = 0
    
    async def acquire(self):
        """Intelligent rate limiting."""
        await self._semaphore.acquire()
        async with self._lock:
            now = time()
            self.requests = [req_time for req_time in self.requests 
                           if now - req_time < self.time_window]
            self.requests.append(now)
    
    def release(self):
        """Release semaphore."""
        self._semaphore.release()
    
    def record_success(self):
        self.success_count += 1
    
    def record_error(self):
        self.error_count += 1

# Global instances
cache = TurboCache()
rate_limiter = TurboRateLimiter(max_requests=MAX_CONCURRENT_REQUESTS, time_window=1.0)

def safe_print(s):
    print(s)

async def fetch_block_children_turbo(client, block_id: str, visited: Set[str]) -> List[Dict[str, Any]]:
    """Turbo-fast block fetching with minimal overhead."""
    if block_id in visited:
        return []
    
    visited.add(block_id)
    await rate_limiter.acquire()
    
    try:
        results = []
        has_more = True
        start_cursor = None
        
        while has_more:
            response = await client.blocks.children.list(
                block_id=block_id,
                start_cursor=start_cursor,
                page_size=100  # Maximum page size
            )
            
            # Ultra-fast filtering
            filtered_results = []
            for block in response["results"]:
                if block.get("type") in TURBO_BLOCK_TYPES:
                    filtered_results.append(block)
            results.extend(filtered_results)
            
            has_more = response["has_more"]
            start_cursor = response["next_cursor"]
        
        rate_limiter.record_success()
        return results
    except Exception as e:
        rate_limiter.record_error()
        print(f"Error fetching block {block_id}: {e}")
        return []
    finally:
        rate_limiter.release()

async def fetch_blocks_batch_turbo(client, block_ids: List[str], visited: Set[str]):
    """Turbo batch fetching with optimal concurrency."""
    unique_blocks = [bid for bid in block_ids if bid not in visited]
    
    if not unique_blocks:
        return []
    
    tasks = [fetch_block_children_turbo(client, block_id, visited) 
             for block_id in unique_blocks]
    return await gather(*tasks, return_exceptions=True)

async def fetch_all_blocks_turbo(root_id: str, _recursive: bool = False) -> Dict[str, Any]:
    """Turbo block fetching with maximum speed optimizations and recursive child page caching."""
    # Check cache first
    cached_data = cache.get(root_id)
    if cached_data:
        if not _recursive:
            print(f"Using cached data for page: {root_id}")
        return cached_data.data
    
    if not _recursive:
        print(f"Starting turbo fetch for page: {root_id}")
    start_time = time()
    
    # Create client
    client = AsyncClient(auth=NOTION_API_KEY)
    
    queue = [root_id]
    block_tree = {}
    visited = set()
    total_blocks_processed = 0
    child_page_ids = set()
    
    while queue:
        # Process optimal batch size
        batch = queue[:BATCH_SIZE]
        queue = queue[BATCH_SIZE:]
        
        # Fetch all blocks in the current batch
        batch_results = await fetch_blocks_batch_turbo(client, batch, visited)
        
        # Process results efficiently
        new_queue = []
        for parent_id, children in zip(batch, batch_results):
            if isinstance(children, Exception):
                continue
                
            block_tree[parent_id] = children
            total_blocks_processed += len(children)
            
            # Track child pages and add blocks with children to queue
            for child in children:
                if child.get("has_children") and child["id"] not in visited:
                    new_queue.append(child["id"])
                
                # If this is a child page, add it to pages to cache
                if child.get("type") == "child_page":
                    child_page_ids.add(child["id"])
        
        queue.extend(new_queue)
        
        # Progress update (only for top-level fetches)
        if not _recursive:
            elapsed = time() - start_time
            if elapsed > 0:
                rate = total_blocks_processed / elapsed
                print(f"Processed {total_blocks_processed} blocks at {rate:.0f} blocks/sec")
    
    # Cache this page
    cache.set(root_id, block_tree)
    
    # If this is the top-level fetch, recursively fetch all child pages in parallel
    if not _recursive and child_page_ids:
        print(f"Fetching {len(child_page_ids)} child pages in parallel...")
        child_tasks = [fetch_all_blocks_turbo(child_id, _recursive=True) for child_id in child_page_ids]
        await gather(*child_tasks, return_exceptions=True)
        print(f"All child pages cached!")
    
    if not _recursive:
        total_time = time() - start_time
        print(f"Completed in {total_time:.2f}s! Total blocks: {total_blocks_processed}")
    
    return block_tree

async def fetch_page_by_id(page_id: str) -> Dict[str, Any]:
    """Fetch a specific page by its ID."""
    if not NOTION_API_KEY:
        print("Please set NOTION_API_KEY in your .env file.")
        return {}
    
    print(f"Fetching page: {page_id}")
    return await fetch_all_blocks_turbo(page_id)

async def fetch_child_page(page_id: str) -> Dict[str, Any]:
    """Fetch a specific child page and cache it."""
    print(f"Fetching child page: {page_id}")
    return await fetch_all_blocks_turbo(page_id)

async def fetch_all_children(page_id: str) -> None:
    """Fetch all child pages of a specific page recursively."""
    print(f"Fetching all children of page: {page_id}")
    
    # First get the page to see its child references
    cached_data = cache.get(page_id)
    if not cached_data:
        print(f"Page {page_id} not found in cache. Please fetch it first.")
        return
    
    # Extract child page references
    child_pages = get_child_page_references(cached_data.data)
    if not child_pages:
        print(f"No child pages found for {page_id}")
        return
    
    print(f"Found {len(child_pages)} child pages to fetch...")
    
    # Fetch all child pages in parallel
    child_tasks = [fetch_all_blocks_turbo(child["id"], _recursive=True) for child in child_pages]
    await gather(*child_tasks, return_exceptions=True)
    
    print(f"All {len(child_pages)} child pages cached!")

def get_child_page_references(block_tree: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract child page references from a block tree."""
    child_pages = []
    
    def extract_from_blocks(blocks):
        for block in blocks:
            if block.get("type") == "child_page":
                page_title = block.get("child_page", {}).get("title", "Untitled Page")
                child_pages.append({
                    "id": block["id"],
                    "title": page_title
                })
            elif block.get("has_children"):
                # Recursively check nested blocks
                nested_blocks = block.get("children", [])
                if nested_blocks:
                    extract_from_blocks(nested_blocks)
    
    # Extract from all blocks in the tree
    for blocks in block_tree.values():
        extract_from_blocks(blocks)
    
    return child_pages

def render_markdown_turbo(block_tree, root_id, indent=0):
    """Turbo markdown rendering with minimal overhead."""
    blocks = block_tree.get(root_id, [])
    
    # Pre-process all blocks for maximum speed
    processed_blocks = []
    for block in blocks:
        block_type = block.get("type")
        if block_type not in TURBO_BLOCK_TYPES:
            continue
            
        if block_type == "child_page":
            # Handle child page blocks
            page_title = block.get("child_page", {}).get("title", "Untitled Page")
            processed_blocks.append((block_type, page_title, block))
        elif block_type == "toggle":
            # Handle toggle blocks
            rich_text = block.get("toggle", {}).get("rich_text", [])
            text = "".join([t.get("plain_text", "") for t in rich_text])
            processed_blocks.append((block_type, text, block))
        else:
            # Handle text blocks
            rich_text = block.get(block_type, {}).get("rich_text", [])
            text = "".join([t.get("plain_text", "") for t in rich_text])
            processed_blocks.append((block_type, text, block))
    
    # Render blocks efficiently and track character count
    total_chars = 0
    for block_type, text, block in processed_blocks:
        prefix = "  " * indent
        
        if block_type == "heading_1":
            output = f"{prefix}# {text}"
            safe_print(output)
            total_chars += len(output)
        elif block_type == "heading_2":
            output = f"{prefix}## {text}"
            safe_print(output)
            total_chars += len(output)
        elif block_type == "heading_3":
            output = f"{prefix}### {text}"
            safe_print(output)
            total_chars += len(output)
        elif block_type == "bulleted_list_item":
            output = f"{prefix}- {text}"
            safe_print(output)
            total_chars += len(output)
        elif block_type == "numbered_list_item":
            output = f"{prefix}1. {text}"
            safe_print(output)
            total_chars += len(output)
        elif block_type == "paragraph":
            if text:
                output = f"{prefix}{text}"
                safe_print(output)
                total_chars += len(output)
        elif block_type == "child_page":
            # Render child page with special formatting
            output = f"{prefix}**{text}** (Page ID: {block['id']})"
            safe_print(output)
            total_chars += len(output)
        elif block_type == "toggle":
            # Render toggle with special formatting
            output = f"{prefix}<details><summary>{text}</summary>"
            safe_print(output)
            total_chars += len(output)
        
        # Handle nested content
        if block.get("has_children"):
            nested_chars = render_markdown_turbo(block_tree, block["id"], indent + 1)
            total_chars += nested_chars
            
            # Close toggle if it's a toggle block
            if block_type == "toggle":
                close_output = f"{prefix}</details>"
                safe_print(close_output)
                total_chars += len(close_output)
    
    return total_chars

def render_block_to_markdown(block: Dict[str, Any]) -> str:
    """
    Recursively render a single block to markdown.
    Handles nested blocks and rich text.
    """
    if block.get("type") == "child_page":
        page_title = block.get("child_page", {}).get("title", "Untitled Page")
        return f"**{page_title}** (Page ID: {block['id']})"
    elif block.get("type") == "toggle":
        rich_text = block.get("toggle", {}).get("rich_text", [])
        text = "".join([t.get("plain_text", "") for t in rich_text])
        return f"<details><summary>{text}</summary>"
    elif block.get("type") == "paragraph":
        rich_text = block.get("paragraph", {}).get("rich_text", [])
        text = "".join([t.get("plain_text", "") for t in rich_text])
        return text
    elif block.get("type") == "heading_1":
        rich_text = block.get("heading_1", {}).get("rich_text", [])
        text = "".join([t.get("plain_text", "") for t in rich_text])
        return f"# {text}"
    elif block.get("type") == "heading_2":
        rich_text = block.get("heading_2", {}).get("rich_text", [])
        text = "".join([t.get("plain_text", "") for t in rich_text])
        return f"## {text}"
    elif block.get("type") == "heading_3":
        rich_text = block.get("heading_3", {}).get("rich_text", [])
        text = "".join([t.get("plain_text", "") for t in rich_text])
        return f"### {text}"
    elif block.get("type") == "bulleted_list_item":
        rich_text = block.get("bulleted_list_item", {}).get("rich_text", [])
        text = "".join([t.get("plain_text", "") for t in rich_text])
        return f"- {text}"
    elif block.get("type") == "numbered_list_item":
        rich_text = block.get("numbered_list_item", {}).get("rich_text", [])
        text = "".join([t.get("plain_text", "") for t in rich_text])
        return f"1. {text}"
    else:
        return ""

async def fetch_and_render_page(page_id: str):
    """
    Fetch a page and render its markdown output.
    """
    if not NOTION_API_KEY:
        print("Please set NOTION_API_KEY in your .env file.")
        return
    
    print(f"Fetching page: {page_id}")
    start_time = time()
    
    block_tree = await fetch_all_blocks_turbo(page_id)
    fetch_time = time() - start_time
    
    print(f"\n--- Markdown Output (generated in {fetch_time:.3f}s) ---\n")
    render_start_time = time()
    total_chars = render_markdown_turbo(block_tree, page_id)
    render_time = time() - render_start_time
    total_time = time() - start_time
    
    print(f"\n" + "="*60)
    print(f"📊 TURBO PERFORMANCE SUMMARY")
    print(f"="*60)
    print(f"⏱️  Total time: {total_time:.3f} seconds")
    print(f"📄 Total characters: {total_chars:,}")
    print(f"🚀 Fetch time: {fetch_time:.3f} seconds")
    print(f"✍️  Render time: {render_time:.3f} seconds")
    print(f"📈 Characters per second: {total_chars/total_time:,.0f}")
    print(f"="*60)

def get_page_tree_for_context(page_id: str, max_depth: int = 3) -> Dict[str, Any]:
    """
    Get a page and its children as a tree structure for LLM context.
    
    Args:
        page_id: The ID of the page to retrieve
        max_depth: Maximum depth of children to include (default: 3)
    
    Returns:
        Dictionary with page content and children tree
    """
    cached_data = cache.get(page_id)
    if not cached_data:
        return {"error": f"Page {page_id} not found in cache"}
    
    def build_tree(blocks, depth=0):
        if depth > max_depth:
            return []
        
        tree = []
        for block in blocks:
            node = {
                "id": block["id"],
                "type": block["type"],
                "content": render_block_to_markdown(block)
            }
            
            # Add children if they exist and we haven't reached max depth
            if block.get("has_children") and depth < max_depth:
                child_blocks = block.get("children", [])
                if child_blocks:
                    node["children"] = build_tree(child_blocks, depth + 1)
            
            tree.append(node)
        
        return tree
    
    return {
        "page_id": page_id,
        "content": build_tree(cached_data.data.get(page_id, [])),
        "cached_at": cached_data.timestamp.isoformat()
    }

def list_cached_pages_with_tree() -> None:
    """List all cached pages with their complete tree structure."""
    cached_pages = cache.list_cached_pages()
    
    if not cached_pages:
        print("❌ No cached pages found")
        return
    
    print(f"\n📋 CACHED PAGES TREE ({len(cached_pages)} total):")
    print("=" * 60)
    
    for i, page_info in enumerate(cached_pages, 1):
        page_id = page_info["page_id"]
        age_hours = page_info["age_hours"]
        
        print(f"\n{i}. 📄 {page_id} ({age_hours:.1f}h ago)")
        
        # Get tree structure for this page
        tree = get_page_tree_for_context(page_id, max_depth=2)
        if "error" not in tree:
            print(f"   📝 Content blocks: {len(tree['content'])}")
            
            # Show child page references (without fetching them)
            cached_data = cache.get(page_id)
            if cached_data:
                child_pages = get_child_page_references(cached_data.data)
                if child_pages:
                    print(f"   📚 Child pages: {len(child_pages)}")
                    for child in child_pages[:5]:  # Show first 5
                        print(f"      - {child['title']} (ID: {child['id']})")
                    if len(child_pages) > 5:
                        print(f"      ... and {len(child_pages) - 5} more")
        else:
            print(f"   ❌ {tree['error']}")

def show_cache_info():
    """Display cache information."""
    cached_pages = cache.list_cached_pages()
    stats = cache.get_cache_stats()
    
    print(f"\n📊 CACHE INFORMATION")
    print(f"="*50)
    print(f"📁 Cache directory: {stats['cache_dir']}")
    print(f"💾 Memory entries: {stats['memory_entries']}")
    print(f"📄 File entries: {stats['file_entries']}")
    print(f"📦 Total size: {stats['total_size_mb']} MB")
    print(f"\n📋 CACHED PAGES ({len(cached_pages)} total):")
    print(f"-"*50)
    
    if not cached_pages:
        print("No cached pages found.")
    else:
        for i, page_info in enumerate(cached_pages, 1):
            memory_status = "💾" if page_info['in_memory'] else "📄"
            age_str = f"{page_info['age_hours']:.1f}h ago"
            print(f"{i:2d}. {memory_status} {page_info['page_id']} ({age_str})")

async def main_async():
    """Async main function for maximum performance."""
    import sys
    
    if not NOTION_API_KEY:
        print("Please set NOTION_API_KEY in your .env file.")
        return
    
    # Handle command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "cache":
            show_cache_info()
            return
        elif command == "list":
            list_cached_pages_with_tree()
            return
        elif command == "clear":
            if len(sys.argv) > 2 and sys.argv[2] == "all":
                cache.clear_cache()
                print("🗑️ All cache cleared")
            else:
                cache.clear_old()
                print("🗑️ Old cache entries cleared")
            return
        elif command == "tree":
            if len(sys.argv) > 2:
                page_id = sys.argv[2]
                tree = get_page_tree_for_context(page_id)
                if "error" not in tree:
                    print(f"\n🌳 TREE STRUCTURE FOR PAGE: {page_id}")
                    print("=" * 50)
                    print(json.dumps(tree, indent=2))
                else:
                    print(f"❌ {tree['error']}")
            else:
                print("❌ Please provide a page ID: python turbo_notion_fetch.py tree <page_id>")
            return
        elif command == "fetch":
            if len(sys.argv) > 2:
                page_id = sys.argv[2]
                await fetch_and_render_page(page_id)
            else:
                print("❌ Please provide a page ID: python turbo_notion_fetch.py fetch <page_id>")
            return
        elif command == "fetch-children":
            if len(sys.argv) > 2:
                page_id = sys.argv[2]
                await fetch_all_children(page_id)
            else:
                print("❌ Please provide a page ID: python turbo_notion_fetch.py fetch-children <page_id>")
            return
        elif command == "help":
            print("""
🔧 TURBO NOTION FETCH - COMMANDS
================================

python turbo_notion_fetch.py                    # Fetch main page (with all children)
python turbo_notion_fetch.py <page_id>          # Fetch specific page (with all children)
python turbo_notion_fetch.py fetch <page_id>    # Fetch specific page (explicit)
python turbo_notion_fetch.py fetch-children <page_id>  # Fetch all children of a page
python turbo_notion_fetch.py cache              # Show cache info
python turbo_notion_fetch.py list               # List cached pages with tree
python turbo_notion_fetch.py tree <page_id>     # Show tree structure for page
python turbo_notion_fetch.py clear              # Clear old cache entries
python turbo_notion_fetch.py clear all          # Clear all cache
python turbo_notion_fetch.py help               # Show this help
            """)
            return
        else:
            # Treat as page ID
            page_id = sys.argv[1]
            await fetch_and_render_page(page_id)
            return
    
    # Default behavior - fetch HOME_PAGE_ID
    if not HOME_PAGE_ID:
        print("Please set NOTION_HOME_PAGE_ID in your .env file.")
        return
    
    await fetch_and_render_page(HOME_PAGE_ID)

def main():
    """Main function with async execution."""
    run(main_async())

if __name__ == "__main__":
    main() 