import asyncio
import os
import time
import json
import hashlib
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
from dotenv import load_dotenv
from notion_client import AsyncClient
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading

load_dotenv()

NOTION_API_KEY = os.getenv("NOTION_API_KEY")
HOME_PAGE_ID = os.getenv("NOTION_HOME_PAGE_ID")

# Turbo configuration - optimized for speed
MAX_CONCURRENT_REQUESTS = 30  # Conservative but effective
BATCH_SIZE = 15  # Optimal batch size
CACHE_DURATION_HOURS = 168  # 1 week cache
CACHE_DIR = Path("./cache")
ENABLE_CACHE = True

# Only fetch the most essential block types
TURBO_BLOCK_TYPES = {
    "paragraph", "heading_1", "heading_2", "heading_3", 
    "bulleted_list_item", "numbered_list_item"
    # Removed toggle and child_page for maximum speed
}

@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    data: Dict[str, Any]
    timestamp: datetime
    page_id: str
    etag: Optional[str] = None

class TurboCache:
    """Ultra-fast cache with memory-first approach."""
    
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self._memory_cache = {}
        self._lock = threading.Lock()
    
    def _get_cache_key(self, page_id: str) -> str:
        return hashlib.md5(f"notion_page_{page_id}".encode()).hexdigest()
    
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
        
        # Check file cache
        cache_path = self._get_cache_path(page_id)
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
            
            timestamp = datetime.fromisoformat(data['timestamp'])
            if datetime.now() - timestamp > timedelta(hours=CACHE_DURATION_HOURS):
                cache_path.unlink()
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
                json.dump(cache_entry, f, indent=2)
        except Exception as e:
            print(f"Cache write error: {e}")

class TurboRateLimiter:
    """Intelligent rate limiter that adapts to API performance."""
    
    def __init__(self, max_requests: int = 30, time_window: float = 1.0):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_requests)
        self.error_count = 0
        self.success_count = 0
    
    async def acquire(self):
        """Intelligent rate limiting."""
        await self._semaphore.acquire()
        async with self._lock:
            now = time.time()
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
    return await asyncio.gather(*tasks, return_exceptions=True)

async def fetch_all_blocks_turbo(root_id: str) -> Dict[str, Any]:
    """Turbo block fetching with maximum speed optimizations."""
    # Check cache first
    cached_data = cache.get(root_id)
    if cached_data:
        print(f"⚡ Using cached data (instant)")
        return cached_data.data
    
    print(f"🚀 Starting turbo fetch...")
    start_time = time.time()
    
    # Create client
    client = AsyncClient(auth=NOTION_API_KEY)
    
    queue = [root_id]
    block_tree = {}
    visited = set()
    total_blocks_processed = 0
    
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
            
            # Only add blocks with children to queue
            for child in children:
                if child.get("has_children") and child["id"] not in visited:
                    new_queue.append(child["id"])
        
        queue.extend(new_queue)
        
        # Progress update
        elapsed = time.time() - start_time
        if elapsed > 0:
            rate = total_blocks_processed / elapsed
            print(f"⚡ Processed {total_blocks_processed} blocks at {rate:.0f} blocks/sec")
    
    # Cache the results
    cache.set(root_id, block_tree)
    
    total_time = time.time() - start_time
    print(f"✅ Completed in {total_time:.2f}s! Total blocks: {total_blocks_processed}")
    
    return block_tree

def render_markdown_turbo(block_tree, root_id, indent=0):
    """Turbo markdown rendering with minimal overhead."""
    blocks = block_tree.get(root_id, [])
    
    # Pre-process all blocks for maximum speed
    processed_blocks = []
    for block in blocks:
        block_type = block.get("type")
        if block_type not in TURBO_BLOCK_TYPES:
            continue
            
        rich_text = block.get(block_type, {}).get("rich_text", [])
        text = "".join([t.get("plain_text", "") for t in rich_text])
        processed_blocks.append((block_type, text, block))
    
    # Render blocks efficiently and track character count
    total_chars = 0
    for block_type, text, block in processed_blocks:
        prefix = "  " * indent
        
        if block_type == "heading_1":
            output = f"{prefix}# {text}"
            print(output)
            total_chars += len(output)
        elif block_type == "heading_2":
            output = f"{prefix}## {text}"
            print(output)
            total_chars += len(output)
        elif block_type == "heading_3":
            output = f"{prefix}### {text}"
            print(output)
            total_chars += len(output)
        elif block_type == "bulleted_list_item":
            output = f"{prefix}- {text}"
            print(output)
            total_chars += len(output)
        elif block_type == "numbered_list_item":
            output = f"{prefix}1. {text}"
            print(output)
            total_chars += len(output)
        elif block_type == "paragraph":
            if text:
                output = f"{prefix}{text}"
                print(output)
                total_chars += len(output)
        
        # Handle nested content
        if block.get("has_children"):
            nested_chars = render_markdown_turbo(block_tree, block["id"], indent + 1)
            total_chars += nested_chars
    
    return total_chars

async def main_async():
    """Async main function for maximum performance."""
    if not NOTION_API_KEY or not HOME_PAGE_ID:
        print("Please set NOTION_API_KEY and NOTION_HOME_PAGE_ID in your .env file.")
        return
    
    print(f"🚀 Turbo Notion fetch for page: {HOME_PAGE_ID}")
    print(f"Cache enabled: {ENABLE_CACHE}")
    print(f"Max concurrent requests: {MAX_CONCURRENT_REQUESTS}")
    print(f"Batch size: {BATCH_SIZE}")
    
    start_time = time.time()
    block_tree = await fetch_all_blocks_turbo(HOME_PAGE_ID)
    fetch_time = time.time() - start_time
    
    print(f"\n--- Markdown Output (generated in {fetch_time:.3f}s) ---\n")
    render_start_time = time.time()
    total_chars = render_markdown_turbo(block_tree, HOME_PAGE_ID)
    render_time = time.time() - render_start_time
    total_time = time.time() - start_time
    
    print(f"\n" + "="*60)
    print(f"📊 TURBO PERFORMANCE SUMMARY")
    print(f"="*60)
    print(f"⏱️  Total time: {total_time:.3f} seconds")
    print(f"📄 Total characters: {total_chars:,}")
    print(f"🚀 Fetch time: {fetch_time:.3f} seconds")
    print(f"✍️  Render time: {render_time:.3f} seconds")
    print(f"📈 Characters per second: {total_chars/total_time:,.0f}")
    print(f"="*60)

def main():
    """Main function with async execution."""
    asyncio.run(main_async())

if __name__ == "__main__":
    main() 