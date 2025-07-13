# Notion RAG CLI

A streamlined Retrieval-Augmented Generation (RAG) system for Notion that fetches content, loads it into a vector database, and provides fast semantic search.

## 🚀 Quick Start

### 1. Setup Environment
Create a `.env` file:
```env
NOTION_API_KEY=your_notion_api_key_here
NOTION_HOME_PAGE_ID=your_notion_home_page_id_here
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Complete Workflow
```bash
# Fetch, load, and start interactive search
python notion_rag_workflow.py

# Or with a specific page ID
python notion_rag_workflow.py <page_id>

# Or with a specific search query
python notion_rag_workflow.py <page_id> "your search query"
```

## 📁 Essential Files

- **`notion_rag_workflow.py`** - Main workflow script (fetch → load → search)
- **`performance_test.py`** - Performance testing and benchmarking
- **`search_notion_vector.py`** - Standalone search tool
- **`turbo_notion_fetch.py`** - High-performance Notion content fetcher
- **`notion_rag/`** - Core RAG system modules

## 🔧 Usage Examples

### Complete Workflow
```bash
# Interactive mode (fetch + load + search)
python notion_rag_workflow.py

# Single search query
python notion_rag_workflow.py "cybersecurity best practices"
```

### Performance Testing
```bash
# Test complete workflow with timing
python performance_test.py

# Test with specific page
python performance_test.py <page_id>
```

### Standalone Search
```bash
# Interactive search (requires existing vector database)
python search_notion_vector.py

# Single search query
python search_notion_vector.py "your query"
```

## 🎯 Features

- ✅ **Recursive Fetching**: Automatically fetches main page + all child pages
- ✅ **Vector Database**: ChromaDB with semantic search
- ✅ **Fast Search**: Sub-second query response times
- ✅ **Content Chunking**: Intelligent text segmentation
- ✅ **Metadata Preservation**: Page IDs, URLs, titles, etc.
- ✅ **Performance Monitoring**: Built-in timing and statistics

## 📊 Performance

Typical performance metrics:
- **Fetch Time**: ~0.8s for 9 pages
- **Load Time**: ~14s for 54K characters
- **Search Time**: ~1.4s average per query
- **Processing Rate**: ~4K chars/sec

## 🔍 Search Examples

```bash
# Search for specific topics
python notion_rag_workflow.py "security controls"

# Search for processes
python notion_rag_workflow.py "incident response"

# Search for concepts
python notion_rag_workflow.py "threat modeling"
```

## 🚀 Next Steps

The system is ready for LLM integration (RAG):
1. Use search results as context for LLM
2. Implement response generation
3. Add conversation memory
4. Deploy as a chatbot

## 📝 License

MIT License - see LICENSE file for details. 