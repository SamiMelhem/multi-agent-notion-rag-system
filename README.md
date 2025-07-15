# Notion RAG CLI

A comprehensive Retrieval-Augmented Generation (RAG) system for Notion that fetches content, loads it into a vector database, and provides intelligent search with Gemini 2.5 Flash-Lite Preview.

## 🚀 Quick Start

### 1. Setup Environment
Create a `.env` file:
```env
NOTION_API_KEY=your_notion_api_key_here
NOTION_HOME_PAGE_ID=your_notion_home_page_id_here
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_CLOUD_PROJECT=your_google_cloud_project_id
```

### 2. Install Dependencies
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Complete Workflow
```bash
# Interactive RAG system with Gemini
python search_notion_vector.py

# Complete workflow (fetch + load + search)
python notion_rag_workflow.py

# Or with a specific page ID
python notion_rag_workflow.py <page_id>

# Or with a specific search query
python notion_rag_workflow.py <page_id> "your search query"
```

## 📁 Core Files

- **`search_notion_vector.py`** - Interactive RAG system with Gemini 2.5 Flash-Lite Preview
- **`notion_rag_workflow.py`** - Complete workflow script (fetch → load → search)
- **`turbo_notion_fetch.py`** - High-performance Notion content fetcher
- **`notion_rag/`** - Core RAG system modules
- **`SYSTEM_DOCUMENTATION.md`** - Detailed system architecture and API documentation

## 🎯 Features

- ✅ **Gemini 2.5 Flash-Lite Preview Integration**: Advanced LLM for intelligent responses
- ✅ **Cost Tracking**: Real-time token counting and cost estimation
- ✅ **Prompt Engineering**: Multiple specialized templates (QA, summary, analysis, extraction)
- ✅ **Recursive Fetching**: Automatically fetches main page + all child pages
- ✅ **Vector Database**: ChromaDB with semantic search
- ✅ **Fast Search**: Sub-second query response times
- ✅ **Content Chunking**: Intelligent text segmentation
- ✅ **Metadata Preservation**: Page IDs, URLs, titles, etc.
- ✅ **Performance Monitoring**: Built-in timing and statistics
- ✅ **Secure API Management**: Keyring-based credential storage
- ✅ **Interactive Chat**: Natural language querying with context awareness

## 🔧 Usage Examples

### Interactive RAG System (Recommended)
```bash
# Start interactive session with Gemini
python search_notion_vector.py

# Available commands when your about to search:
# - help: Show available commands
# - stats: Show database statistics  
# - costs: Show cost summary
# - templates: Show available prompt templates
# - 'summarize: [question]' - Use summarization template
# - 'analyze: [question]' - Use analysis template
# - 'extract: [question]' - Use extraction template
# - 'bullet: [question]' - Use bullet-point template
```

### Complete Workflow
```bash
# Interactive mode (fetch + load + search)
python notion_rag_workflow.py

# Single search query
python notion_rag_workflow.py "cybersecurity best practices"

# With specific page
python notion_rag_workflow.py <page_id> "incident response procedures"
```

### CLI Commands
```bash
# Initialize RAG system
python -m notion_rag.cli init --database-id <id>

# Test Gemini connection
python -m notion_rag.cli gemini-test

# RAG query with specific collection
python -m notion_rag.cli rag-query --collection-name notion_documents --query "your question"

# Interactive RAG chat
python -m notion_rag.cli rag-chat --collection-name notion_documents

# Search documents
python -m notion_rag.cli search "your search query"

# Manage collections
python -m notion_rag.cli collections
python -m notion_rag.cli collection-info --collection-name notion_documents
```

## 💰 Cost Optimization

The system includes comprehensive cost tracking:

- **Token Counting**: Accurate input/output token estimation
- **Cost Estimation**: Pre-call cost calculation using Gemini pricing
- **Cost Logging**: Persistent cost tracking in `cost_log.json`
- **Cost Summary**: Detailed breakdown of input/output costs

```bash
# View cost summary in interactive mode
costs

# Check cost log file
cat cost_log.json
```

## 🎨 Prompt Templates

The system includes specialized prompt templates:

- **`rag_qa`**: Standard question answering
- **`rag_summary`**: Detailed summarization
- **`rag_analysis`**: Comprehensive analysis
- **`rag_extraction`**: Key points extraction

```bash
# Use different templates in interactive mode
summarize: What is cybersecurity?
analyze: What are the main security threats?
extract: Key points about incident response
bullet: List security best practices
```

## 📊 Performance

Typical performance metrics:
- **Fetch Time**: ~0.8s for 9 pages
- **Load Time**: ~14s for 54K characters
- **Search Time**: ~1.4s average per query
- **Processing Rate**: ~4K chars/sec
- **Token Estimation**: Real-time with cost breakdown

## 🔍 Search Examples

```bash
# Search for specific topics
python search_notion_vector.py
> What are the security controls for data protection?

# Use specialized templates
> summarize: What is the incident response process?
> analyze: What are the main cybersecurity threats?
> extract: Key points about threat modeling
> bullet: List security best practices

# Check costs and stats
> costs
> stats
```

## 🚀 Advanced Features

### Cost Tracking
- Real-time token counting with `tiktoken`
- Pre-call cost estimation
- Persistent cost logging
- Detailed cost breakdowns

### Prompt Engineering
- Multiple specialized templates
- Context-aware prompting
- Chain-of-thought reasoning
- Structured output support

### Vector Search
- Semantic similarity search
- Metadata filtering
- Batch processing
- Collection management

## 📝 Environment Variables

Required environment variables:
- `NOTION_API_KEY`: Your Notion API key
- `NOTION_HOME_PAGE_ID`: Your Notion home page ID
- `GEMINI_API_KEY`: Your Gemini API key
- `GOOGLE_CLOUD_PROJECT`: Your Google Cloud project ID

## 🔧 Development

### Project Structure
```
notion-rag-cli/
├── notion_rag/           # Core RAG modules
│   ├── cli.py           # Command-line interface
│   ├── config.py        # Configuration management
│   ├── gemini_client.py # Gemini 2.5 Flash-Lite integration
│   ├── vector_store.py  # ChromaDB management
│   ├── embeddings.py    # Embedding generation
│   ├── chunking.py      # Text chunking
│   ├── cost_tracker.py  # Cost tracking
│   └── prompt_utils.py  # Prompt engineering
├── search_notion_vector.py    # Interactive RAG system
├── notion_rag_workflow.py     # Complete workflow
├── turbo_notion_fetch.py      # Notion content fetcher
└── requirements.txt           # Dependencies
```

### Testing
```bash
# Run tests
python -m pytest tests/ -v

# Test specific modules
python -m pytest tests/test_gemini_client.py -v
python -m pytest tests/test_vector_store.py -v
```

## 📄 License

MIT License - see LICENSE file for details. 